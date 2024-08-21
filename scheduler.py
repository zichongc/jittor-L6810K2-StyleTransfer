import jittor as jt
from typing import Optional, Union, List
import numpy as np
import math
import json
from dataclasses import dataclass


@dataclass
class DDIMSchedulerOutput:
    prev_sample: jt.Var
    pred_original_sample: jt.Var


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return jt.array(betas, dtype=jt.float32)


def rescale_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = jt.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = jt.concat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class DDIMScheduler:
    
    @staticmethod
    def from_config(path):
        with open(path, 'r') as file:
            kwargs = json.load(file)
        return DDIMScheduler(**kwargs)

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        steps_offset: int = 1,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        skip_prk_steps: bool = True,
        **kwargs
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.clip_sample_range = clip_sample_range
        self.sample_max_value = sample_max_value
        self.timestep_spacing = timestep_spacing
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.skip_prk_steps = skip_prk_steps

        if self.trained_betas is not None:
            self.betas = jt.Var(self.trained_betas).to(jt.float32)
        elif self.beta_schedule == "linear":
            self.betas = jt.linspace(self.beta_start, self.beta_end, self.num_train_timesteps).to(jt.float32)
        elif self.beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = jt.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps).to(jt.float32) ** 2
        elif self.beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(self.num_train_timesteps)
        else:
            raise NotImplementedError(f"{self.beta_schedule} does is not implemented for {self.__class__}")
        
        # Rescale for zero SNR
        if self.rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jt.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = jt.array(1.0) if self.set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = jt.array(np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64))

    def scale_model_input(self, sample: jt.Var, timestep=None) -> jt.Var:
        return sample
    
    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    
    def set_timesteps(self, num_inference_steps: int, device=None):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
        
        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
            )
        elif self.timestep_spacing == 'leading':
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1 
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = jt.array(timesteps)

    def step(
        self,
        model_output: jt.Var,
        timestep: int,
        sample: jt.Var,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        return_dict: bool = True,
    ):
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
        
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )
        
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (prev_sample, pred_original_sample,)
        
        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: jt.Var,
        noise: jt.Var,
        timesteps: jt.Var
    ) -> jt.Var:
        alphas_cumprod = self.alphas_cumprod.to(original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


@dataclass
class DDPMSchedulerOutput:
    prev_sample: jt.Var
    pred_original_sample: Optional[jt.Var] = None
    

class DDPMScheduler():
    
    @staticmethod
    def from_config(path):
        with open(path, mode='r') as file:
            kwargs = json.load(fp=file)
        return DDPMScheduler(**kwargs)
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: int = False,
        **kwargs
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.clip_sample_range = clip_sample_range
        self.sample_max_value = sample_max_value 
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

        # print(self.num_train_timesteps)
        # print(self.clip_sample)
        # print(prediction_type)
        # print(self.thresholding)
        # print(beta_start, beta_end)
        
        if trained_betas is not None:
            self.betas = jt.Var(trained_betas).to(jt.float32)
        elif beta_schedule == "linear":
            self.betas = jt.linspace(beta_start, beta_end, num_train_timesteps).to(jt.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = jt.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps).to(jt.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = jt.linspace(-6, 6, num_train_timesteps)
            self.betas = jt.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jt.cumprod(self.alphas, dim=0)
        self.one = jt.array(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = jt.array(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type 
        
    def scale_model_input(self, sample: jt.Var, timestep: Optional[int] = None) -> jt.Var:
        return sample
    
    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
    ):
        self.num_inference_steps = num_inference_steps
        self.custom_timesteps = False
        
        if self.timestep_spacing == "linspace":
            timesteps = (np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64))
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
        
        self.timesteps = jt.array(timesteps)
        
    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self.previous_timestep(t)
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = jt.clamp(variance, min=1e-20)
        
        if variance_type is None:
            variance_type = self.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        elif variance_type == "fixed_small_log":
            variance = jt.log(variance)
            variance = jt.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = jt.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = jt.log(variance)
            max_log = jt.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance
        
    def _threshold_sample(self, sample: jt.Var) -> jt.Var:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (jt.float32, jt.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = jt.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = jt.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = jt.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample
    
    def step(
        self,
        model_output: jt.Var,
        timestep: int,
        sample: jt.Var,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, tuple]:
        t = timestep
        
        prev_t = self.previous_timestep(t)
        
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = jt.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None
        
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
            
        # 3. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )
            
        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = jt.randn(model_output.shape, requires_grad=False)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = jt.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        
        if not return_dict:
            return (pred_prev_sample, )
        
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
        
    def add_noise(
        self,
        original_samples: jt.Var,
        noise: jt.Var,
        timesteps: jt.Var,
    ) -> jt.Var:
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def __len__(self):
        return self.num_train_timesteps
    
    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = jt.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
            )
            prev_t = timestep - self.num_train_timesteps // num_inference_steps

        return prev_t
    
    def get_velocity(
        self, sample: jt.Var, noise: jt.Var, timesteps: jt.Var
    ) -> jt.Var:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
