from PIL import Image
from tqdm import tqdm
from typing import Union, Optional, List, Any, Dict
import jittor as jt
import jittor.transform as transform
from JDiffusion import StableDiffusionPipeline
from JDiffusion.pipelines.pipeline_stable_diffusion_jittor import (
    retrieve_timesteps, 
    rescale_noise_cfg,
    adjust_lora_scale_text_encoder,
    scale_lora_layers,
    unscale_lora_layers,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    USE_PEFT_BACKEND,
    logger
)
from transformers import CLIPTokenizer

from modules import (
    indexes, 
    feature_blender,
    structure_injector, 
    appearance_injector, 
    styleblend_cross_attention, 
    styleblend_self_attention,
)
from utils import jtvar_to_pil, CustomizedStableDiffusionOutput, Handler
from utils.clip_lora import LoRACLIPTextModel


class CustomizedStableDiffusionPipeline(StableDiffusionPipeline):
    @jt.no_grad()
    def image2latent(self, image: Union[Image.Image, jt.Var]):
        # process a single PIL image or a batch of jt images.
        if isinstance(image, Image.Image):
            image = jt.array(transform.ToTensor()(image))
            image = (image - 0.5) / 0.5
            
        if image.ndim == 3:
            image = image.unsqueeze(0)
            
        latent = self.vae.encode(image).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent

    @jt.no_grad()
    def latent2image(self, latents):
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = (images / 2 + .5).clamp(0, 1)
        return images

    @jt.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        seed: Optional[Union[int, List[int]]] = None,
        latents: Optional[jt.Var] = None,
        prompt_embeds: Optional[jt.Var] = None,
        negative_prompt_embeds: Optional[jt.Var] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        output_type: str = 'pil',
    ) -> CustomizedStableDiffusionOutput:
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks
        # print(height, width)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = jt.concat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            seed,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = jt.array(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        
        pred_z0_list, latents_list, noises_list = [latents], [latents], []
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling')):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = jt.concat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                return_dict=False,
            )[0]

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
            latents, pred_z0 = output.prev_sample, output.pred_original_sample
            latents_list.append(latents)
            pred_z0_list.append(pred_z0)
            noises_list.append(noise_pred)

        images = self.latent2image(latents)
        if output_type == 'pil':
            images = jtvar_to_pil(images)

        return CustomizedStableDiffusionOutput(images=images)



class SDEditPipeline(CustomizedStableDiffusionPipeline):
    @jt.no_grad()
    def __call__(
        self,
        images,
        noises=None,
        strength: float = 0.8,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        seed: Optional[Union[int, List[int]]] = None,
        latents: Optional[jt.Var] = None,
        prompt_embeds: Optional[jt.Var] = None,
        negative_prompt_embeds: Optional[jt.Var] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        output_type: str = 'pil',
        **kwargs
    ) -> CustomizedStableDiffusionOutput:
        if isinstance(images, Image.Image):
            images = [images]
        assert isinstance(images, list)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = jt.concat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
        start_timestep = num_inference_steps - int(strength * num_inference_steps)
        timesteps = timesteps[start_timestep:]
        start_timestep = timesteps[0:1]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            seed,
            latents,
        )
        # prepare latents
        latents_gts = [self.image2latent(image) for image in images]
        latents_gt = jt.concat(latents_gts)
        noises = jt.randn_like(latents_gt) if noises is None else noises
        latents = self.scheduler.add_noise(latents_gt, noises, start_timestep)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = jt.array(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)

        for i, t in enumerate(tqdm(timesteps, desc='SDEdit Sampling')):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = jt.concat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
            latents, pred_z0 = output.prev_sample, output.pred_original_sample
            
        images = self.latent2image(latents)
        if output_type == 'pil':
            images = jtvar_to_pil(images)
        return CustomizedStableDiffusionOutput(images=images)


class StyleBlendT2IPipeline(CustomizedStableDiffusionPipeline):
    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[jt.Var] = None,
        negative_prompt_embeds: Optional[jt.Var] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        tokenizer=None,
        text_encoder=None,
    ):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if text_encoder is None:
            text_encoder = self.text_encoder
            
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder, lora_scale)
            else:
                scale_lora_layers(text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not jt.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = text_encoder(
                    text_input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds
    
    @jt.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,  # "A ...", inference pipeline will expand "in the style of <C>/<F>." to prompt automatically.
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        latents: Optional[jt.Var] = None,
        prompt_embeds: Optional[jt.Var] = None,
        negative_prompt_embeds: Optional[jt.Var] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        output_type: Optional[str] = "pil",
        seed: int = None,
        **kwargs,
    ):
        assert hasattr(self, 'handler')
        assert hasattr(self, 'text_encoder_coarse_style')
        assert hasattr(self, 'text_encoder_fine_style')
        assert hasattr(self, 'tokenizer_coarse_style')
        assert hasattr(self, 'tokenizer_fine_style')
        assert hasattr(self, 'placeholder_coarse_style')
        assert hasattr(self, 'placeholder_fine_style')

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            prompt = [prompt]
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        coarse_style_prompt = [p + f' in the style of {self.placeholder_coarse_style}.' for p in prompt]
        fine_style_prompt = [p + f' in the style of {self.placeholder_fine_style}.' for p in prompt]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        # fine style prompt
        fine_style_prompt_embeds, fine_style_neg_prompt_embeds = self.encode_prompt(
            fine_style_prompt,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            tokenizer=self.tokenizer_fine_style,
            text_encoder=self.text_encoder_fine_style,
        )
        # coarse style prompt
        coarse_style_prompt_embeds, coarse_style_neg_prompt_embeds = self.encode_prompt(
            coarse_style_prompt,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            [''],
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            tokenizer=self.tokenizer_coarse_style,
            text_encoder=self.text_encoder_coarse_style,
        )  
        
        if self.do_classifier_free_guidance:
            prompt_embeds = jt.concat([coarse_style_neg_prompt_embeds, fine_style_neg_prompt_embeds, coarse_style_prompt_embeds, fine_style_prompt_embeds])
        else:
            prompt_embeds = jt.concat([coarse_style_prompt_embeds, fine_style_prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            seed,
            latents,
        )
        
        latents = jt.concat([latents]*2)  # for CSB and FSB respectively.
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = jt.array(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        self.handler.reset_step()
        
        pbar = tqdm(range(len(timesteps)), desc='Style Blend')
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = jt.concat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    
            self.handler.update_step()
            pbar.update()

        self.handler.reset_step()
        
        images = self.latent2image(latents)
        if output_type == 'pil':
            images = jtvar_to_pil(images)
            
        return CustomizedStableDiffusionOutput(images=images)

    def load_styleblend_weights(
        self,
        te_lora_path,
        unet_lora_path,
        coarse_style_embeds_path,
        fine_style_embeds_path,
        pretrained_sd_model_path,
        placeholder_coarse_style='<C>',
        placeholder_fine_style='<F>',
    ):
        self.placeholder_coarse_style = placeholder_coarse_style
        self.placeholder_fine_style = placeholder_fine_style
        
        self.load_lora_weights(unet_lora_path)
        
        self.tokenizer_fine_style = self.tokenizer
        self.text_encoder_fine_style = self.text_encoder
        
        self.text_encoder_coarse_style = LoRACLIPTextModel.from_pretrained(pretrained_sd_model_path, subfolder='text_encoder')
        self.tokenizer_coarse_style = CLIPTokenizer.from_pretrained(pretrained_sd_model_path, subfolder='tokenizer')
        self.text_encoder_coarse_style.load_lora_weights(te_lora_path)
        
        def register_embeddings(tokenizer, text_encoder, custom_embeds):
            placeholder_tokens = list(custom_embeds.keys())
            
            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            if num_added_tokens != len(placeholder_tokens):
                raise ValueError('The tokenizer already contains some of the tokens. Please pass different placeholder tokens.')
            
            placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            
            with jt.no_grad():
                for new_token_id, token in zip(placeholder_token_ids, placeholder_tokens):
                    token_embeds[new_token_id] = custom_embeds[token].clone()

        custom_coarse_style_embeds = jt.load(coarse_style_embeds_path)
        register_embeddings(tokenizer=self.tokenizer_coarse_style, text_encoder=self.text_encoder_coarse_style, custom_embeds=custom_coarse_style_embeds)
        custom_fine_style_embeds = jt.load(fine_style_embeds_path)
        register_embeddings(tokenizer=self.tokenizer_fine_style, text_encoder=self.text_encoder_fine_style, custom_embeds=custom_fine_style_embeds)
        
    def set_coarse_style_unet_lora_scale(self, scale=0.3):
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'Attention' and (name.endswith('attn2') or name.endswith('attn1')):
                if hasattr(module, '_scale'):
                    module._scale = scale
        print(f'Set CSB unet lora scale to {scale}.')

    def register_styleblend_modules(
        self,
        c2f_cross_attn_layers_to_register=None,
        f2c_cross_attn_layers_to_register=None,
        c2f_self_attn_layers_to_register=None,
        f2c_self_attn_layers_to_register=None,
        scale=0.3,
        num_inference_timesteps=30,
        c2f_step_ratio=0.8,
        f2c_step_ratio=0.6,
    ):
        if c2f_cross_attn_layers_to_register is None:
            c2f_cross_attn_layers_to_register = []
        if f2c_cross_attn_layers_to_register is None:
            f2c_cross_attn_layers_to_register = []
        if c2f_self_attn_layers_to_register is None:
            # In SA c2f direction, we inject the structure features. This configuration is fine to most style cases.
            c2f_self_attn_layers_to_register = [4, 5, 6, 7, 8, 9]
        if f2c_self_attn_layers_to_register is None:
            # In SA f2c direction, we inject the appearance features. This configuration is fine to most style cases.
            f2c_self_attn_layers_to_register = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15] 
        
        handler = Handler(total_steps=num_inference_timesteps)
        self.handler = handler
        
        step_ratio = 1.
        steps_to_blend = []
        
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'Attention' and name.endswith('attn2'):
                injector = None
                if indexes[name] in c2f_cross_attn_layers_to_register:
                    injector = structure_injector(direction='c2f')
                elif indexes[name] in f2c_cross_attn_layers_to_register:
                    injector = structure_injector(direction='f2c')

                module.__class__ = styleblend_cross_attention(
                    block_class=module.__class__, 
                    csb_lora_scale=scale, 
                    feature_injector=injector, 
                    handler=handler,
                    steps_to_blend=[i for i in range(int(step_ratio*num_inference_timesteps))],
                )
                
            elif module.__class__.__name__ == 'Attention' and name.endswith('attn1'):
                injector = None
                if indexes[name] in c2f_self_attn_layers_to_register and indexes[name] in f2c_self_attn_layers_to_register:
                    injector = feature_blender()
                    step_ratio = c2f_step_ratio
                    steps_to_blend=[i for i in range(int(step_ratio*num_inference_timesteps))]
                
                elif indexes[name] in c2f_self_attn_layers_to_register:
                    injector = structure_injector(direction='c2f')  # q inject
                    step_ratio = c2f_step_ratio
                    steps_to_blend=[i for i in range(int(step_ratio*num_inference_timesteps))]
                elif indexes[name] in f2c_self_attn_layers_to_register:
                    injector = appearance_injector(direction='f2c')  # kv inject
                    step_ratio = f2c_step_ratio
                    steps_to_blend=[i for i in range(int(step_ratio*num_inference_timesteps))]
                    
                module.__class__ = styleblend_self_attention(
                    block_class=module.__class__, 
                    csb_lora_scale=scale, 
                    feature_injector=injector,
                    handler=handler,
                    steps_to_blend=steps_to_blend,
                )
                
    def unregister_styleblend_modules(self):
        if not hasattr(self, 'handler'):
            return
        
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == 'Attention' and name.endswith('attn2'):
                module.__class__ = styleblend_cross_attention(block_class=module.__class__, scale=1., feature_injector=None, handler=self.handler)
            elif module.__class__.__name__ == 'Attention' and name.endswith('attn1'):
                module.__class__ = styleblend_self_attention(block_class=module.__class__, scale=1., feature_injector=None, handler=self.handler)
