from utils.clip_lora import LoRACLIPTextModel
import os
import argparse
from omegaconf import OmegaConf
import random
import math
from tqdm import tqdm
from PIL import Image
import jittor as jt
from jittor.lr_scheduler import StepLR
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import Dataset, DataLoader

from pipeline import SDEditPipeline, CustomizedStableDiffusionPipeline
from scheduler import DDIMScheduler, DDPMScheduler
from utils.dataset import ImageFilter, CoarseStyleTrainingDataset, cst_collate_fn
from utils import customize_token_embeddings, save_embeddings



class AssistantDataset(Dataset):
    def __init__(self, style: str, data_dir: str, resolution=768):
        self.style = style
        self.resolution = resolution
        self.image_dir = os.path.join(data_dir, style, 'images')
        self.class_prompts, self.images, self.filenames = self.load_data()
        
    def load_data(self):
        files = os.listdir(self.image_dir)
        
        images = []
        class_prompts = []
        filenames = []
        for file in files:
            image = Image.open(os.path.join(self.image_dir, file)).convert('RGB').resize((self.resolution, self.resolution))
            class_prompt = file.split('.')[0].replace('_', ' ').lower()  # create text prompt harnessing the given image info.
            class_prompts.append(class_prompt)
            images.append(image)
            filenames.append(file)

        return class_prompts, images, filenames
    
    @staticmethod
    def aan(string: str):
        return 'an' if string[0].lower() in 'aeiou' else 'a'
    
    def __len__(self):
        return len(self.class_prompts)
    
    def __getitem__(self, index):
        prompt = self.aan(self.class_prompts[index]).title() + ' ' + self.class_prompts[index] + ', realistic.'
        return prompt, self.images[index], self.filenames[index]
    

class PlausibleDataCreator:
    """
    Generate a set of plausible images using SDEdit to create a dataset for coarse style learning.
    """
    def __init__(
        self,
        pretrained_sd_model_path,
        pretrained_clip_model=None,
        k=10,
        select_n=10,
        resolution=768,
        **kwargs,
    ):
        self.sd_pipe = SDEditPipeline.from_pretrained(pretrained_sd_model_path, dtype=jt.float32)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(os.path.join(pretrained_sd_model_path, 'scheduler/scheduler_config.json'))
        if pretrained_clip_model is not None:
            self.image_filter = ImageFilter(pretrained_clip_model)
            
        self.k = k
        self.select_n = select_n
        self.resolution = resolution

    def generate(self, style, source_dir, output_dir):
        """
        Examples:
        ```python
        >>> creator = PlausibleDataCreator(pretrained_sd_model_path, pretrained_clip_model)
        >>> creator.generate('00', './data/B', './data/plausibleB')
        ```"""
        os.makedirs(os.path.join(output_dir, style))
        dataset = AssistantDataset(style, source_dir, resolution=self.resolution)
        
        for prompt, image, filename in dataset:
            print(filename, prompt)
            candidates = []
            strengths = [random.choice([0.85, 0.8, 0.75]) for _ in range(self.k)]
            print('strengths', strengths)
            
            for s in strengths:
                candidate = self.sd_pipe(
                    images=[image],
                    prompt=prompt, 
                    strength=s,
                    num_inference_steps=20,
                    negative_prompt='ugly, low quality, blur, distorted'
                ).images[0]
                candidates.append(candidate)
             
            if self.image_filter is not None:
                candidates = self.image_filter(image, candidates, select_n=self.select_n)
                    
            for i, img in enumerate(candidates):
                img.save(os.path.join(output_dir, style, filename[:-4]+f'-{i:03d}.png'))
            
        print(f'Generated images saved to {output_dir}.')


def create_plausible_dataset(cfg, style):
    sample_dir = cfg.sample_dir
    cst_cfg = cfg.coarse_style_training
    output_dir = cst_cfg.dir_to_save_dataset
    
    creator = PlausibleDataCreator(
        pretrained_sd_model_path=cfg.pretrained_model_path,
        **cst_cfg
    )
    creator.generate(style=style, source_dir=sample_dir, output_dir=output_dir)
    del creator
    

def train(
    cfg, 
    style,
):
    pipeline: CustomizedStableDiffusionPipeline = CustomizedStableDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_path, dtype=jt.float32
    )
    training_cfg = cfg.coarse_style_training
    tokenizer = pipeline.tokenizer
    # NOTE: CLIPTextModel in transformers_jittor (low version) does not support add_adapter
    # text_encoder = pipeline.text_encoder
    text_encoder = LoRACLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder='text_encoder')
    text_encoder.add_adatper(target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])

    noise_scheduler = DDPMScheduler.from_config(os.path.join(cfg.pretrained_model_path, "scheduler/scheduler_config.json"))

    # freeze unnecessary parameters
    unet, vae = pipeline.unet, pipeline.vae
    vae.requires_grad_(False)
    
    token_info = {'coarse_style': (training_cfg.placeholder, training_cfg.init_token)}
    placeholder_token_ids, token_info = customize_token_embeddings(
        tokenizer=tokenizer, text_encoder=text_encoder,
        token_info=token_info,
        mean_init=False
    )
    
    resolution = training_cfg.resolution
    style_data_dir = os.path.join(cfg.sample_dir, style, 'images')
    plausible_data_dir = os.path.join(training_cfg.dir_to_save_dataset, style)
    dataset = CoarseStyleTrainingDataset(
        style_data_dir=style_data_dir, plausible_data_dir=plausible_data_dir, 
        image_encoder=vae, resolution=resolution, placeholder=training_cfg.placeholder
    )
    dataloader = DataLoader(dataset, training_cfg.batch_size, shuffle=True, collate_fn=cst_collate_fn)
    
    # NOTE: CLIPTextModel in transformers_jittor (low version) does not support add_adapter
    # text_lora_config = LoraConfig(
    #     r=training_cfg.lora_rank,
    #     lora_alpha=training_cfg.lora_rank,
    #     init_lora_weights="gaussian",
    #     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    # )
    # text_encoder.add_adapter(text_lora_config)
    # lora_params_to_optimize = [{'params': p, 'lr': 1e-4} for p in list(filter(lambda p: p.requires_grad, text_encoder.parameters()))]
    lora_params_to_optimize = []
    for name, param in text_encoder.named_parameters():
        if 'lora_layer' in name and param.requires_grad:
            # lora_params_to_optimize.append({'params': param, 'lr': training_cfg.lora_lr})
            lora_params_to_optimize.append(param)
            
    # embed_params_to_optimize = [{'params': p, 'lr': training_cfg.embeds_lr} for p in list(text_encoder.get_input_embeddings().parameters())]
    embed_params_to_optimize = list(text_encoder.get_input_embeddings().parameters())
    
    params_to_optim = lora_params_to_optimize  # + embed_params_to_optimize
    optimizer = AdamW(
        params=params_to_optim,
        lr=training_cfg.lr,
        betas=(training_cfg.adam_beta1, training_cfg.adam_beta2),
        weight_decay=training_cfg.adam_weight_decay,
        eps=training_cfg.adam_epsilon
    )
    lr_scheduler = StepLR(optimizer=optimizer, step_size=50, gamma=0.5)
    
    num_update_steps_per_epoch = len(dataloader)
    epochs = math.ceil(training_cfg.max_train_steps / num_update_steps_per_epoch)

    # start training
    print("***** Coarse Style Training *****")
    print(f"  Total optimization steps = {training_cfg.max_train_steps}")
    print(f"  Total epochs = {epochs}")
    print(f"  Data samples = {len(dataloader)}")
    print(f'  LoRA rank = {training_cfg.lora_rank}')
    print(f"  Style = {style}")

    global_steps = 0
    progress_bar = tqdm(range(0, training_cfg.max_train_steps), initial=0, desc='Steps')
    # keep original embeddings as reference
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if global_steps >= training_cfg.max_train_steps:
                break
            
            latents, prompts = batch['latents'], batch['prompts']
            src_latents, src_prompts = batch['src_latents'], batch['src_prompts']
            batch_size = len(latents)

            timesteps = jt.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), 
            )
            timesteps = timesteps.long()
            src_noise = jt.randn_like(src_latents)

            token_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt').input_ids
            encoder_hidden_states = text_encoder(token_ids)[0]

            src_noisy_latents = noise_scheduler.add_noise(src_latents, src_noise, timesteps)
            src_model_pred = unet(src_noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.prediction_type == "epsilon":
                src_target = src_noise
            elif noise_scheduler.prediction_type == "v_prediction":
                src_target = noise_scheduler.get_velocity(src_latents, src_noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

            loss = jt.nn.mse_loss(src_model_pred, src_target)
            loss.backward()
            
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            
             # make sure no update on any embedding weights except the newly added tokens
            index_no_updates = jt.ones((len(tokenizer),)).bool()
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

            with jt.no_grad():
                text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

            progress_bar.update(1)
            global_steps += 1
            logs = {'epoch': epoch+1, 'step': step+1, 'loss': f'{loss:.5f}'}
            progress_bar.set_postfix(**logs)
            del loss, latents, src_model_pred, encoder_hidden_states, src_target, src_noisy_latents, token_ids, src_noise

    # Save the newly trained embeddings
    weight_name = f"{style}_coarse_style_embeds.bin"
    save_path = os.path.join(cfg.output_dir, weight_name)
    save_embeddings(text_encoder=text_encoder, save_path=save_path, token_info=token_info)
    
    # save the lora weights of text encoder
    text_encoder.save_lora_weights(cfg.output_dir, f'{style}_text_encoder_lora.bin')
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml')
    parser.add_argument('--style', type=int)
    parser.add_argument('--create_data', type=str, default='false')
    args = parser.parse_args()
    args.style = f'{args.style:02d}'
    args.create_data = False if 'f' in args.create_data.lower() else True
    
    config = OmegaConf.load(args.config)
    
    jt.flags.use_cuda = 1
    jt.set_global_seed(seed=config.seed)

    config.output_dir = os.path.join(config.output_dir, args.style)
    os.makedirs(config.output_dir, exist_ok=True)
    
    config_to_save = OmegaConf.create({
        'seed': config.seed,
        'output_dir': config.output_dir,
        'coarse_style_training': config.coarse_style_training
    })
    OmegaConf.save(config_to_save, os.path.join(config.output_dir, 'coarse_style_training_config.yaml'))
    print(f'Configurations saved to {os.path.join(config.output_dir, "coarse_style_training_config.yaml")}')
    
    # create training data
    if args.create_data:
        create_plausible_dataset(config, style=args.style)
    # coarse style training
    train(config, style=args.style)
