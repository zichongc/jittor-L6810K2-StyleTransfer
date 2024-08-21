import os
import argparse
import math
from tqdm import tqdm
from omegaconf import OmegaConf
import jittor as jt
from jittor.lr_scheduler import StepLR
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import DataLoader
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from JDiffusion.pipelines.pipeline_stable_diffusion_jittor import LoraLoaderMixin

from scheduler import DDPMScheduler
from pipeline import CustomizedStableDiffusionPipeline
from utils.dataset import FineStyleTrainingDataset, fst_collate_fn
from utils import customize_token_embeddings, save_embeddings, convert_state_dict_to_diffusers


def textual_inversion(
    cfg,
    pipeline,
    unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    dataloader,
    placeholder_token_ids,
    token_info,
    style,
    output_dir=None,
):
    num_update_steps_per_epoch = len(dataloader)
    epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    
    params_to_optim = list(text_encoder.get_input_embeddings().parameters())
    optimizer = AdamW(
        params=params_to_optim,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon
    )
    lr_scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.7)
    
    # start training
    print("***** Textual Inversion *****")
    print(f"  Total optimization steps = {cfg.max_train_steps}")
    print(f"  Total epochs = {epochs}")
    print(f"  Data samples = {len(dataloader)}")
    print(f"  Style = {style}")

    global_steps = 0
    progress_bar = tqdm(range(0, cfg.max_train_steps), initial=0, desc='Steps')
    # keep original embeddings as reference
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if global_steps >= cfg.max_train_steps:
                break
            
            images = batch['images']
            prompts = batch['prompts']                 
            batch_size = len(images)
            
            latents = jt.concat([pipeline.image2latent(image) for image in images])
            noise = jt.randn_like(latents)
            timesteps = jt.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), 
            )
            timesteps = timesteps.long()
            
            token_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt').input_ids
            encoder_hidden_states = text_encoder(token_ids)[0]
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

            loss = jt.nn.mse_loss(model_pred, target)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
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
            del loss, latents, model_pred, encoder_hidden_states, target, noisy_latents, token_ids, noise

    if output_dir is not None:
        # Save the newly trained embeddings
        weight_name = f"{style}_fine_style_embeds.bin"
        save_path = os.path.join(output_dir, weight_name)
        save_embeddings(text_encoder=text_encoder, save_path=save_path, token_info=token_info)


def dreambooth_lora(
    cfg,
    pipeline,
    unet,
    tokenizer,
    text_encoder,
    noise_scheduler,
    dataloader,
    style,
    output_dir=None,
):
    num_update_steps_per_epoch = len(dataloader)
    epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    params_to_optim = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = AdamW(
        params=params_to_optim,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon
    )
    lr_scheduler = StepLR(optimizer=optimizer, step_size=200, gamma=0.5)
    
    # start training
    print("***** DreamBooth lora *****")
    print(f"  Total optimization steps = {cfg.max_train_steps}")
    print(f"  Total epochs = {epochs}")
    print(f"  Data samples = {len(dataloader)}")
    print(f'  LoRA rank = {cfg.rank}')
    print(f"  Style = {style}")

    global_steps = 0
    progress_bar = tqdm(range(0, cfg.max_train_steps), initial=0, desc='Steps')
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if global_steps >= cfg.max_train_steps:
                break
            
            images, prompts = batch['images'], batch['prompts']
            batch_size = len(images)

            latents = jt.concat([pipeline.image2latent(image) for image in images])
            noise = jt.randn_like(latents)

            timesteps = jt.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), 
            )
            timesteps = timesteps.long()

            token_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt').input_ids
            encoder_hidden_states = text_encoder(token_ids)[0]

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Get the target for loss depending on the prediction type
            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

            loss = jt.nn.mse_loss(model_pred, target)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_steps += 1
            logs = {'epoch': epoch+1, 'step': step+1, 'loss': f'{loss:.5f}'}
            progress_bar.set_postfix(**logs)
            del loss, latents, model_pred, encoder_hidden_states, target, noisy_latents, token_ids, noise

    if output_dir is not None:
        # Save the lora weights following the dreambooth-lora code example of JDiffusion
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        weight_name = f"{style}_unet_lora.bin"
        LoraLoaderMixin.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=False,
            weight_name=weight_name
        )
        print('Saved unet lora weights to', os.path.join(output_dir, weight_name))
        
        
def train(cfg, style):    
    pipeline: CustomizedStableDiffusionPipeline = CustomizedStableDiffusionPipeline.from_pretrained(
        cfg.pretrained_model_path, dtype=jt.float32
    )
    training_cfg = cfg.fine_style_training
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    noise_scheduler = DDPMScheduler.from_config(os.path.join(cfg.pretrained_model_path, "scheduler/scheduler_config.json"))

    token_info = {'fine_style': (training_cfg.placeholder, training_cfg.init_token)}
    placeholder_token_ids, token_info = customize_token_embeddings(
        tokenizer=tokenizer, text_encoder=text_encoder,
        token_info=token_info,
    )
    
    # freeze unnecessary parameters
    unet, vae = pipeline.unet, pipeline.vae
    vae.requires_grad_(False)

    images_dir = os.path.join(cfg.sample_dir, style, 'images')
    dataset = FineStyleTrainingDataset(
        data_dir=images_dir, 
        placeholder=training_cfg.placeholder, 
        resolution=training_cfg.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=training_cfg.batch_size, shuffle=True, num_workers=0, collate_fn=fst_collate_fn)

    # train embeddings for fine style representation
    ti_cfg = training_cfg.textual_inversion
    textual_inversion(
        ti_cfg, 
        pipeline, unet, tokenizer, text_encoder, 
        noise_scheduler, dataloader, placeholder_token_ids, token_info,
        style=style, output_dir=cfg.output_dir
    )
    
    # train lora for fine style representation
    text_encoder.requires_grad_(False)
    lora_cfg = training_cfg.dreambooth_lora
    dreambooth_lora(
        lora_cfg,
        pipeline, unet, tokenizer, text_encoder, 
        noise_scheduler, dataloader, 
        style=style, output_dir=cfg.output_dir
    )
    
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml')
    parser.add_argument('--style', type=int)
    args = parser.parse_args()
    args.style = f'{args.style:02d}'
    
    config = OmegaConf.load(args.config)
    
    jt.flags.use_cuda = 1
    jt.set_global_seed(seed=config.seed)

    config.output_dir = os.path.join(config.output_dir, args.style)
    os.makedirs(config.output_dir, exist_ok=True)
    
    config_to_save = OmegaConf.create({
        'seed': config.seed,
        'output_dir': config.output_dir,
        'fine_style_training': config.fine_style_training
    })
    OmegaConf.save(config_to_save, os.path.join(config.output_dir, 'fine_style_training_config.yaml'))
    print(f'Configurations saved to {os.path.join(config.output_dir, "fine_style_training_config.yaml")}')
    
    train(config, style=args.style)
