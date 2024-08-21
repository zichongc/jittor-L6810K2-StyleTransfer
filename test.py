import json
import os
from tqdm import tqdm
import jittor as jt
from omegaconf import OmegaConf


def aan(string: str):
    return 'an' if string[0].lower() in 'aeiou' else 'a'


def randn_latents(seed, shape=(1, 4, 96, 96)):
    jt.set_global_seed(seed)
    latents = jt.randn(shape)
    str(latents)
    return latents


cfg = OmegaConf.load('./configs/inference_config.yaml')
with open('./configs/parameters.json', 'r') as file:
    parameters = json.load(file)

dataset_root = cfg.sample_dir
weights_dir = cfg.weights_dir
# assert cfg.output_dir == './result'

cnt = 0
all_latents_with_seeds = {}
for i in tqdm(range(28)):
    style = f'{i:02d}'
    all_latents_with_seeds[style] = {}
    for prompt, seed in parameters[style]['seed'].items():
        latents = randn_latents(seed)
        all_latents_with_seeds[style][prompt] = latents
        cnt += 1
print(f'Generated latents for all test samples, {cnt} in total.')



jt.flags.use_cuda = 1
from pipeline import StyleBlendT2IPipeline
from scheduler import DDIMScheduler

for i in range(28):
    taskid = f"{i:02d}"
    output_dir = os.path.join(cfg.output_dir, taskid)
    
    print('='*20, 'STYLE', taskid, '='*20)
    
    params = parameters[taskid]
    
    with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
        prompts = json.load(file)

    pipeline: StyleBlendT2IPipeline = StyleBlendT2IPipeline.from_pretrained(cfg.pretrained_model_path, dtype=jt.float32)
    pipeline.scheduler = DDIMScheduler.from_config(os.path.join(cfg.pretrained_model_path, 'scheduler/scheduler_config.json'))
    pipeline.load_styleblend_weights(    
        te_lora_path=os.path.join(weights_dir, taskid, f'{taskid}_text_encoder_lora.bin'),
        unet_lora_path=os.path.join(weights_dir, taskid, f'{taskid}_unet_lora.bin'),
        fine_style_embeds_path=os.path.join(weights_dir, taskid, f'{taskid}_fine_style_embeds.bin'),
        coarse_style_embeds_path=os.path.join(weights_dir, taskid, f'{taskid}_coarse_style_embeds.bin'),
        pretrained_sd_model_path=cfg.pretrained_model_path,
        placeholder_coarse_style = cfg.placeholder_coarse_style,
        placeholder_fine_style = cfg.placeholder_fine_style,
    )
    
    c2f_self_attn_layers_to_register = params['c2f_self_attn_layers_to_register']
    f2c_self_attn_layers_to_register = params['f2c_self_attn_layers_to_register']
    scale = params['scale']
    c2f_step_ratio = params['c2f_step_ratio']
    f2c_step_ratio = params['f2c_step_ratio']
    
    pipeline.register_styleblend_modules(
        c2f_self_attn_layers_to_register=c2f_self_attn_layers_to_register,
        f2c_self_attn_layers_to_register=f2c_self_attn_layers_to_register,  
        scale=scale, 
        c2f_step_ratio=c2f_step_ratio,  
        f2c_step_ratio=f2c_step_ratio,
    )
            
    os.makedirs(output_dir, exist_ok=True)
    
    with jt.no_grad():
        for idx, prompt in prompts.items():
            print(prompt)
            latents = all_latents_with_seeds[taskid][prompt]
            input_prompt = f'{aan(prompt.lower()).title()} {prompt.lower()}'
            
            images = pipeline(prompt=input_prompt, num_inference_steps=30, guidance_scale=7.5, negative_prompt='blur', latents=latents).images
            target = images[1].resize((512, 512))
            # csb = images[0].resize((512, 512))
            
            image_path = os.path.join(output_dir, f'{prompt}.png')
            target.save(image_path)