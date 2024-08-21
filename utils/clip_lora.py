import os
import jittor as jt


class LoRALinearLayer(jt.nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.down = jt.nn.Linear(in_features=in_features, out_features=rank, bias=False)
        self.up = jt.nn.Linear(in_features=rank, out_features=out_features, bias=False)
        
        jt.nn.init.kaiming_normal_(self.down.weight)
        jt.nn.init.zero_(self.up.weight)
        
    def execute(self, x):
        x = self.up(self.down(x))
        return x


# NOTE: We found the `requires_grad` setting is invalid for `jittor.nn` modules after importing packages like `transformers` or `JDuffision`.
# Therefore, in this jittor implementation, we prebuild the LoRALinearLayer in advance to avoid the gradient issue.
clip_model_lora_layers = {
    'text_model.encoder.layers.0.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.0.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.0.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.0.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.1.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.1.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.1.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.1.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.2.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.2.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.2.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.2.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.3.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.3.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.3.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.3.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.4.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.4.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.4.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.4.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.5.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.5.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.5.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.5.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.6.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.6.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.6.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.6.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.7.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.7.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.7.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.7.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.8.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.8.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.8.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.8.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.9.self_attn.k_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.9.self_attn.v_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.9.self_attn.q_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.9.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.10.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.10.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.10.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.10.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.11.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.11.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.11.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.11.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.12.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.12.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.12.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.12.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.13.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.13.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.13.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.13.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.14.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.14.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.14.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.14.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.15.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.15.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.15.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.15.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.16.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.16.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.16.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.16.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.17.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.17.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.17.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.17.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.18.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.18.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.18.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.18.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.19.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.19.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.19.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.19.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.20.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.20.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.20.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.20.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.21.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.21.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.21.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.21.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.22.self_attn.k_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.22.self_attn.v_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.22.self_attn.q_proj':  LoRALinearLayer(1024, 1024, 16),
    'text_model.encoder.layers.22.self_attn.out_proj': LoRALinearLayer(1024, 1024, 16),
}


def custom_lora_linear(layer, layer_name, lora_layer: LoRALinearLayer):
    class LoRALinear(jt.nn.Module):
        def __init__(self):
            self._layer_name = layer_name
            self.lora_layer = lora_layer
            self.linear = layer
            self.in_features = lora_layer.in_features
            self.out_features = lora_layer.out_features

        def execute(self, x):
            res_x = self.lora_layer(x)
            x = self.linear(x)
            
            x = x + res_x  # fix the lora scale to 1.
            return x
    return LoRALinear()


from transformers import CLIPTextModel
class LoRACLIPTextModel(CLIPTextModel):
    def add_adatper(self, lora_rank=16, target_modules=None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        # TODO: in this implementation, we force to add lora layers (rank=16) for all linears: ["q_proj", "k_proj", "v_proj", "out_proj"] 
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        
        for name, module in self.named_modules():
            if name.split('.')[-1] == 'self_attn' and module.__class__.__name__ == 'CLIPAttention':
                for tgt_module in target_modules:
                    lora_layer = clip_model_lora_layers[name+f'.{tgt_module}']
                    if tgt_module == 'q_proj':
                        module.q_proj = custom_lora_linear(module.q_proj, name, lora_layer=lora_layer)
                    elif tgt_module == 'k_proj':
                        module.k_proj = custom_lora_linear(module.k_proj, name, lora_layer=lora_layer)
                    elif tgt_module == 'v_proj':
                        module.v_proj = custom_lora_linear(module.v_proj, name, lora_layer=lora_layer)
                    elif tgt_module == 'out_proj':
                        module.out_proj = custom_lora_linear(module.out_proj, name, lora_layer=lora_layer)
        
    def load_lora_weights(self, pretrained_model_path):
        state_dict = jt.load(pretrained_model_path)
        
        has_lora_layer = False
        for name, module in self.named_modules():
            if 'lora_layer' in name:
                has_lora_layer = True
                break
        if not has_lora_layer:
            self.add_adatper()
                
        for name, module in self.named_modules():
            if name.endswith('lora_layer'):
                module.load_state_dict(state_dict[name])
        print(f'Loaded weights from {pretrained_model_path}.')
        
    def save_lora_weights(self, output_dir, weight_name):
        state_dict = {}
        for name, module in self.named_modules():
            if name.endswith('lora_layer'):
                state_dict[name] = module.state_dict()
        
        jt.save(state_dict, os.path.join(output_dir, weight_name))
        
    def unload_lora_weights(self):
        # TODO
        raise NotImplementedError
    