from typing import Literal, Optional, Callable
import jittor as jt
from utils import Handler


indexes = {
    'down_blocks.0.attentions.0.transformer_blocks.0.attn2': 0,
    'down_blocks.0.attentions.1.transformer_blocks.0.attn2': 1,
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2': 2,
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2': 3,
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2': 4,
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2': 5,
    'mid_block.attentions.0.transformer_blocks.0.attn2': 6,
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2': 7,
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2': 8,
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2': 9,
    'up_blocks.2.attentions.0.transformer_blocks.0.attn2': 10,
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2': 11,
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2': 12,
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2': 13,
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2': 14,
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2': 15,
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1': 0,
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1': 1,
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1': 2,
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1': 3,
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1': 4,
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1': 5,
    'mid_block.attentions.0.transformer_blocks.0.attn1': 6,
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1': 7,
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1': 8,
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1': 9,
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1': 10,
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1': 11,
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1': 12,
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1': 13,
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1': 14,
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1': 15,
}


def adain(feat, direction: Literal['c2f', 'f2c'] = 'c2f'):
    feat_mean = feat.mean(dim=-2, keepdims=True)
    feat_std = (feat.var(dim=-2, keepdims=True) + 1e-5).sqrt()
    if direction == 'c2f':
        feat_mean_ref = feat_mean[2:3].repeat(4, 1, 1)
        feat_std_ref = feat_std[2:3].repeat(4, 1, 1)
    else:
        feat_mean_ref = feat_mean[3:4].repeat(4, 1, 1)
        feat_std_ref = feat_std[3:4].repeat(4, 1, 1)
        
    feat = (feat - feat_mean) / feat_std * feat_std_ref + feat_mean_ref
    return feat
                

def feature_blender():
    def inject(attn, query, key, value):
        """
        query from coarse style branch to fine style branch
        key and value from fine style branch to coarse style branch
        """
        query[[1, 3], ...] = query[[2, 2], ...] 
        key[[0, 2], ...] = key[[3, 3], ...]
        value[[0, 2], ...] = value[[3, 3], ...]
        return query, key, value
    return inject    


def structure_injector(direction: Literal['c2f', 'f2c'] = 'c2f'):
    """
    direction:
        c2f: features inject from coarse style branch to fine style branch
        f2c: features inject from fine style branch to coarse style branch
    """
    def inject(attn, query, key, value):
        if direction == 'c2f':
            query[[1, 3], ...] = query[[2, 2], ...]
        elif direction == 'f2c':
            query[[0, 2], ...] = query[[3, 3], ...]
            
        return query, key, value
    return inject


def appearance_injector(direction: Literal['c2f', 'f2c'] = 'c2f'):
    """
    direction:
        c2f: features inject from coarse style branch to fine style branch
        f2c: features inject from fine style branch to coarse style branch
    """
    def inject(attn, query, key, value):
        if direction == 'c2f':
            key[[1, 3], ...] = key[[2, 2], ...]
            value[[1, 3], ...] = value[[2, 2], ...]
        elif direction == 'f2c':
            key[[0, 2], ...] = key[[3, 3], ...]
            value[[0, 2], ...] = value[[3, 3], ...]
            
        return query, key, value
    return inject


def styleblend_self_attention(block_class, csb_lora_scale=1.0, feature_injector=None, handler: Handler = None, steps_to_blend=None):
    class Attention(block_class):
        _parent_class = block_class
        _scale = csb_lora_scale
        _feature_injector = feature_injector
        _steps_to_blend = steps_to_blend if steps_to_blend is not None else [i for i in range(handler.total_steps)]
            
        def forward(
            attn,
            hidden_states: jt.Var,
            encoder_hidden_states: Optional[jt.Var] = None,
            attention_mask: Optional[jt.Var] = None,
            temb: Optional[jt.Var] = None,
            scale: float = 1.0,
        ) -> jt.Var:
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            # Adjust the lora scale for coarse style branch (csb).
            # Keep the lora scale to 1.0 for fine style branch (fsb).
            # For each batch: {neg_coarse, neg_fine, coarse, fine}, hidden states in shape [4, S, D]
            fsb_hidden_states = hidden_states[[1, 3], ...]
            fsb_encoder_hidden_states = encoder_hidden_states[[1, 3], ...]
            
            fsb_args = (scale,)
            fsb_query = attn.to_q(fsb_hidden_states, *fsb_args)
            fsb_key = attn.to_k(fsb_encoder_hidden_states, *fsb_args)
            fsb_value = attn.to_v(fsb_encoder_hidden_states, *fsb_args)
            
            csb_args = (attn._scale,)
            csb_hidden_states = hidden_states[[0, 2], ...]
            csb_encoder_hidden_states = encoder_hidden_states[[0, 2], ...]
            
            csb_query = attn.to_q(csb_hidden_states, *csb_args)
            csb_key = attn.to_k(csb_encoder_hidden_states, *csb_args)
            csb_value = attn.to_v(csb_encoder_hidden_states, *csb_args)
            
            query = jt.stack([csb_query[0], fsb_query[0], csb_query[1], fsb_query[1]])
            key = jt.stack([csb_key[0], fsb_key[0], csb_key[1], fsb_key[1]])
            value = jt.stack([csb_value[0], fsb_value[0], csb_value[1], fsb_value[1]])

            # Custom QKV feature injection if `attn._feature_injector` is callable.
            if attn._feature_injector is not None and isinstance(attn._feature_injector, Callable) and handler.cur_step in attn._steps_to_blend:
                # print(query.shape, key.shape, value.shape)
                query, key, value = attn._feature_injector(query, key, value)
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = jt.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *fsb_args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
    return Attention


def styleblend_cross_attention(block_class, csb_lora_scale=1.0, feature_injector=None, handler: Handler = None, steps_to_blend=None):
    class Attention(block_class):
        _parent_class = block_class
        _scale = csb_lora_scale
        _feature_injector = feature_injector
        _steps_to_blend = steps_to_blend if steps_to_blend is not None else [i for i in range(handler.total_steps)]
            
        def forward(
            attn,
            hidden_states: jt.Var,
            encoder_hidden_states: Optional[jt.Var] = None,
            attention_mask: Optional[jt.Var] = None,
            temb: Optional[jt.Var] = None,
            scale: float = 1.0,
        ) -> jt.Var:
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            # Adjust the lora scale for coarse style branch (csb).
            # Keep the lora scale to 1.0 for fine style branch (fsb).
            # For each batch: {neg_coarse, neg_fine, coarse, fine}, hidden states in shape [4, S, D]
            fsb_hidden_states = hidden_states[[1, 3], ...]
            fsb_encoder_hidden_states = encoder_hidden_states[[1, 3], ...]
            
            fsb_args = (scale,)
            fsb_query = attn.to_q(fsb_hidden_states, *fsb_args)
            fsb_key = attn.to_k(fsb_encoder_hidden_states, *fsb_args)
            fsb_value = attn.to_v(fsb_encoder_hidden_states, *fsb_args)
            
            csb_args = (attn._scale,)
            csb_hidden_states = hidden_states[[0, 2], ...]
            csb_encoder_hidden_states = encoder_hidden_states[[0, 2], ...]
            
            csb_query = attn.to_q(csb_hidden_states, *csb_args)
            csb_key = attn.to_k(csb_encoder_hidden_states, *csb_args)
            csb_value = attn.to_v(csb_encoder_hidden_states, *csb_args)
            
            query = jt.stack([csb_query[0], fsb_query[0], csb_query[1], fsb_query[1]])
            key = jt.stack([csb_key[0], fsb_key[0], csb_key[1], fsb_key[1]])
            value = jt.stack([csb_value[0], fsb_value[0], csb_value[1], fsb_value[1]])

            # Custom QKV feature injection if `attn._feature_injector` is callable.
            if attn._feature_injector is not None and isinstance(attn._feature_injector, Callable) and handler.cur_step in attn._steps_to_blend:
                # print(query.shape, key.shape, value.shape)
                query, key, value = attn._feature_injector(query, key, value)
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = jt.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *fsb_args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
    return Attention
