# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class STSGSelfAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("STSGSelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        bs_seq_length, num_frames, _ = hidden_states.shape

        # First shuffle the indices of num_frames
        first_shuffled_indices = torch.randperm(num_frames)
        shuffled_hidden_states_first_half = hidden_states[:bs_seq_length//2, first_shuffled_indices, :].clone()

        # Then shuffle the indices of the first half of the sequence length
        second_shuffled_indices = torch.randperm(shuffled_hidden_states_first_half.shape[0])
        shuffled_hidden_states_first_half = shuffled_hidden_states_first_half[second_shuffled_indices]

        # Concatenate the shuffled first half with the unchanged second half
        shuffled_hidden_states = torch.cat([shuffled_hidden_states_first_half, hidden_states[bs_seq_length//2:]], dim=0)

        # Update hidden_states to the new shuffled version
        hidden_states = shuffled_hidden_states


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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class STSGMixin:
    r"""Mixin class for STSG."""

    @staticmethod
    def _check_input_stsg_applied_layer(layer):
        r"""
        Check if each layer input in `applied_stsg_layers` is valid. It should be either one of these 3 formats:
        "{block_type}", "{block_type}.{block_index}", or "{block_type}.{block_index}.{attention_index}". `block_type`
        can be "down", "mid", "up". `block_index` should be in the format of "block_{i}". `attention_index` should be
        in the format of "attentions_{j}".
        """

        layer_splits = layer.split(".")

        if len(layer_splits) > 3:
            raise ValueError(f"stsg layer should only contains block_type, block_index and attention_index{layer}.")

        if len(layer_splits) >= 1:
            if layer_splits[0] not in ["down", "mid", "up"]:
                raise ValueError(
                    f"Invalid block_type in stsg layer {layer}. Accept 'down', 'mid', 'up', got {layer_splits[0]}"
                )

        if len(layer_splits) >= 2:
            if not layer_splits[1].startswith("block_"):
                raise ValueError(f"Invalid block_index in stsg layer: {layer}. Should start with 'block_'")

        if len(layer_splits) == 3:
            if not layer_splits[2].startswith("attentions_"):
                raise ValueError(f"Invalid attention_index in stsg layer: {layer}. Should start with 'attentions_'")

    def _set_stsg_attn_processor(self, stsg_applied_layers):
        r"""
        Set the attention processor for the STSG layers.
        """
        stsg_attn_proc = STSGSelfAttnProcessor2_0()

        def is_self_attn(module_name):
            r"""
            Check if the module is self-attention module based on its name.
            """
            return "attn1" in module_name and "to" not in name

        def get_block_type(module_name):
            r"""
            Get the block type from the module name. can be "down", "mid", "up".
            """
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "down"
            return module_name.split(".")[0].split("_")[0]

        def get_block_index(module_name):
            r"""
            Get the block index from the module name. can be "block_0", "block_1", ... If there is only one block (e.g.
            mid_block) and index is ommited from the name, it will be "block_0".
            """
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "block_1"
            # mid_block.attentions.0.transformer_blocks.0.attn1 -> "block_0"
            if "attentions" in module_name.split(".")[1]:
                return "block_0"
            else:
                return f"block_{module_name.split('.')[1]}"

        def get_attn_index(module_name):
            r"""
            Get the attention index from the module name. can be "attentions_0", "attentions_1", ...
            """
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "attentions_0"
            # mid_block.attentions.0.transformer_blocks.0.attn1 -> "attentions_0"
            if "attentions" in module_name.split(".")[2]:
                return f"attentions_{module_name.split('.')[3]}"
            elif "attentions" in module_name.split(".")[1]:
                return f"attentions_{module_name.split('.')[2]}"

        for stsg_layer_input in stsg_applied_layers:
            # for each STSG layer input, we find corresponding self-attention layers in the unet model
            target_modules = []

            stsg_layer_input_splits = stsg_layer_input.split(".")

            if len(stsg_layer_input_splits) == 1:
                # when the layer input only contains block_type. e.g. "mid", "down", "up"
                block_type = stsg_layer_input_splits[0]
                for name, module in self.unet.named_modules():
                    if is_self_attn(name) and get_block_type(name) == block_type:
                        target_modules.append(module)

            elif len(stsg_layer_input_splits) == 2:
                # when the layer inpput contains both block_type and block_index. e.g. "down.block_1", "mid.block_0"
                block_type = stsg_layer_input_splits[0]
                block_index = stsg_layer_input_splits[1]
                for name, module in self.unet.named_modules():
                    if (
                        is_self_attn(name)
                        and get_block_type(name) == block_type
                        and get_block_index(name) == block_index
                    ):
                        target_modules.append(module)

            elif len(stsg_layer_input_splits) == 3:
                # when the layer input contains block_type, block_index and attention_index. e.g. "down.blocks_1.attentions_1"
                block_type = stsg_layer_input_splits[0]
                block_index = stsg_layer_input_splits[1]
                attn_index = stsg_layer_input_splits[2]

                for name, module in self.unet.named_modules():
                    if (
                        is_self_attn(name)
                        and get_block_type(name) == block_type
                        and get_block_index(name) == block_index
                        and get_attn_index(name) == attn_index
                    ):
                        target_modules.append(module)

            if len(target_modules) == 0:
                raise ValueError(f"Cannot find stsg layer to set attention processor for: {stsg_layer_input}")

            for module in target_modules:
                module.processor = stsg_attn_proc

    def set_stsg_applied_layers(self, stsg_applied_layers):
        r"""
        set the the self-attention layers to apply STSG. Raise ValueError if the input is invalid.
        """

        if not isinstance(stsg_applied_layers, list):
            stsg_applied_layers = [stsg_applied_layers]

        for stsg_layer in stsg_applied_layers:
            self._check_input_stsg_applied_layer(stsg_layer)

        self.stsg_applied_layers = stsg_applied_layers
