# code mostly taken from https://github.com/huggingface/diffusers
import os
import glob
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_3d_blocks import (
    CrossAttnDownBlockPseudo3D,
    CrossAttnUpBlockPseudo3D,
    DownBlockPseudo3D,
    UNetMidBlockPseudo3DCrossAttn,
    UpBlockPseudo3D,
    get_down_block,
    get_up_block,
)
from .resnet import PseudoConv3d
from .unet_3d_condition import UNetPseudo3DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MyUNetPseudo3DConditionModel(UNetPseudo3DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None, # None
        attention_mask: Optional[torch.Tensor] = None,): # None
        
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True
        
        if attention_mask is not None: # None
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        # print("test videomodel6")
        # from IPython import embed; embed()
      
            
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
            
        # 2. pre-process
        # print("test videomodel6")
        # from IPython import embed; embed()
        # sample 1, 4, 8, 64, 64
        sample = self.conv_in(sample)
        # 1, 320, 8, 64, 64
        
        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # print("test videomodel6")
                # from IPython import embed; embed()
                sample, res_samples = downsample_block(
                    hidden_states=sample, # 1, 320, 8, 64, 64
                    temb=emb, # 1, 1280
                    encoder_hidden_states=encoder_hidden_states, #[2, 77, 768]
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
        
        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        
        # 5. up
        up_ft = []
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
                
            if i in up_ft_indices:
                up_ft.append(sample)
            
        output = {}
        output['up_ft'] = up_ft
        return output