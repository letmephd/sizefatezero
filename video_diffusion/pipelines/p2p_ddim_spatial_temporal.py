# code mostly taken from https://github.com/huggingface/diffusers

from typing import Callable, List, Optional, Union
import os, sys
import PIL
import torch
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm

from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import copy
from basicsr.utils import img2tensor

from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from ..models.unet_3d_condition import UNetPseudo3DConditionModel
from ..models.estimator import  MyUNetPseudo3DConditionModel
from .stable_diffusion import SpatioTemporalStableDiffusionPipeline
from video_diffusion.prompt_attention import attention_util
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}

def process_move(path_mask, h, w, dx, dy, scale, input_scale, resize_scale, up_scale, up_ft_index, w_edit, w_content, w_contrast, w_inpaint,  precision, path_mask_ref=None):
    dx, dy = dx*input_scale, dy*input_scale
#     if isinstance(path_mask, str):
#         mask_x0 = cv2.imread(path_mask)
#     else:
#         mask_x0 = path_mask
#     mask_x0 = cv2.resize(mask_x0, (h, w))
    if path_mask_ref is not None: # 不用看
        if isinstance(path_mask_ref, str):
            mask_x0_ref = cv2.imread(path_mask_ref)
        else:
            mask_x0_ref = path_mask_ref
        mask_x0_ref = cv2.resize(mask_x0_ref, (h, w))
    else:
        mask_x0_ref=None
        
#     mask_x0 = img2tensor(mask_x0)[0]
    mask_x0 = F.interpolate(path_mask,size=(h,w))[:,0]
                            
    mask_x0 = (mask_x0>0.5).float().to('cuda', dtype=precision)
    if mask_x0_ref is not None:
        mask_x0_ref = img2tensor(mask_x0_ref)[0]
        mask_x0_ref = (mask_x0_ref>0.5).float().to('cuda', dtype=precision)
    mask_org = F.interpolate(mask_x0[:,None], (int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)))>0.5

    mask_tar = F.interpolate(mask_x0[:,None], (int(mask_x0.shape[-2]//scale*resize_scale), int(mask_x0.shape[-1]//scale*resize_scale)))>0.5 # torch.Size([8, 1, 128, 128])
    mask_cur = torch.roll(mask_tar, (int(dy//scale*resize_scale), int(dx//scale*resize_scale)), (-2,-1)) # torch.Size([8, 1, 128, 128])
    
    pad_size_x = abs(mask_tar.shape[-1]-mask_org.shape[-1])//2
    pad_size_y = abs(mask_tar.shape[-2]-mask_org.shape[-2])//2
    if resize_scale>1:
        sum_before = torch.sum(mask_cur)
        mask_cur = mask_cur[:,:,pad_size_y:pad_size_y+mask_org.shape[-2],pad_size_x:pad_size_x+mask_org.shape[-1]]
        # from IPython import embed; embed() 
        sum_after = torch.sum(mask_cur)
        if sum_after != sum_before:
            raise ValueError('Resize out of bounds, exiting.')
    else:
        temp = torch.zeros(mask_org.shape[0],1,mask_org.shape[-2], mask_org.shape[-1]).to(mask_org.device)
        temp[:,:,pad_size_y:pad_size_y+mask_cur.shape[-2],pad_size_x:pad_size_x+mask_cur.shape[-1]]=mask_cur
        mask_cur =temp>0.5

    mask_other = (1-((mask_cur+mask_org)>0.5).float())>0.5
    mask_overlap = ((mask_cur.float()+mask_org.float())>1.5).float()
    mask_non_overlap = (mask_org.float()-mask_overlap)>0.5
    # print("test_model0")
    #     # 将mask保存在本地        
    # from IPython import embed; embed()

    return {
        "mask_x0":mask_x0, 
        "mask_x0_ref":mask_x0_ref, 
        "mask_tar":mask_tar, 
        "mask_cur":mask_cur, 
        "mask_other":mask_other, 
        "mask_overlap":mask_overlap, 
        "mask_non_overlap":mask_non_overlap, 
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "resize_scale":resize_scale,
        "w_edit":w_edit,
        "w_content":w_content,
        "w_contrast":w_contrast,
        "w_inpaint":w_inpaint, 
    }

class P2pDDIMSpatioTemporalPipeline(SpatioTemporalStableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler,],
        disk_store: bool=False
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.store_controller = attention_util.AttentionStore(disk_store=disk_store)
        self.empty_controller = attention_util.EmptyControl()
        
        self.up_ft_index = [1,2]
        self.up_scale = 2  
        self.precision = torch.float16
        
    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """

    def check_inputs(self, prompt, height, width, callback_steps, strength=None):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if strength is not None:
            if strength <= 0 or strength > 1:
                raise ValueError(f"The value of strength should in (0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
    
    @torch.no_grad()
    def prepare_latents_ddim_inverted(self, image, batch_size, num_images_per_prompt, 
                                        text_embeddings,
                                        store_attention=False, prompt=None,
                                        generator=None,
                                        LOW_RESOURCE = True,
                                        save_path = None
                                      ): # inversion过程
        self.prepare_before_train_loop()
        if store_attention: # True
            attention_util.register_attention_control(self, self.store_controller)
        resource_default_value = self.store_controller.LOW_RESOURCE
        self.store_controller.LOW_RESOURCE = LOW_RESOURCE  # in inversion, no CFG, record all latents attention
        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        # get latents
        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)
        #Invert clean image to noise latents by DDIM and Unet
        ddim_latents_all_step = self.ddim_clean2noisy_loop(init_latents_bcfhw, text_embeddings, self.store_controller)
        if store_attention and (save_path is not None) :
            os.makedirs(save_path+'/cross_attention')
            attention_output = attention_util.show_cross_attention(self.tokenizer, prompt, 
                                                                   self.store_controller, 16, ["up", "down"],
                                                                   save_path = save_path+'/cross_attention')

            # Detach the controller for safety
            attention_util.register_attention_control(self, self.empty_controller)
        self.store_controller.LOW_RESOURCE = resource_default_value
        
        return ddim_latents_all_step
    
    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings, controller:attention_util.AttentionControl=None):
        weight_dtype = latent.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        print('Invert clean image to noise latents by DDIM and Unet')
        for i in trange(len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            
            # [1, 4, 8, 64, 64] ->  [1, 4, 8, 64, 64])
            noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            
            latent = self.next_clean2noise_step(noise_pred, t, latent) # 将当前生成的潜在表示（x_t）添加到 latents_store 列表中。
            if controller is not None: controller.step_callback(latent)
            all_latent.append(latent.to(dtype=weight_dtype))
        
        return all_latent
    
    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """
        Assume the eta in DDIM=0
        """
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    def p2preplace_edit(self, **kwargs):
        # Edit controller during inference
        # The controller must know the source prompt for replace mapping
        
        len_source = {len(kwargs['source_prompt'].split(' '))}
        len_target = {len(kwargs['prompt'].split(' '))}
        equal_length = (len_source == len_target)
        print(f" len_source: {len_source}, len_target: {len_target}, equal_length: {equal_length}")
        edit_controller = attention_util.make_controller(
                            self.tokenizer, 
                            [ kwargs['source_prompt'], kwargs['prompt']],
                            NUM_DDIM_STEPS = kwargs['num_inference_steps'],
                            is_replace_controller=kwargs.get('is_replace_controller', True) and equal_length,
                            cross_replace_steps=kwargs['cross_replace_steps'], 
                            self_replace_steps=kwargs['self_replace_steps'], 
                            blend_words=kwargs.get('blend_words', None),
                            equilizer_params=kwargs.get('eq_params', None),
                            additional_attention_store=self.store_controller,
                            use_inversion_attention = kwargs['use_inversion_attention'],
                            blend_th = kwargs.get('blend_th', (0.3, 0.3)),
                            blend_self_attention = kwargs.get('blend_self_attention', None),
                            blend_latents=kwargs.get('blend_latents', None),
                            save_path=kwargs.get('save_path', None),
                            save_self_attention = kwargs.get('save_self_attention', True),
                            disk_store = kwargs.get('disk_store', False)
                            )
        
        # attention_util.register_attention_control(self, edit_controller)
        attention_util.register_attention_control(self, self.empty_controller)
        

        # In ddim inferece, no need source prompt
        sdimage_output = self.sd_ddim_pipeline(
            # controller = edit_controller, ?
            controller = self.empty_controller, 
            # target_prompt = kwargs['prompts'][1],
            **kwargs)
        if hasattr(edit_controller.latent_blend, 'mask_list'):
            mask_list = edit_controller.latent_blend.mask_list
        else:
            mask_list = None
        if len(edit_controller.attention_store.keys()) > 0:
            attention_output = attention_util.show_cross_attention(self.tokenizer, kwargs['prompt'], 
                                                               edit_controller, 16, ["up", "down"])
        else:
            attention_output = None
        dict_output = {
                "sdimage_output" : sdimage_output,
                "attention_output" : attention_output,
                "mask_list" : mask_list,
            }
        attention_util.register_attention_control(self, self.empty_controller)
        return dict_output

    
    
    
    # @torch.no_grad()
    def __call__(self, **kwargs):
        edit_type = kwargs['edit_type']
        assert edit_type in ['save', 'swap', None]
        if edit_type is None:
            return self.sd_ddim_pipeline(controller = None, **kwargs)

        if edit_type == 'save':
            del self.store_controller
            self.store_controller = attention_util.AttentionStore()
            attention_util.register_attention_control(self, self.store_controller)
            sdimage_output = self.sd_ddim_pipeline(controller = self.store_controller, **kwargs)
            
            mask_list = None
            
            attention_output = attention_util.show_cross_attention(self.tokenizer, kwargs['prompt'], self.store_controller, 16, ["up", "down"])
            

            dict_output = {
                "sdimage_output" : sdimage_output,
                "attention_output"   : attention_output,
                'mask_list':  mask_list
            }

            # Detach the controller for safety
            attention_util.register_attention_control(self, self.empty_controller)
            return dict_output
        
        if edit_type == 'swap':

            return self.p2preplace_edit(**kwargs)

    
    # @torch.no_grad()
    def sd_ddim_pipeline(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = None,
        estimator: MyUNetPseudo3DConditionModel = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        ddim_latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controller: attention_util.AttentionControl = None,
        **args
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Only used in DDIM or strength<1.0
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.            
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, strength)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        text_embeddings_org = text_embeddings
        
        edit_obj_text_embeddings= self._encode_prompt(
            "a black swan", device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        if latents is None:
            ddim_latents_all_step = self.prepare_latents_ddim_inverted(
                image, batch_size, num_images_per_prompt, 
                text_embeddings,
                store_attention=False, # avoid recording attention in first inversion
                generator = generator,
            )
            latents = ddim_latents_all_step[-1]
        else:
            ddim_latents_all_step=None

        latents_dtype = latents.dtype

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # # 这一部分参考process_move
        resize_scale = 0.5
        mask_tmp = (F.interpolate(mask[:,0,:].unsqueeze(0), (int(latents.shape[-2]*resize_scale), int(latents.shape[-1]*resize_scale)))>0).float().to('cuda', dtype=latents.dtype).unsqueeze(1)
        origin_lshape = latents.shape #1, 4, 8, 64, 64
        # from IPython import embed; embed() 
        latents_tmp = F.interpolate(latents.view(-1,origin_lshape[-3],origin_lshape[-2],origin_lshape[-1]), (int(latents.shape[-2]*resize_scale), int(latents.shape[-1]*resize_scale))).unsqueeze(0)
        pad_size_x = abs(mask_tmp.shape[-1]-latents.shape[-1])//2
        pad_size_y = abs(mask_tmp.shape[-2]-latents.shape[-2])//2
        
        
        if resize_scale>1:
            sum_before = torch.sum(mask_tmp)
            mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latents.shape[-2],pad_size_x:pad_size_x+latents.shape[-1]]
            latents_tmp = latents_tmp[:,:,pad_size_y:pad_size_y+latents.shape[-2],pad_size_x:pad_size_x+latents.shape[-1]]
            sum_after = torch.sum(mask_tmp)
            if sum_after != sum_before:
                raise ValueError('Resize out of bounds.')
                exit(0)
        elif resize_scale<1:
            temp = torch.zeros(1,1,latents.shape[-3],latents.shape[-2], latents.shape[-1]).to(latents.device, dtype=latents.dtype)
            temp[:,:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
            mask_tmp =(temp>0.5).float()
            temp = torch.zeros_like(latents)
            temp[:,:,:,pad_size_y:pad_size_y+latents_tmp.shape[-2],pad_size_x:pad_size_x+latents_tmp.shape[-1]]=latents_tmp
            latents_tmp = temp
            
        latents = (latents*(1-mask_tmp)+latents_tmp*mask_tmp).to(dtype=latents.dtype)
        
        # 对Mask更加简化一些,应该把它直接放缩就可以了
        
        # 设置process_move
        h, w = image.shape[2], image.shape[3] 
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        input_scale = 1 # 这块得再看看
        # from IPython import embed;embed()
        # mask_process = (mask>0).int().cpu().numpy()
        edit_kwargs = process_move(
            path_mask=mask, #就是物体单独的mask
            h=h, #960   
            w=w, #640
            dx=0, #位置变化情况
            dy=0, #位置变化情况
            scale=scale, #
            input_scale=input_scale, #图像边的放缩倍数
            resize_scale=resize_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=4, 
            w_content=6, 
            w_contrast=0.2, 
            w_inpaint=0.8,  
            precision=self.precision, 
            path_mask_ref=None
        )
        
        
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        latent_noise_ref = ddim_latents
        
        SDE_strength = 0.4
        SDE_strength_un = 0
        energy_scale = 0.5
        # from IPython import embed; embed()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(timesteps)):
                next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
                next_timestep = max(next_timestep, 0)
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # print("test1")
                # from IPython import embed; embed()
                if energy_scale==0:
                    repeat=1
                elif 20<i<30 and i%2==0 : 
                    repeat = 3
                else:
                    repeat = 1
                for ri in range(repeat):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample.to(dtype=latents_dtype)
                    
                    # try:
                    #     guidance = self.guidance_move(latent=latents.to(latents_dtype), latent_noise_ref=latent_noise_ref[-(i+1)], t=t, estimator = estimator, text_embeddings=text_embeddings_org[1].unsqueeze(0), energy_scale=500, **edit_kwargs) 
                    # except RuntimeError as e:
                    #     from IPython import embed;embed()
        
                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    # guidance = self.guidance_move(latent=latents.to(latents_dtype), latent_noise_ref=latent_noise_ref[-(i+1)], t=t, estimator = estimator, text_embeddings=text_embeddings_org[1].unsqueeze(0), energy_scale=500, **edit_kwargs) 
                    
                    guidance = self.guidance_move(latent=latents.to(latents_dtype), latent_noise_ref=latent_noise_ref[-(i+1)], t=t, estimator = estimator, text_embeddings=edit_obj_text_embeddings[1].unsqueeze(0), energy_scale=500, **edit_kwargs) 
                    noise_pred = noise_pred+100*guidance
                    # from IPython import embed; embed()
                    
                    # noise_pred = noise_pred+guidance
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    # from IPython import embed; embed()
                    # Edit the latents using attention map
                    # if controller is not None: 
                    #     dtype = latents.dtype
                    #     latents_new = controller.step_callback(latents)
                    #     latents = latents_new.to(dtype)
                    # # call the callback, if provided
                    # if i == len(timesteps) - 1 or (
                    #     (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    # ):
                    #     progress_bar.update()
                    #     if callback is not None and i % callback_steps == 0:
                    #         callback(i, t, latents)
                    # torch.cuda.empty_cache()
        
        # 上面是8.02之前可以复现的 
        
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        #     for i, t in enumerate(tqdm(timesteps)):
        #         # expand the latents if we are doing classifier free guidance
        #         next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        #         next_timestep = max(next_timestep, 0)
                
        #         if energy_scale==0: # 这步也不用
        #             repeat=1
        #         elif 20<i<30 and i%2==0 :  
        #             repeat = 1
        #         else:
        #             repeat = 1
        #         stack = []
                
        #         for ri in range(repeat):
        #             latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        #             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # [2, 4, 8, 64, 64]
        #             with torch.no_grad():
        #                 # 
        #                 # print(i)
                        
        #                 # if i==22:
        #                 # print("test1")
        #                 # from IPython import embed; embed()
                        
        #                 noise_pred = self.unet(
        #                     latent_model_input, t, encoder_hidden_states=text_embeddings
        #                 ).sample.to(dtype=latents_dtype)
        #                 # [2, 4, 8, 64, 64] 541 [2, 77, 768]
        #             if do_classifier_free_guidance:
        #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #                 noise_pred = noise_pred_uncond + guidance_scale * (
        #                     noise_pred_text - noise_pred_uncond
        #                 )
                    
        #             guidance = self.guidance_move(latent=latents, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, estimator = estimator, text_embeddings=text_embeddings_org[1].unsqueeze(0), energy_scale=500, **edit_kwargs) 
        #             noise_pred = noise_pred + guidance
    #   8.03 下午两点半改的
        #             if energy_scale!=0 and i<30 and (i%2==0 or i<10):
        #             #     # print("tets2")
        #             #     # print(i)
        #             #     # from IPython import embed; embed()
        #                 noise_pred_org = noise_pred
        #                 # guidance = self.guidance_move(latent=latents.to(latents_dtype), latent_noise_ref=latent_noise_ref[-(i+1)], t=t, estimator = estimator, text_embeddings=text_embeddings_org[1].unsqueeze(0), energy_scale=500, **edit_kwargs) 
        #                 # noise_pred = noise_pred + guidance
        #             else:
        #                 noise_pred_org = None

        #             prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        #             alpha_prod_t = self.scheduler.alphas_cumprod[t]
        #             alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        #             beta_prod_t = 1 - alpha_prod_t
        #             pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                    
        #             if 10<i<20:
        #                 eta, eta_rd = SDE_strength_un, SDE_strength
        #             else:
        #                 eta, eta_rd = 0., 0.
                    
        #             variance = self.scheduler._get_variance(t, prev_timestep)
        #             std_dev_t = eta * variance ** (0.5)
        #             std_dev_t_rd = eta_rd * variance ** (0.5)
                    
        #             if noise_pred_org is not None:
        #                 pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
        #                 pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
        #             else:
        #                 pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
        #                 pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

        #             latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        #             latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd
        #             # if i>10:
        #                 # from IPython import embed; embed()
        #             if (eta_rd > 0 or eta>0):
        #                 variance_noise = torch.randn_like(latent_prev)
        #                 variance_rd = std_dev_t_rd * variance_noise
        #                 variance = std_dev_t * variance_noise
                    
                        
        #                 mask = (F.interpolate(edit_kwargs["mask_x0"][None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
        #                 mask = ((edit_kwargs["mask_cur"].transpose(0,1)+mask)>0.5).float()
        #                 mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latents.dtype)
        #                 latent_prev = (latent_prev+variance)*(1-mask) + (latent_prev_rd+variance_rd)*mask
        #                 latents = latent_prev
                    
        # #             if controller is not None: 
        # #                 dtype = latents.dtype
        # #                 latents_new = controller.step_callback(latents)
        # #                 latents = latents_new.to(dtype)
        #             if repeat>1:# 这些步骤都是在中间几个step才会用到的
        #                 with torch.no_grad():
        #                     alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
        #                     alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
        #                     beta_prod_t = 1 - alpha_prod_t
        #                     # 这块还是有点问题
        #                     # from IPython import embed; embed()
        #                     model_output = self.unet(latent_prev, next_timestep, encoder_hidden_states=text_embeddings[1].unsqueeze(0))["sample"].squeeze(2)
        #                     next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        #                     next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        #                     latents = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
                

                
                # print("test videomodel3")
                # from IPython import embed; embed()
                # latent_model_input 2, 4, 8, 64, 64
                # latent_noise_ref[-(i+1)] 1, 4, 8, 64, 64
                # latents 1, 4, 8, 64, 64
                        
                # if 20<i<30 and i%2==0 :  
                #     repeat = 3
                # else:
                #     repeat = 1

                # predict the noise residual
                
                #  2, 4, 8, 64, 64
                

                # perform guidance
                # if do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #     noise_pred = noise_pred_uncond + guidance_scale * (
                #         noise_pred_text - noise_pred_uncond
                #     )
                # noise_pred = noise_pred + guidance
                
                # prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                # next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
                # next_timestep = max(next_timestep, 0)
                # if energy_scale!=0 and i<30 and (i%2==0 or i<10):
                #     guidance = self.guidance_move(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                #     noise_pred = noise_pred + guidance
                
                # prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                # alpha_prod_t = self.scheduler.alphas_cumprod[t]
                # beta_prod_t = 1 - alpha_prod_t
                # pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                # if 10<i<20:
                #     eta, eta_rd = SDE_strength_un, SDE_strength
                # else:
                #     eta, eta_rd = 0., 0.
                
                # variance = self.scheduler._get_variance(t, prev_timestep)
                # std_dev_t = eta * variance ** (0.5)
                # std_dev_t_rd = eta_rd * variance ** (0.5)
                # if noise_pred_org is not None:
                #     pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
                #     pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
                # else:
                #     pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                #     pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred
                
                # latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                # latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd
                
                # if (eta_rd > 0 or eta>0):
                #     variance_noise = torch.randn_like(latent_prev)
                #     variance_rd = std_dev_t_rd * variance_noise
                #     variance = std_dev_t * variance_noise
                #     mask = (F.interpolate(edit_kwargs["mask_x0"][None,None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
                #     mask = ((edit_kwargs["mask_cur"]+mask)>0.5).float()
                #     mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                
                # if repeat>1:# 这些步骤都是在中间几个step才会用到的
                #     with torch.no_grad():
                #         alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                #         alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                #         beta_prod_t = 1 - alpha_prod_t
                #         model_output = self.unet(latent_prev.unsqueeze(2), next_timestep, encoder_hidden_states=text_embeddings, mask=dict_mask, save_kv=False, mode=mode, iter_cur=-2)["sample"].squeeze(2)
                #         next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                #         next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                #         latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
                
                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Edit the latents using attention map
                # from IPython import embed; embed()
                # if controller is not None: 
                #     dtype = latents.dtype
                #     latents_new = controller.step_callback(latents)
                #     latents = latents_new.to(dtype)
                # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0: #这里的callback就是none
                            callback(i, t, latents)
                    torch.cuda.empty_cache()
        # 8. Post-processing
        # from IPython import embed;embed()
        image = self.decode_latents(latents.to(latents_dtype))

        # 9. Run safety checker
        has_nsfw_concept = None

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def print_pipeline(self, logger):
        print('Overview function of pipeline: ')
        print(self.__class__)

        print(self)
        
        expected_modules, optional_parameters = self._get_signature_keys(self)        
        components_details = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }
        import json
        logger.info(str(components_details))
        # logger.info(str(json.dumps(components_details, indent = 4)))
        # print(str(components_details))
        # print(self._optional_components)
        
        print(f"python version {sys.version}")
        print(f"torch version {torch.__version__}")
        print(f"validate gpu status:")
        print( torch.tensor(1.0).cuda()*2)
        os.system("nvcc --version")

        import diffusers
        print(diffusers.__version__)
        print(diffusers.__file__)

        try:
            import bitsandbytes
            print(bitsandbytes.__file__)
        except:
            print("fail to import bitsandbytes")
        # os.system("accelerate env")
        # os.system("python -m xformers.info")
        
    def guidance_move(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
        ):
        
        cos = nn.CosineSimilarity(dim=1)
        loss_scale = [0.5, 0.5]
        with torch.no_grad():
            up_ft_tar = estimator( # video_diffusion.models.estimator.MyUNetPseudo3DConditionModel
                        sample=latent_noise_ref.squeeze(2), # [1, 4, 8, 64, 64]
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings.cuda())['up_ft']
            up_ft_tar_org = copy.deepcopy(up_ft_tar) # 2
            for f_id in range(len(up_ft_tar_org)):
                # 这里注意其实还是有点问题
                d1,d2,d3,d4,d5 = up_ft_tar_org[f_id].shape
                up_ft_tar_org[f_id] = up_ft_tar_org[f_id].view(d1*d2,d3,d4,d5)
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale)) #1280, 8, 32, 32->1280, 8, 128, 128
                _,d3,d4,d5 = up_ft_tar_org[f_id].shape
                up_ft_tar_org[f_id] = up_ft_tar_org[f_id].view(d1,d2,d3,d4,d5)
        # [1, 1280, 8, 128, 128], [1, 640, 8, 128, 128]
        # self.estimator.train()
        # print("test videomodel4")
        # from IPython import embed; embed()
        # for param in self.estimator.parameters():
        #     param.requires_grad = True
        
        # print("test_move")
        # from IPython import embed; embed()
        
        latent = latent.detach().requires_grad_(True) # 2, 4, 8, 64, 64]
        for f_id in range(len(up_ft_tar)):
            d1,d2,d3,d4,d5 = up_ft_tar[f_id].shape
            up_ft_tar[f_id] = up_ft_tar[f_id].view(d1*d2,d3,d4,d5)
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))
            _,d3,d4,d5 = up_ft_tar[f_id].shape
            up_ft_tar[f_id] = up_ft_tar[f_id].view(d1,d2,d3,d4,d5)
        # print("test videomodel4")
        # from IPython import embed; embed()
        
        text_embeddings = text_embeddings.detach().requires_grad_(True)
        up_ft_cur = estimator(
                    sample=latent[:,:,:2,:,:],
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft'] # 1, 1280, 8, 32, 32][1, 640, 8, 64, 64]
        up_ft_cur1 = estimator(
                    sample=latent[:,:,2:4,:,:],
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        up_ft_cur2 = estimator(
                    sample=latent[:,:,4:6,:,:],
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        up_ft_cur3 = estimator(
                    sample=latent[:,:,6:,:,:],
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        up_ft_cur[0] = torch.cat((up_ft_cur[0],up_ft_cur1[0],up_ft_cur2[0],up_ft_cur3[0]),dim=2)
        up_ft_cur[1] = torch.cat((up_ft_cur[1],up_ft_cur1[1],up_ft_cur2[1],up_ft_cur3[1]),dim=2)
        
        
        for f_id in range(len(up_ft_cur)):
            d1,d2,d3,d4,d5 = up_ft_cur[f_id].shape
            up_ft_cur[f_id] = up_ft_cur[f_id].view(d1*d2,d3,d4,d5)
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
            _,d3,d4,d5 = up_ft_cur[f_id].shape
            up_ft_cur[f_id] = up_ft_cur[f_id].view(d1,d2,d3,d4,d5)
            
        # print("test videomodel5")
        # from IPython import embed; embed() 
        # editing energy
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,1,up_ft_tar[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id]
         
        
        # content energy
        loss_con = 0
        if mask_x0_ref is not None:
            mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        else:
            mask_x0_ref_cur = mask_other
        for f_id in range(len(up_ft_tar_org)):
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[:,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]

        for f_id in range(len(up_ft_tar)):
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
            sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
            loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_x0_ref_cur.repeat(1,1,up_ft_tar_org[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
            loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())
        
        # print("test videomodel6")
        # from IPython import embed; embed()
        # print("test videomodel5")
        # from IPython import embed; embed()

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0] # [1, 4, 8, 64, 64]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        # mask_edit2 = (F.interpolate(mask_x0[None,None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
        mask_edit2 = (F.interpolate(mask_x0[None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float() # [ 1, 8, 128, 128]
        mask_edit1 = (mask_cur>0.5).float().transpose(0,1) # [1, 8 , 128, 128]
        mask = ((mask_cur.transpose(0,1)+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        estimator.zero_grad()

        return guidance