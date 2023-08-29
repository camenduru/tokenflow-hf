from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
# suppress partial model loading warning
logging.set_verbosity_error()

import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from torchvision.io import write_video
from pathlib import Path
from util import *
import torchvision.transforms as T


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start
    
@torch.no_grad()
def decode_latents(pipe, latents):
    decoded = []
    batch_size = 8
    for b in range(0, latents.shape[0], batch_size):
            latents_batch = 1 / 0.18215 * latents[b:b + batch_size]
            imgs = pipe.vae.decode(latents_batch).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            decoded.append(imgs)
    return torch.cat(decoded)

@torch.no_grad()
def ddim_inversion(pipe, cond, latent_frames,  batch_size, save_latents=True, timesteps_to_save=None):
    
    timesteps = reversed(pipe.scheduler.timesteps)
    timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, latent_frames.shape[0], batch_size):
            x_batch = latent_frames[b:b + batch_size]
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
            #remove comment from commented block to support controlnet
            # if self.sd_version == 'depth':
            #     depth_maps = torch.cat([self.depth_maps[b: b + batch_size]])
            #     model_input = torch.cat([x_batch, depth_maps],dim=1)

            alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                pipe.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else pipe.scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            
            #remove line below and replace with commented block to support controlnet
            eps = pipe.unet(model_input, t, encoder_hidden_states=cond_batch).sample
            # if self.sd_version != 'ControlNet':
            #     eps = pipe.unet(model_input, t, encoder_hidden_states=cond_batch).sample
            # else:
            #     eps = self.controlnet_pred(x_batch, t, cond_batch, torch.cat([self.canny_cond[b: b + batch_size]]))
            
            pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
            latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps

    #     if save_latents and t in timesteps_to_save:
    #         torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
    # torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
    return latent_frames    
    
@torch.no_grad()
def ddim_sample(pipe, x, cond, batch_size):
    timesteps = pipe.scheduler.timesteps
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, x.shape[0], batch_size):
            x_batch = x[b:b + batch_size]
            model_input = x_batch
            cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
            
            #remove comment from commented block to support controlnet
            # if self.sd_version == 'depth':
            #     depth_maps = torch.cat([self.depth_maps[b: b + batch_size]])
            #     model_input = torch.cat([x_batch, depth_maps],dim=1)

            alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                pipe.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else pipe.scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            #remove line below and replace with commented block to support controlnet
            eps = pipe.unet(model_input, t, encoder_hidden_states=cond_batch).sample
            # if self.sd_version != 'ControlNet':
            #     eps = pipe.unet(model_input, t, encoder_hidden_states=cond_batch).sample
            # else:
            #     eps = self.controlnet_pred(x_batch, t, cond_batch, torch.cat([self.canny_cond[b: b + batch_size]]))

            pred_x0 = (x_batch - sigma * eps) / mu
            x[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
    return x


@torch.no_grad()
def get_text_embeds(pipe, prompt, negative_prompt, batch_size=1, device="cuda"):
    # Tokenize text and get embeddings
    text_input = pipe.tokenizer(prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]

    # Do the same for unconditional embeddings
    uncond_input = pipe.tokenizer(negative_prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length,
                                  return_tensors='pt')

    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
    return text_embeddings

@torch.no_grad()
def extract_latents(pipe,
                    num_steps,
                    latent_frames,
                    batch_size,
                    timesteps_to_save,
                    inversion_prompt=''):
    pipe.scheduler.set_timesteps(num_steps)
    cond = get_text_embeds(pipe, inversion_prompt, "", device=pipe.device)[1].unsqueeze(0)
    # latent_frames = self.latents

    inverted_latents = ddim_inversion(pipe, cond,
                                latent_frames,
                                batch_size=batch_size,
                                save_latents=False,
                                timesteps_to_save=timesteps_to_save)
    
    # latent_reconstruction = ddim_sample(pipe, inverted_latents, cond, batch_size=batch_size)

#     rgb_reconstruction = decode_latents(pipe, latent_reconstruction)

#     return rgb_reconstruction
    return inverted_latents
    
@torch.no_grad()
def encode_imgs(pipe, imgs, batch_size=10, deterministic=True):
    imgs = 2 * imgs - 1
    latents = []
    for i in range(0, len(imgs), batch_size):
        posterior = pipe.vae.encode(imgs[i:i + batch_size]).latent_dist
        latent = posterior.mean if deterministic else posterior.sample()
        latents.append(latent * 0.18215)
    latents = torch.cat(latents)
    return latents
    
def get_data(pipe, frames, n_frames):
    """
    converts frames to tensors, saves to device and encodes to obtain latents
    """
    frames = frames[:n_frames]
    if frames[0].size[0] == frames[0].size[1]:
        frames = [frame.convert("RGB").resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
    stacked_tensor_frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(pipe.device)
    # encode to latents
    latents = encode_imgs(pipe, stacked_tensor_frames, deterministic=True).to(torch.float16).to(pipe.device)
    return stacked_tensor_frames, latents

