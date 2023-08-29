import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler



# load sd model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "stabilityai/stable-diffusion-2-1-base"
inv_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
inv_pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")


def preprocess(data_path:str = 'examples/woman-running.mp4',
              height:int = 512, 
              weidth: int = 512,
              # save_dir: str = "latents",
              steps: int = 500,
               batch_size: int = 8,
               save_steps: int = 50,
               n_frames: int = 40,
               inversion_prompt:str = ''
              ):
        
    # save_video_frames(data_path, img_size=(height, weidth))
    frames = video_to_frames(data_path, img_size=(height, weidth))
    # data_path = os.path.join('data', Path(video_path).stem)

    toy_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    toy_scheduler.set_timesteps(save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=save_steps,
                                                           strength=1.0,
                                                           device=device)
    seed_everything(1)
    
    frames, latents = get_data(inv_pipe, frames, n_frames)
    # inverted_latents = noisy_latents
    inverted_latents = extract_latents(inv_pipe, num_steps = steps, 
                                       latent_frames = latents, 
                                       batch_size = batch_size, 
                                       timesteps_to_save = timesteps_to_save,
                                       inversion_prompt = inversion_prompt,)
   




    return frames, latents, inverted_latents

import gradio as gr

########
# demo #
########


intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   TokenFlow
</h1>
"""



with gr.Blocks(css="style.css") as demo:
    
    gr.HTML(intro)


    with gr.Row():
        input_vid = gr.Video(label="Input Video", interactive=True, elem_id="input_video")
        output_vid = gr.Image(label="Edited Video", interactive=False, elem_id="output_video")
        input_vid.style(height=365, width=365)
        output_vid.style(height=365, width=365)
    
    # with gr.Group(visible=False) as share_btn_container:
        # with gr.Group(elem_id="share-btn-container"):
        #     community_icon = gr.HTML(community_icon_html, visible=True)
        #     loading_icon = gr.HTML(loading_icon_html, visible=False)
        #     share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
        
   
    with gr.Row():
        inversion_progress = gr.Textbox(visible=False, label="Inversion progress")
        

        
    with gr.Row():
        run_button = gr.Button("Edit your video!", visible=True)
        

    # with gr.Accordion("Advanced Options", open=False):



demo.queue()
demo.launch()