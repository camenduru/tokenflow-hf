import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from utils import *






# load sd model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "stabilityai/stable-diffusion-2-1-base"
inv_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
inv_pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

def randomize_seed_fn():
    seed = random.randint(0, np.iinfo(np.int32).max)
    return seed

def preprocess_and_invert(video,
                          frames,
                          latents,
                          inverted_latents,
                          seed, 
                          randomize_seed,
                          do_inversion,
                          height:int = 512, 
                          weidth: int = 512,
                          # save_dir: str = "latents",
                          steps: int = 500,
                          batch_size: int = 8,
                          # save_steps: int = 50,
                          n_frames: int = 40,
                          inversion_prompt:str = ''
              ):

    if do_inversion or randomize_seed:
        
        # save_video_frames(data_path, img_size=(height, weidth))
        frames = video_to_frames(video, img_size=(height, weidth))
        # data_path = os.path.join('data', Path(video_path).stem)
    
        toy_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        toy_scheduler.set_timesteps(save_steps)
        timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=save_steps,
                                                               strength=1.0,
                                                               device=device)
        if randomize_seed:
            seed = randomize_seed_fn()
        seed_everything(seed)
        
        frames, latents = get_data(inv_pipe, frames, n_frames)

        inverted_latents = extract_latents(inv_pipe, num_steps = steps, 
                                           latent_frames = latents, 
                                           batch_size = batch_size, 
                                           timesteps_to_save = timesteps_to_save,
                                           inversion_prompt = inversion_prompt,)
        frames = gr.State(value=frames)
        latents = gr.State(value=latents)
        inverted_latents = gr.State(value=inverted_latents)
        do_inversion = False
   

    return frames, latents, inverted_latents, do_inversion



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
    frames = gr.State()
    inverted_latents = gr.State()
    latents = gr.State()
    do_inversion = gr.State(value=True)

    with gr.Row():
        input_vid = gr.Video(label="Input Video", interactive=True, elem_id="input_video")
        output_vid = gr.Video(label="Edited Video", interactive=False, elem_id="output_video")
        input_vid.style(height=365, width=365)
        output_vid.style(height=365, width=365)


    with gr.Row():
            tar_prompt = gr.Textbox(
                            label="Describe your edited video",
                            max_lines=1, value=""
                        )
    # with gr.Group(visible=False) as share_btn_container:
        # with gr.Group(elem_id="share-btn-container"):
        #     community_icon = gr.HTML(community_icon_html, visible=True)
        #     loading_icon = gr.HTML(loading_icon_html, visible=False)
        #     share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
        
   
    # with gr.Row():
    #     inversion_progress = gr.Textbox(visible=False, label="Inversion progress")
               
    with gr.Row():
        run_button = gr.Button("Edit your video!", visible=True)

    with gr.Accordion("Advanced Options", open=False):
      with gr.Tabs() as tabs:

          with gr.TabItem('General options', id=2):
            with gr.Row():
                with gr.Column(min_width=100):
                    seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                    randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
                    steps = gr.Slider(label='Inversion steps', minimum=100, maximum=500,
                                          value=500, step=1, interactive=True)
                with gr.Column(min_width=100):
                    inversion_prompt = gr.Textbox(lines=1, label="Inversion prompt", interactive=True, placeholder="")
                    batch_size = gr.Slider(label='Batch size', minimum=1, maximum=10,
                                          value=8, step=1, interactive=True)
                    n_frames = gr.Slider(label='Num frames', minimum=20, maximum=200,
                                          value=40, step=1, interactive=True)
                    
    input_vid.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)

    input_vid.upload(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)
    ).then(fn = preprocess_and_invert,
          inputs = [input_vid,
                      frames,
                      latents,
                      inverted_latents,
                      seed, 
                      randomize_seed,
                      do_inversion,
                      steps,
                      batch_size,
                      n_frames,
                      inversion_prompt
          ],
          outputs = [frames,
                     latents,
                     inverted_latents,
                     do_inversion
              
          ])



demo.queue()
demo.launch()