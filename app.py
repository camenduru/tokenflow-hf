import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from utils import video_to_frames, add_dict_to_yaml_file, save_video, seed_everything
# from diffusers.utils import export_to_video
from tokenflow_pnp import TokenFlow
from preprocess_utils import *
from tokenflow_utils import *
# load sd model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "stabilityai/stable-diffusion-2-1-base"

# components for the Preprocessor
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision="fp16",
                                                 torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", revision="fp16",
                                                  torch_dtype=torch.float16).to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", revision="fp16",
                                           torch_dtype=torch.float16).to(device)

# pipe for TokenFlow
tokenflow_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
tokenflow_pipe.enable_xformers_memory_efficient_attention()

def randomize_seed_fn():
    seed = random.randint(0, np.iinfo(np.int32).max)
    return seed
    
def reset_do_inversion():
    return True

def get_example():
    case = [
        [
            'examples/wolf.mp4',     
        ],
        [
            'examples/woman-running.mp4',     
        ],
        [
            'examples/cutting_bread.mp4',
        ],
        [
            'examples/running_dog.mp4',
        ]
    ]
    return case


def prep(config):
    # timesteps to save
    if config["sd_version"] == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif config["sd_version"] == '2.0':
        model_key = "stabilityai/stable-diffusion-2-base"
    elif config["sd_version"] == '1.5' or config["sd_version"] == 'ControlNet':
        model_key = "runwayml/stable-diffusion-v1-5"
    elif config["sd_version"] == 'depth':
        model_key = "stabilityai/stable-diffusion-2-depth"
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(config["save_steps"])
    print("config[save_steps]", config["save_steps"])
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=config["save_steps"],
                                                           strength=1.0,
                                                           device=device)
    print("YOOOO timesteps to save", timesteps_to_save)

    # seed_everything(config["seed"])
    if not config["frames"]: # original non demo setting
        save_path = os.path.join(config["save_dir"],
                                 f'sd_{config["sd_version"]}',
                                 Path(config["data_path"]).stem,
                                 f'steps_{config["steps"]}',
                                 f'nframes_{config["n_frames"]}') 
        os.makedirs(os.path.join(save_path, f'latents'), exist_ok=True)
        add_dict_to_yaml_file(os.path.join(config["save_dir"], 'inversion_prompts.yaml'), Path(config["data_path"]).stem, config["inversion_prompt"])    
        # save inversion prompt in a txt file
        with open(os.path.join(save_path, 'inversion_prompt.txt'), 'w') as f:
            f.write(config["inversion_prompt"])
    else:
        save_path = None
    
    model = Preprocess(device, config,
                      vae=vae,
                      text_encoder=text_encoder,
                      scheduler=scheduler,
                      tokenizer=tokenizer,
                      unet=unet)
    print(type(model.config["batch_size"]))
    frames, latents, total_inverted_latents, rgb_reconstruction = model.extract_latents(
                                         num_steps=model.config["steps"],
                                         save_path=save_path,
                                         batch_size=model.config["batch_size"],
                                         timesteps_to_save=timesteps_to_save,
                                         inversion_prompt=model.config["inversion_prompt"],
    )

    
    return frames, latents, total_inverted_latents, rgb_reconstruction
    
def preprocess_and_invert(input_video,
                          frames,
                          latents,
                          inverted_latents,
                          seed, 
                          randomize_seed,
                          do_inversion,
                          # save_dir: str = "latents",
                          steps,
                          n_timesteps = 50,
                          batch_size: int = 8,
                          n_frames: int = 40,
                          inversion_prompt:str = '',
                          
              ):
    sd_version = "2.1"
    height = 512
    weidth: int = 512
    print("n timesteps", n_timesteps)
    if do_inversion or randomize_seed:
        preprocess_config = {}
        preprocess_config['H'] = height
        preprocess_config['W'] = weidth
        preprocess_config['save_dir'] = 'latents'
        preprocess_config['sd_version'] = sd_version
        preprocess_config['steps'] = steps
        preprocess_config['batch_size'] = batch_size
        preprocess_config['save_steps'] = int(n_timesteps)
        preprocess_config['n_frames'] = n_frames
        preprocess_config['seed'] = seed
        preprocess_config['inversion_prompt'] = inversion_prompt
        preprocess_config['frames'] = video_to_frames(input_video)
        preprocess_config['data_path'] = input_video.split(".")[0]
        

        if randomize_seed:
            seed = randomize_seed_fn()
        seed_everything(seed)
        
        frames, latents, total_inverted_latents, rgb_reconstruction = prep(preprocess_config)
        print(total_inverted_latents.keys())
        print(len(total_inverted_latents.keys()))
        frames = gr.State(value=frames)
        latents = gr.State(value=latents)
        inverted_latents = gr.State(value=total_inverted_latents)
        do_inversion = False
   
    return frames, latents, inverted_latents, do_inversion


def edit_with_pnp(input_video,
                  frames, 
                  latents,
                  inverted_latents,
                  seed,
                  randomize_seed,
                  do_inversion,
                  steps,
                  prompt: str = "a marble sculpture of a woman running, Venus de Milo",
                  # negative_prompt: str = "ugly, blurry, low res, unrealistic, unaesthetic",
                  pnp_attn_t: float = 0.5,
                  pnp_f_t: float = 0.8,
                  batch_size: int = 8, #needs to be the same as for preprocess
                  n_frames: int = 40,#needs to be the same as for preprocess
                  n_timesteps: int = 50,
                  gudiance_scale: float = 7.5,
                  inversion_prompt: str = "", #needs to be the same as for preprocess
                  n_fps: int = 10,
                  progress=gr.Progress(track_tqdm=True)
):
    config = {}
 
    config["sd_version"] = "2.1"
    config["device"] = device
    config["n_timesteps"] = int(n_timesteps)
    config["n_frames"] = n_frames
    config["batch_size"] = batch_size
    config["guidance_scale"] = gudiance_scale
    config["prompt"] = prompt
    config["negative_prompt"] = "ugly, blurry, low res, unrealistic, unaesthetic",
    config["pnp_attn_t"] = pnp_attn_t
    config["pnp_f_t"] = pnp_f_t
    config["pnp_inversion_prompt"] = inversion_prompt
    
    
    if do_inversion:
        frames, latents, inverted_latents, do_inversion =  preprocess_and_invert(
                          input_video,
                          frames,
                          latents,
                          inverted_latents,
                          seed, 
                          randomize_seed,
                          do_inversion,
                          steps,
                          n_timesteps,
                          batch_size,
                          n_frames,
                          inversion_prompt)
        do_inversion = False
        
    
    if randomize_seed:
            seed = randomize_seed_fn()
    seed_everything(seed)
    
    
    editor = TokenFlow(config=config,pipe=tokenflow_pipe, frames=frames.value, inverted_latents=inverted_latents.value)
    edited_frames = editor.edit_video()

    save_video(edited_frames, 'tokenflow_PnP_fps_30.mp4', fps=n_fps)
    # path = export_to_video(edited_frames)
    return 'tokenflow_PnP_fps_30.mp4', frames, latents, inverted_latents, do_inversion

########
# demo #
########


intro = """
<div style="text-align:center">
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   TokenFlow - <small>Temporally consistent video editing</small>
</h1>
<span>[<a target="_blank" href="https://diffusion-tokenflow.github.io">Project page</a>], [<a target="_blank" href="https://github.com/omerbt/TokenFlow">GitHub</a>], [<a target="_blank" href="https://huggingface.co/papers/2307.10373">Paper</a>]</span>
<div style="display:flex; justify-content: center;margin-top: 0.5em">Each edit takes ~5 min <a href="https://huggingface.co/weizmannscience/tokenflow?duplicate=true" target="_blank">
<img style="margin-top: 0em; margin-bottom: 0em; margin-left: 0.5em" src="https://bit.ly/3CWLGkA" alt="Duplicate Space"></a></div>
</div>
"""



with gr.Blocks(css="style.css") as demo:
    
    gr.HTML(intro)
    frames = gr.State()
    inverted_latents = gr.State()
    latents = gr.State()
    do_inversion = gr.State(value=True)

    with gr.Row():
        input_video = gr.Video(label="Input Video", interactive=True, elem_id="input_video")
        output_video = gr.Video(label="Edited Video", interactive=False, elem_id="output_video")
        input_video.style(height=365, width=365)
        output_video.style(height=365, width=365)


    with gr.Row():
            prompt = gr.Textbox(
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
            with gr.TabItem('General options'):
                with gr.Row():
                    with gr.Column(min_width=100):
                        seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                        randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
                        gudiance_scale = gr.Slider(label='Guidance Scale', minimum=1, maximum=30,
                                              value=7.5, step=0.5, interactive=True)
                        steps = gr.Slider(label='Inversion steps', minimum=10, maximum=500,
                                              value=500, step=1, interactive=True)
                        
                    with gr.Column(min_width=100):
                        inversion_prompt = gr.Textbox(lines=1, label="Inversion prompt", interactive=True, placeholder="")
                        batch_size = gr.Slider(label='Batch size', minimum=1, maximum=10,
                                              value=8, step=1, interactive=True)
                        n_frames = gr.Slider(label='Num frames', minimum=2, maximum=200,
                                              value=24, step=1, interactive=True)
                        n_timesteps = gr.Slider(label='Diffusion steps', minimum=25, maximum=100,
                                              value=50, step=25, interactive=True)
                        n_fps = gr.Slider(label='Frames per second', minimum=1, maximum=60,
                                              value=10, step=1, interactive=True)
                        
            with gr.TabItem('Plug-and-Play Parameters'):
                 with gr.Column(min_width=100):
                    pnp_attn_t = gr.Slider(label='pnp attention threshold', minimum=0, maximum=1,
                                              value=0.5, step=0.5, interactive=True)
                    pnp_f_t = gr.Slider(label='pnp feature threshold', minimum=0, maximum=1,
                                              value=0.8, step=0.05, interactive=True)
                    
                    
    input_video.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)

    inversion_prompt.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)

    randomize_seed.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)

    seed.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)

    

    input_video.upload(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False).then(fn = preprocess_and_invert,
          inputs = [input_video,
                      frames,
                      latents,
                      inverted_latents,
                      seed, 
                      randomize_seed,
                      do_inversion,
                      steps,
                      n_timesteps,
                      batch_size,
                      n_frames,
                      inversion_prompt
          ],
          outputs = [frames,
                     latents,
                     inverted_latents,
                     do_inversion
              
          ])
    
    run_button.click(fn = edit_with_pnp,
                     inputs = [input_video,
                               frames, 
                              latents,
                              inverted_latents,
                              seed,
                              randomize_seed,
                              do_inversion,
                              steps,
                              prompt,                             
                              pnp_attn_t,
                              pnp_f_t,
                              batch_size,
                              n_frames,
                              n_timesteps,
                              gudiance_scale,
                              inversion_prompt,
                              n_fps ],
                                 outputs = [output_video, frames, latents, inverted_latents, do_inversion]
                                )

    gr.Examples(
        examples=get_example(),
        label='Examples',
        inputs=[input_video],
        outputs=[output_video]
    )

demo.queue()
demo.launch()