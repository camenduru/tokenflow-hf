import gradio as gr
import os
from torchvision.io import read_video,write_video
def video_identity(video):
    video, _, _ = read_video(video, output_format="TCHW")
    return video

with gr.Blocks() as demo:
  vid = gr.Video()
  vid_out = gr.Video()
  text = gr.Textbox()
  vid.upload(fn = video_identity, inputs = [vid], outputs=[vid_out])
# demo = gr.Interface(video_identity, 
#                     gr.Video(), 
#                     "playable_video", 
#                     # examples=[
#                     #     os.path.join(os.path.abspath(''), 
#                     #                  "video/video_sample.mp4")], 
#                     )

if __name__ == "__main__":
    demo.launch()