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
    
    with gr.Group(visible=False) as share_btn_container:
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=True)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
        
   
    with gr.Row():
        inversion_progress = gr.Textbox(visible=False, label="Inversion progress")
        

        
    with gr.Row():
        run_button = gr.Button("Edit your video!", visible=True)
        

    # with gr.Accordion("Advanced Options", open=False):



demo.queue()
demo.launch()