import gradio as gr

import modules.config
import modules.html
import modules.gradio_hijack as grh
from modules.ui.generation import stop_clicked, skip_clicked


def create_preview_components(gradio_root):
    components = {}

    with gr.Row():
        progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768,
                                    elem_classes=['main_view'])
        progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                      height=768, visible=False, elem_classes=['main_view', 'image_gallery'])

    progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                            elem_id='progress-bar', elem_classes='progress-bar')

    gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                         elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                         elem_id='final_gallery')

    components['progress_window'] = progress_window
    components['progress_gallery'] = progress_gallery
    components['progress_html'] = progress_html
    components['gallery'] = gallery

    return components


def create_prompt_and_buttons(gradio_root):
    components = {}

    with gr.Row():
        with gr.Column(scale=17):
            prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                autofocus=True, lines=3)

            default_prompt = modules.config.default_prompt
            if isinstance(default_prompt, str) and default_prompt != '':
                gradio_root.load(lambda: default_prompt, outputs=prompt)

        with gr.Column(scale=3, min_width=0):
            generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
            reset_button = gr.Button(label="Reconnect", value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
            load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
            skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
            stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

    components['prompt'] = prompt
    components['generate_button'] = generate_button
    components['reset_button'] = reset_button
    components['load_parameter_button'] = load_parameter_button
    components['skip_button'] = skip_button
    components['stop_button'] = stop_button

    return components


def create_top_checkboxes():
    components = {}

    with gr.Row(elem_classes='advanced_check_row'):
        input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
        enhance_checkbox = gr.Checkbox(label='Enhance', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
        advanced_checkbox = gr.Checkbox(label='Advanced', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')

    components['input_image_checkbox'] = input_image_checkbox
    components['enhance_checkbox'] = enhance_checkbox
    components['advanced_checkbox'] = advanced_checkbox

    return components


def bind_prompt_control_events(components, currentTask):
    stop_button = components['stop_button']
    skip_button = components['skip_button']

    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
