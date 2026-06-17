import gradio as gr

import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
from modules.ui.generation import (
    trigger_metadata_preview, trigger_metadata_import,
    trigger_compare, trigger_compare_fill
)


def create_metadata_tab():
    components = {}

    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
        components['metadata_tab'] = metadata_tab
        with gr.Column():
            metadata_input_image = grh.Image(label='For images created by Fooocus', source='upload', type='filepath')
            metadata_json = gr.JSON(label='Metadata')
            metadata_import_button = gr.Button(value='Apply Metadata')

    components['metadata_input_image'] = metadata_input_image
    components['metadata_json'] = metadata_json
    components['metadata_import_button'] = metadata_import_button

    return components


def create_compare_tab():
    components = {}

    with gr.Tab(label='Parameter Compare', id='compare_tab') as compare_tab:
        components['compare_tab'] = compare_tab
        with gr.Row():
            with gr.Column():
                compare_image_left = grh.Image(label='Base Image (left side)', source='upload', type='filepath')
                components['compare_image_left'] = compare_image_left
            with gr.Column():
                compare_image_right = grh.Image(label='Target Image (right side, to import)', source='upload', type='filepath')
                components['compare_image_right'] = compare_image_right

        with gr.Row():
            compare_run_button = gr.Button(value='Compare Parameters', variant='primary')
            compare_fill_diff_button = gr.Button(value='Apply only different parameters')
            compare_fill_all_button = gr.Button(value='Apply all target parameters')

        components['compare_run_button'] = compare_run_button
        components['compare_fill_diff_button'] = compare_fill_diff_button
        components['compare_fill_all_button'] = compare_fill_all_button

        with gr.Row():
            compare_summary_html = gr.HTML(value='Upload two Fooocus images and click Compare.')
            components['compare_summary_html'] = compare_summary_html
        with gr.Row():
            compare_diff_json = gr.JSON(label='Parameter Differences')
            components['compare_diff_json'] = compare_diff_json

    return components


def bind_metadata_events(components, state_is_generating, inpaint_mode,
                          load_data_outputs, style_selections):
    metadata_input_image = components['metadata_input_image']
    metadata_json = components['metadata_json']
    metadata_import_button = components['metadata_import_button']

    metadata_input_image.upload(
        trigger_metadata_preview, inputs=[metadata_input_image],
        outputs=[metadata_json], queue=False, show_progress=True)

    metadata_import_button.click(
        trigger_metadata_import,
        inputs=[metadata_input_image, state_is_generating, inpaint_mode],
        outputs=load_data_outputs, queue=False, show_progress=True
    ).then(
        style_sorter.sort_styles, inputs=style_selections, outputs=style_selections,
        queue=False, show_progress=False
    )


def bind_compare_events(components, state_is_generating, inpaint_mode,
                         load_data_outputs, style_selections):
    compare_image_left = components['compare_image_left']
    compare_image_right = components['compare_image_right']
    compare_run_button = components['compare_run_button']
    compare_fill_diff_button = components['compare_fill_diff_button']
    compare_fill_all_button = components['compare_fill_all_button']
    compare_summary_html = components['compare_summary_html']
    compare_diff_json = components['compare_diff_json']

    compare_run_button.click(
        trigger_compare,
        inputs=[compare_image_left, compare_image_right],
        outputs=[compare_summary_html, compare_diff_json],
        queue=False, show_progress=True
    )

    compare_fill_diff_button.click(
        lambda l, r, g: trigger_compare_fill(l, r, g, 'diff_only', inpaint_mode),
        inputs=[compare_image_left, compare_image_right, state_is_generating],
        outputs=load_data_outputs, queue=False, show_progress=True
    ).then(
        style_sorter.sort_styles, inputs=style_selections, outputs=style_selections,
        queue=False, show_progress=False
    )

    compare_fill_all_button.click(
        lambda l, r, g: trigger_compare_fill(l, r, g, 'target_all', inpaint_mode),
        inputs=[compare_image_left, compare_image_right, state_is_generating],
        outputs=load_data_outputs, queue=False, show_progress=True
    ).then(
        style_sorter.sort_styles, inputs=style_selections, outputs=style_selections,
        queue=False, show_progress=False
    )
