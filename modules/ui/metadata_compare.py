import gradio as gr

import modules.meta_parser
from modules.ui.generation import parse_meta


def trigger_metadata_preview(file):
    if file is not None:
        return modules.meta_parser.read_info_from_image(file)
    else:
        return gr.update(value=None)


def trigger_metadata_import(file, state_is_generating, inpaint_mode):
    if file is None:
        yield {state_is_generating: gr.update(value=state_is_generating.value)}
        return

    if state_is_generating:
        yield {state_is_generating: gr.update(value=state_is_generating.value)}
        return

    raw_prompt_txt = modules.meta_parser.read_info_from_image(file)
    if raw_prompt_txt is None:
        yield {state_is_generating: gr.update(value=state_is_generating.value)}
        return

    for output in parse_meta(raw_prompt_txt, state_is_generating, inpaint_mode):
        yield output


def bind_metadata_events(components, state_is_generating, inpaint_mode, load_data_outputs):
    metadata_input_image = components['metadata_input_image']
    metadata_json = components['metadata_json']
    metadata_import_button = components['metadata_import_button']

    metadata_input_image.upload(
        trigger_metadata_preview, inputs=[metadata_input_image],
        outputs=[metadata_json], show_progress=False, queue=False
    )

    metadata_input_image.clear(
        lambda: gr.update(value=None), outputs=[metadata_json],
        show_progress=False, queue=False
    )

    metadata_import_button.click(
        trigger_metadata_import,
        inputs=[metadata_input_image, state_is_generating, inpaint_mode],
        outputs=load_data_outputs
    )
