import threading
import time
import os
import traceback

from modules.patch import patch_all
import modules.config

from modules.worker.context import GenerationContext, SingleTaskContext, WorkerRuntime
from modules.worker.pipeline import execute_pipeline, PipelineState, DEFAULT_PIPELINE
from modules.worker.result_saver import build_image_wall_from_results

patch_all()


class AsyncTask:
    def __init__(self, args):
        from modules.flags import Performance, MetadataScheme, ip_list, disabled
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import args_manager

        self.args = args.copy()
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False

        self.performance_loras = []

        if len(args) == 0:
            return

        args.reverse()
        self.generate_image_grid = args.pop()
        self.prompt = args.pop()
        self.negative_prompt = args.pop()
        self.style_selections = args.pop()

        self.performance_selection = Performance(args.pop())
        self.steps = self.performance_selection.steps()
        self.original_steps = self.steps

        self.aspect_ratios_selection = args.pop()
        self.image_number = args.pop()
        self.output_format = args.pop()
        self.seed = int(args.pop())
        self.read_wildcards_in_order = args.pop()
        self.sharpness = args.pop()
        self.cfg_scale = args.pop()
        self.base_model_name = args.pop()
        self.refiner_model_name = args.pop()
        self.refiner_switch = args.pop()
        self.loras = get_enabled_loras([(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in
                                        range(default_max_lora_number)])
        self.input_image_checkbox = args.pop()
        self.current_tab = args.pop()
        self.uov_method = args.pop()
        self.uov_input_image = args.pop()
        self.outpaint_selections = args.pop()
        self.inpaint_input_image = args.pop()
        self.inpaint_additional_prompt = args.pop()
        self.inpaint_mask_image_upload = args.pop()

        self.disable_preview = args.pop()
        self.disable_intermediate_results = args.pop()
        self.disable_seed_increment = args.pop()
        self.black_out_nsfw = args.pop()
        self.adm_scaler_positive = args.pop()
        self.adm_scaler_negative = args.pop()
        self.adm_scaler_end = args.pop()
        self.adaptive_cfg = args.pop()
        self.clip_skip = args.pop()
        self.sampler_name = args.pop()
        self.scheduler_name = args.pop()
        self.vae_name = args.pop()
        self.overwrite_step = args.pop()
        self.overwrite_switch = args.pop()
        self.overwrite_width = args.pop()
        self.overwrite_height = args.pop()
        self.overwrite_vary_strength = args.pop()
        self.overwrite_upscale_strength = args.pop()
        self.mixing_image_prompt_and_vary_upscale = args.pop()
        self.mixing_image_prompt_and_inpaint = args.pop()
        self.debugging_cn_preprocessor = args.pop()
        self.skipping_cn_preprocessor = args.pop()
        self.canny_low_threshold = args.pop()
        self.canny_high_threshold = args.pop()
        self.refiner_swap_method = args.pop()
        self.controlnet_softness = args.pop()
        self.freeu_enabled = args.pop()
        self.freeu_b1 = args.pop()
        self.freeu_b2 = args.pop()
        self.freeu_s1 = args.pop()
        self.freeu_s2 = args.pop()
        self.debugging_inpaint_preprocessor = args.pop()
        self.inpaint_disable_initial_latent = args.pop()
        self.inpaint_engine = args.pop()
        self.inpaint_strength = args.pop()
        self.inpaint_respective_field = args.pop()
        self.inpaint_advanced_masking_checkbox = args.pop()
        self.invert_mask_checkbox = args.pop()
        self.inpaint_erode_or_dilate = args.pop()
        self.save_final_enhanced_image_only = args.pop() if not args_manager.args.disable_image_log else False
        self.save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
        self.metadata_scheme = MetadataScheme(
            args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        self.cn_tasks = {x: [] for x in ip_list}
        for _ in range(modules.config.default_controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        self.debugging_dino = args.pop()
        self.dino_erode_or_dilate = args.pop()
        self.debugging_enhance_masks_checkbox = args.pop()

        self.enhance_input_image = args.pop()
        self.enhance_checkbox = args.pop()
        self.enhance_uov_method = args.pop()
        self.enhance_uov_processing_order = args.pop()
        self.enhance_uov_prompt_type = args.pop()
        self.enhance_ctrls = []
        for _ in range(modules.config.default_enhance_tabs):
            enhance_enabled = args.pop()
            enhance_mask_dino_prompt_text = args.pop()
            enhance_prompt = args.pop()
            enhance_negative_prompt = args.pop()
            enhance_mask_model = args.pop()
            enhance_mask_cloth_category = args.pop()
            enhance_mask_sam_model = args.pop()
            enhance_mask_text_threshold = args.pop()
            enhance_mask_box_threshold = args.pop()
            enhance_mask_sam_max_detections = args.pop()
            enhance_inpaint_disable_initial_latent = args.pop()
            enhance_inpaint_engine = args.pop()
            enhance_inpaint_strength = args.pop()
            enhance_inpaint_respective_field = args.pop()
            enhance_inpaint_erode_or_dilate = args.pop()
            enhance_mask_invert = args.pop()
            if enhance_enabled:
                self.enhance_ctrls.append([
                    enhance_mask_dino_prompt_text,
                    enhance_prompt,
                    enhance_negative_prompt,
                    enhance_mask_model,
                    enhance_mask_cloth_category,
                    enhance_mask_sam_model,
                    enhance_mask_text_threshold,
                    enhance_mask_box_threshold,
                    enhance_mask_sam_max_detections,
                    enhance_inpaint_disable_initial_latent,
                    enhance_inpaint_engine,
                    enhance_inpaint_strength,
                    enhance_inpaint_respective_field,
                    enhance_inpaint_erode_or_dilate,
                    enhance_mask_invert
                ])
        self.should_enhance = self.enhance_checkbox and (self.enhance_uov_method != disabled.casefold() or len(self.enhance_ctrls) > 0)
        self.images_to_enhance_count = 0
        self.enhance_stats = {}


async_tasks = []


class EarlyReturnException(BaseException):
    pass


def stop_processing(async_task, processing_start_time, time_module):
    async_task.processing = False
    processing_time = time_module.perf_counter() - processing_start_time
    print(f'Processing time (total): {processing_time:.2f} seconds')


def build_worker_runtime() -> WorkerRuntime:
    import torch
    import modules.default_pipeline as pipeline
    import modules.inpaint_worker as inpaint_worker
    import modules.flags as flags
    import ldm_patched.modules.model_management as ldm_model_management
    import extras.ip_adapter as ip_adapter
    from extras.censor import default_censor
    from modules.sdxl_styles import fooocus_expansion
    from modules.private_logger import log

    pid = os.getpid()
    print(f'Started worker with PID {pid}')

    return WorkerRuntime(
        pid=pid,
        pipeline=pipeline,
        inpaint_worker=inpaint_worker,
        flags=flags,
        ldm_model_management=ldm_model_management,
        ip_adapter=ip_adapter,
        default_censor=default_censor,
        fooocus_expansion=fooocus_expansion,
        log=log,
        time_module=time,
    )


def print_welcome():
    try:
        import shared
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)


import torch


@torch.no_grad()
@torch.inference_mode()
def execute_handler(runtime: WorkerRuntime, async_task: AsyncTask):
    preparation_start_time = time.perf_counter()
    async_task.processing = True

    state = execute_pipeline(runtime, async_task, DEFAULT_PIPELINE)

    if state.processing_started:
        stop_processing(async_task, state.processing_start_time, time)

    preparation_time = time.perf_counter() - preparation_start_time
    print(f'Total preparation time: {preparation_time:.2f} seconds')


def worker():
    global async_tasks

    import modules.patch

    runtime = build_worker_runtime()
    print_welcome()

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)

            try:
                execute_handler(runtime, task)
                if task.generate_image_grid:
                    wall = build_image_wall_from_results(task.results)
                    if wall is not None:
                        task.results = task.results + [wall]
                task.yields.append(['finish', task.results])
                runtime.pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(['finish', task.results])
            finally:
                if runtime.pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[runtime.pid]


threading.Thread(target=worker, daemon=True).start()
