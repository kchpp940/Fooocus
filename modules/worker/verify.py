import importlib
import sys
import traceback
from typing import List, Dict, Tuple

MODULE_ORDER = [
    'modules.worker.context',
    'modules.worker.progress',
    'modules.worker.result_saver',
    'modules.worker.tasks',
    'modules.worker.stages',
    'modules.worker.enhance',
    'modules.worker.__init__',
    'modules.async_worker',
]

EXPECTED_EXPORTS = {
    'modules.worker.context': [
        'GenerationContext', 'SingleTaskContext', 'EnhanceCtrl', 'WorkerRuntime',
    ],
    'modules.worker.progress': [
        'ProgressReporter', 'progressbar', 'yield_result',
    ],
    'modules.worker.result_saver': [
        'build_metadata_list', 'setup_metadata_parser',
        'save_images_with_metadata', 'build_image_wall_from_results',
    ],
    'modules.worker.tasks': [
        'build_generation_context', 'parse_dimensions_from_aspect_ratio',
        'normalize_performance_settings', 'apply_patch_settings_to_context',
        'apply_overrides_to_context', 'expand_tasks_from_prompt',
        'process_image_input', 'patch_samplers_from_context',
        'apply_freeu_from_context', 'compute_total_steps', 'execute_diffusion_task',
    ],
    'modules.worker.stages': [
        'apply_vary_to_context', 'apply_upscale_to_context',
        'apply_inpaint_to_context', 'apply_control_nets_from_context',
        'generate_enhance_mask', 'prepare_enhance_prompt',
    ],
    'modules.worker.enhance': [
        'process_enhance', 'enhance_upscale', 'run_enhance_pipeline',
    ],
    'modules.worker': [
        'GenerationContext', 'SingleTaskContext', 'EnhanceCtrl', 'WorkerRuntime',
        'ProgressReporter', 'progressbar', 'yield_result',
        'build_metadata_list', 'setup_metadata_parser',
        'save_images_with_metadata', 'build_image_wall_from_results',
        'build_generation_context', 'parse_dimensions_from_aspect_ratio',
        'normalize_performance_settings', 'apply_patch_settings_to_context',
        'apply_overrides_to_context', 'expand_tasks_from_prompt',
        'process_image_input', 'patch_samplers_from_context',
        'apply_freeu_from_context', 'compute_total_steps', 'execute_diffusion_task',
        'apply_vary_to_context', 'apply_upscale_to_context',
        'apply_inpaint_to_context', 'apply_control_nets_from_context',
        'generate_enhance_mask', 'prepare_enhance_prompt',
        'process_enhance', 'enhance_upscale', 'run_enhance_pipeline',
        'check_module_integrity',
    ],
    'modules.async_worker': [
        'AsyncTask', 'async_tasks', 'EarlyReturnException',
        'stop_processing', 'build_worker_runtime', 'execute_handler', 'worker',
    ],
}


def check_module_integrity(verbose: bool = True) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    problems: List[str] = []
    loaded: Dict[str, List[str]] = {}

    for mod_name in MODULE_ORDER:
        display_name = mod_name if not mod_name.endswith('.__init__') else mod_name[:-9]
        try:
            module = importlib.import_module(mod_name)
        except Exception as e:
            problems.append(f'IMPORT FAILURE in {display_name}: {type(e).__name__}: {e}')
            if verbose:
                traceback.print_exc()
            continue

        missing = []
        export_list = EXPECTED_EXPORTS.get(display_name, [])
        for symbol in export_list:
            if not hasattr(module, symbol):
                missing.append(symbol)
        if missing:
            problems.append(f'{display_name} missing exports: {", ".join(missing)}')

        loaded[display_name] = [
            n for n in dir(module) if not n.startswith('_')
        ]

        if verbose:
            status = 'OK' if not missing else f'MISSING {len(missing)}'
            print(f'  [ {status:>14} ] {display_name} ({len(loaded[display_name])} public symbols)')

    return len(problems) == 0, problems, loaded


if __name__ == '__main__':
    print('Fooocus Worker Module Integrity Check')
    print('=' * 60)
    ok, problems, loaded = check_module_integrity(verbose=True)
    print('=' * 60)
    if ok:
        print('All checks passed.')
    else:
        print(f'FAILED with {len(problems)} problem(s):')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
