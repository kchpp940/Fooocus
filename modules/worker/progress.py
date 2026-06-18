from __future__ import annotations

import os
import modules.config
from extras.censor import default_censor


def progressbar(async_task, value: int, text: str):
    print(f'[Fooocus] {text}')
    async_task.yields.append(['preview', (value, text, None)])


def yield_result(async_task, imgs, progressbar_index, black_out_nsfw, censor=True, do_not_show_finished_images=False):
    if not isinstance(imgs, list):
        imgs = [imgs]

    if censor and (modules.config.default_black_out_nsfw or black_out_nsfw):
        progressbar(async_task, progressbar_index, 'Checking for NSFW content ...')
        imgs = default_censor(imgs)

    async_task.results = async_task.results + imgs

    if do_not_show_finished_images:
        return

    async_task.yields.append(['results', async_task.results])


class ProgressReporter:
    def __init__(self, async_task, total_steps: int, preparation_steps: int, total_count: int):
        self.async_task = async_task
        self.total_steps = total_steps
        self.preparation_steps = preparation_steps
        self.total_count = total_count
        self.current_progress = 0
        self.callback_steps = 0
        self.current_task_id = 0

    def set_current_task(self, task_id: int):
        self.current_task_id = task_id
        self.callback_steps = 0

    def update_progress(self, value: int, text: str):
        print(f'[Fooocus] {text}')
        self.current_progress = value
        self.async_task.yields.append(['preview', (value, text, None)])

    def get_diffusion_callback(self, base_progress: int):
        def callback(step, x0, x, total_steps, y):
            if step == 0:
                self.callback_steps = 0
            self.callback_steps += (100 - self.preparation_steps) / float(self.total_steps)
            progress = int(base_progress + self.callback_steps)
            self.async_task.yields.append(['preview', (
                progress,
                f'Sampling step {step + 1}/{total_steps}, image {self.current_task_id + 1}/{self.total_count} ...', y)])
        return callback

    def yield_result(self, imgs, progress: int, censor: bool = True,
                     do_not_show_finished_images: bool = False):
        if not isinstance(imgs, list):
            imgs = [imgs]

        if censor and (modules.config.default_black_out_nsfw or self.async_task.black_out_nsfw):
            self.update_progress(progress, 'Checking for NSFW content ...')
            imgs = default_censor(imgs)

        self.async_task.results = self.async_task.results + imgs

        if do_not_show_finished_images:
            return

        self.async_task.yields.append(['results', self.async_task.results])

    def compute_progress(self, base: int, done_steps: int) -> int:
        return int(base + (100 - self.preparation_steps) / float(self.total_steps) * done_steps)
