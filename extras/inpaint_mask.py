import sys

import modules.config
import numpy as np
import torch
from extras.GroundingDINO.util.inference import default_groundingdino
from extras.sam.predictor import SamPredictor
from rembg import remove, new_session
from segment_anything import sam_model_registry
from segment_anything.utils.amg import remove_small_regions


def normalize_mask_2d(mask) -> np.ndarray | None:
    """
    归一化任意形式的 mask 为 2D uint8 单通道数组 [0, 255]。
    支持: None -> None; 2D (H,W); 3D (H,W,1); 3D (H,W,3) 通过亮度通道取阈值
    """
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        return None
    m = mask.copy()
    if m.ndim == 3:
        if m.shape[2] == 1:
            m = m[:, :, 0]
        elif m.shape[2] >= 3:
            m = np.mean(m[:, :, :3], axis=2)
        else:
            return None
    if m.ndim != 2:
        return None
    if m.dtype != np.uint8:
        if m.max() <= 1.0 and m.min() >= 0.0:
            m = (m * 255.0).astype(np.uint8)
        else:
            m = np.clip(m, 0, 255).astype(np.uint8)
    return m


def is_mask_valid(mask) -> bool:
    """判断 mask 是否存在且有有效像素（非全黑）"""
    m = normalize_mask_2d(mask)
    if m is None:
        return False
    return bool(np.any(m > 127))


def process_mask(mask, invert: bool = False, morphic_px: int = 0,
                 target_size=None) -> np.ndarray | None:
    """
    inpaint / enhance 共享的蒙版后处理入口。

    执行顺序（与调用方一致）：
        1. normalize_mask_2d: 统一为 2D uint8 [0,255]，失败返回 None
        2. 可选 resize 到 target_size (width, height)
        3. invert: 反选（255 - mask）
        4. morphic_px: 形态学（正=膨胀，负=腐蚀，0=跳过）
        5. 有效性检查：全黑返回 None

    参数:
        mask: 原始蒙版（None / 2D / 3D / float / uint8）
        invert: 是否反选
        morphic_px: 膨胀/腐蚀像素数（正膨胀，负腐蚀）
        target_size: (width, height) 元组，需要 resize 时传入

    返回:
        处理后的 2D uint8 mask，或 None（无效/空蒙版）
    """
    from modules.util import erode_or_dilate, resample_image

    m = normalize_mask_2d(mask)
    if m is None:
        return None

    if target_size is not None:
        tw, th = target_size
        if m.shape[0] != th or m.shape[1] != tw:
            m = resample_image(m, width=tw, height=th)
            m = normalize_mask_2d(m)
            if m is None:
                return None

    if invert:
        m = 255 - m

    morphic_px_int = int(morphic_px)
    if morphic_px_int != 0:
        m = erode_or_dilate(m, morphic_px_int)
        m = normalize_mask_2d(m)
        if m is None:
            return None

    if not is_mask_valid(m):
        return None

    return m


def merge_masks(mask_sources: list, target_size=None) -> np.ndarray | None:
    """
    合并多个蒙版来源，只处理真正的蒙版通道，不合并 RGB 图像内容。

    每个来源可以是:
      - None: 跳过
      - np.ndarray: 原始蒙版数据（会经过 normalize_mask_2d 归一化）
      - dict: {'mask': np.ndarray} 格式，只提取 'mask' 字段，忽略 'image'

    合并逻辑:
      1. 逐个归一化每个来源为 2D uint8
      2. 可选 resize 到 target_size=(width, height)
      3. 逐像素取 np.maximum 合并（并集）

    参数:
        mask_sources: 蒙版来源列表，按优先级从低到高排列（后加入的会覆盖前面的）
        target_size: (width, height) 元组，需要 resize 时传入

    返回:
        合并后的 2D uint8 mask，或 None（所有来源都无效）
    """
    from modules.util import resample_image

    result = None
    for src in mask_sources:
        if src is None:
            continue

        if isinstance(src, dict):
            src = src.get('mask')
            if src is None:
                continue

        if not isinstance(src, np.ndarray):
            continue

        m = normalize_mask_2d(src)
        if m is None:
            continue

        if target_size is not None:
            tw, th = target_size
            if m.shape[0] != th or m.shape[1] != tw:
                m = resample_image(m, width=tw, height=th)
                m = normalize_mask_2d(m)
                if m is None:
                    continue

        if result is None:
            result = m
        else:
            if result.shape != m.shape:
                if target_size is not None:
                    tw, th = target_size
                    m = resample_image(m, width=tw, height=th)
                    result = resample_image(result, width=tw, height=th)
                    m = normalize_mask_2d(m)
                    result = normalize_mask_2d(result)
                    if m is None or result is None:
                        continue
                else:
                    continue
            result = np.maximum(result, m)

    if result is None or not is_mask_valid(result):
        return None

    return result


class SAMOptions:
    def __init__(self,
                 # GroundingDINO
                 dino_prompt: str = '',
                 dino_box_threshold=0.3,
                 dino_text_threshold=0.25,
                 dino_erode_or_dilate=0,
                 dino_debug=False,

                 # SAM
                 max_detections=2,
                 model_type='vit_b'
                 ):
        self.dino_prompt = dino_prompt
        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold
        self.dino_erode_or_dilate = dino_erode_or_dilate
        self.dino_debug = dino_debug
        self.max_detections = max_detections
        self.model_type = model_type


def optimize_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    removes small disconnected regions and holes
    """
    fine_masks = []
    for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
        fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
    masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
    return torch.from_numpy(masks)


def generate_mask_from_image(image: np.ndarray, mask_model: str = 'sam', extras=None,
                             sam_options: SAMOptions | None = SAMOptions) -> tuple[np.ndarray | None, int | None, int | None, int | None]:
    """
    统一返回契约:
      mask: np.ndarray | None - 2D uint8 (H,W) 单通道 [0,255]，无有效检测返回 None
      dino_detection_count: int - DINO 检测框数
      sam_detection_count: int - SAM 分割出的掩码数
      sam_detection_on_mask_count: int - 实际合并到最终 mask 的掩码数

    调用方需要自行处理: 反选、膨胀/腐蚀的执行顺序（建议先反选，再形态学）。
    """
    dino_detection_count = 0
    sam_detection_count = 0
    sam_detection_on_mask_count = 0

    if image is None:
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    if extras is None:
        extras = {}

    if isinstance(image, dict) and 'image' in image:
        image = image['image']

    if mask_model != 'sam' or sam_options is None:
        try:
            result = remove(
                image,
                session=new_session(mask_model, **extras),
                only_mask=True,
                **extras
            )
        except Exception as e:
            print(f'[Mask] {mask_model} failed: {e}')
            return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

        result_2d = normalize_mask_2d(result)
        if not is_mask_valid(result_2d):
            return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count
        return result_2d, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    try:
        detections, boxes, logits, phrases = default_groundingdino(
            image=image,
            caption=sam_options.dino_prompt,
            box_threshold=sam_options.dino_box_threshold,
            text_threshold=sam_options.dino_text_threshold
        )
    except Exception as e:
        print(f'[Mask] GroundingDINO failed: {e}')
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    H, W = image.shape[0], image.shape[1]
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

    dino_detection_count = boxes.size(0)

    if dino_detection_count == 0:
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    try:
        sam_checkpoint = modules.config.download_sam_model(sam_options.model_type)
        sam = sam_model_registry[sam_options.model_type](checkpoint=sam_checkpoint)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f'[Mask] SAM load failed: {e}')
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    final_mask_tensor = torch.zeros((image.shape[0], image.shape[1]))

    sam_predictor.set_image(image)

    if sam_options.dino_erode_or_dilate != 0:
        for index in range(boxes.size(0)):
            assert boxes.size(1) == 4
            boxes[index][0] -= sam_options.dino_erode_or_dilate
            boxes[index][1] -= sam_options.dino_erode_or_dilate
            boxes[index][2] += sam_options.dino_erode_or_dilate
            boxes[index][3] += sam_options.dino_erode_or_dilate

    if sam_options.dino_debug:
        from PIL import ImageDraw, Image
        debug_dino_image = Image.new("RGB", (image.shape[1], image.shape[0]), color="black")
        draw = ImageDraw.Draw(debug_dino_image)
        for box in boxes.numpy():
            draw.rectangle(box.tolist(), fill="white")
        debug_2d = normalize_mask_2d(np.array(debug_dino_image))
        return debug_2d, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    try:
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = optimize_masks(masks)
        sam_detection_count = len(masks)
        if sam_options.max_detections == 0:
            sam_options.max_detections = sys.maxsize
        sam_objects = min(len(logits), sam_options.max_detections)
        for obj_ind in range(sam_objects):
            mask_tensor = masks[obj_ind][0]
            final_mask_tensor += mask_tensor
            sam_detection_on_mask_count += 1
    except Exception as e:
        print(f'[Mask] SAM predict failed: {e}')
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    if sam_detection_on_mask_count == 0:
        return None, dino_detection_count, sam_detection_count, sam_detection_on_mask_count

    final_mask_np = (final_mask_tensor > 0).to('cpu').numpy().astype(np.uint8) * 255
    return final_mask_np, dino_detection_count, sam_detection_count, sam_detection_on_mask_count
