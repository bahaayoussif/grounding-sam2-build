# mechanisms/segmentation_pipe.py

from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from torchvision.ops import box_convert
import numpy as np
import torch
import torch.nn as nn
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Dict, Any, List

# ------------- Device helpers -------------

def _pick_device() -> str:
    """Prefer CUDA, then Apple Silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _to_device(x, device: str):
    """Move tensor to device; coerce to float on MPS to avoid dtype quirks."""
    if torch.is_tensor(x):
        x = x.to(device)
        if device == "mps" and x.dtype not in (torch.float16, torch.float32, torch.float64):
            x = x.float()
    return x

# ------------- Grounded Segmentation pipeline -------------

def grounded_segmentation(
    gd_model,
    sam_model,
    text_prompt: str,
    point_coords,
    point_labels,
    original_image,
    box_threshold: float,
    text_threshold: float,
    device: Optional[str] = None,
):
    """
    Runs grounding then segmentation (SAM2) end-to-end.
    Returns: image_with_box (PIL), size (W,H), boxes_filt (Tensor cpu), pred_phrases (List[str]), all_masks (List[np.ndarray])
    """
    device = device or _pick_device()
    image_with_box, size, boxes_filt, pred_phrases, pred_dict = ground_image(
        gd_model, text_prompt, original_image, box_threshold, text_threshold, device=device
    )
    all_masks = sam_seg_rects(sam_model, point_coords, point_labels, original_image, pred_dict["boxes"])
    return image_with_box, size, boxes_filt, pred_phrases, all_masks

def ground_image(
    gd_model,
    text_prompt: str,
    image,
    box_threshold: float,
    text_threshold: float,
    token_spans=None,
    device: Optional[str] = None,
):
    """
    Runs GroundingDINO on a PIL.Image (or compatible) and returns visualization and detections.
    boxes_filt is a CPU torch.Tensor of shape (N,4) in normalized cx,cy,w,h.
    """
    device = device or _pick_device()
    image_pil, image_tensor = prepare_image(image)

    boxes_filt, pred_phrases = get_grounding_output(
        gd_model,
        image_tensor,
        text_prompt,
        box_threshold,
        text_threshold,
        token_spans=token_spans,
        device=device,
    )

    size = image_pil.size  # (W,H)
    pred_dict = {
        "boxes": boxes_filt,           # tensor on CPU (cx,cy,w,h normalized)
        "size": [size[1], size[0]],    # store as [H, W]
        "labels": pred_phrases,
    }

    image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)[0]
    return image_with_box, size, boxes_filt, pred_phrases, pred_dict

def sam_seg_rects(sam_model, point_coords, point_labels, image, boxes):
    """
    Segments regions defined by `boxes` (normalized cx,cy,w,h) using SAM2.
    Returns: List of masks (np.ndarray, typically HxW or 1xHxW depending on predictor output).
    """
    image_pil, _ = prepare_image(image)
    cv2_img = set_prediction_target(sam_model, image_pil)

    mask_vis, rects = get_grounding_masks(cv2_img, boxes)
    if not rects:
        return []

    all_masks: List[np.ndarray] = []
    for rect in rects:
        # rect shape: [x0, y0, x1, y1] in image pixels
        rect_np = np.array(rect, dtype=np.float32)[None, :]
        masks, _, _ = sam_model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=rect_np,
            multimask_output=False,
        )
        # Normalize mask output shape to (H,W) numpy
        if torch.is_tensor(masks):
            m = masks.detach().cpu().numpy()
        else:
            m = np.asarray(masks)
        if m.ndim >= 3:
            m = np.squeeze(m)  # handle (1,1,H,W) -> (H,W)
        all_masks.append(m)
    return all_masks

# ------------- Drawing / I/O helpers -------------

def get_font(font_size: int):
    """Try a few common fonts; fall back to PIL default."""
    font_names = ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", "Verdana.ttf", "FreeSans.ttf"]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font=font_name, size=font_size)
        except IOError:
            continue
    return ImageFont.load_default()

def plot_boxes_to_image(image_pil: Image.Image, tgt: Dict[str, Any]):
    """
    Draws boxes (cx,cy,w,h normalized) + labels on image_pil.
    Returns (image_with_boxes, mask_visualization)
    """
    H, W = tgt["size"]  # H, W
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    font = get_font(36)

    # Ensure torch tensor (CPU float)
    if isinstance(boxes, np.ndarray):
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    else:
        boxes_t = boxes.detach().cpu().to(dtype=torch.float32)

    for box, label in zip(boxes_t, labels):
        # box normalized cx,cy,w,h -> pixel xyxy
        box = box * torch.tensor([W, H, W, H], dtype=torch.float32)
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = [int(v.item() if torch.is_tensor(v) else v) for v in box]

        # clamp to image bounds
        x0 = max(0, min(W - 1, x0))
        y0 = max(0, min(H - 1, y0))
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

        # text background box
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            w, h = draw.textsize(str(label), font=font)
            bbox = (x0, y0, x0 + w, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)

        # simple mask viz
        mask_draw.rectangle([x0, y0, x1, y1], fill=255)

    return image_pil, mask

def prepare_image(image) -> Tuple[Image.Image, torch.Tensor]:
    """
    Convert to RGB PIL, then to normalized tensor (C,H,W).
    Uses RandomResize like the original code (for consistency).
    """
    image_pil = image.convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # (3, H, W)
    return image_pil, image_tensor

def load_image(image_path: str):
    """Convenience loader: returns (PIL, Tensor) using prepare_image."""
    image_pil = Image.open(image_path).convert("RGB")
    return prepare_image(image_pil)

# ------------- Model loading -------------

def load_model(model_config_path, model_checkpoint_path, device: Optional[str] = None, cpu_only: bool = False):
    """
    Build and load GroundingDINO model. If cpu_only is True, forces CPU,
    otherwise uses provided device or picks one.
    """
    device = "cpu" if cpu_only else (device or _pick_device())

    args = SLConfig.fromfile(model_config_path)
    args.device = device  # some configs read this, but we still .to(device) below

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    try:
        model = model.to(device)
    except Exception:
        pass
    return model

# ------------- Inference utilities -------------

@torch.no_grad()
def get_grounding_output(
    model,
    image,                     # tensor CHW (from prepare_image)
    caption: str,
    box_threshold: float,
    text_threshold: float = None,
    with_logits: bool = True,
    token_spans=None,
    device: Optional[str] = None,
    cpu_only: bool = False,
):
    """
    Runs GroundingDINO and returns:
      - boxes_filt: torch.Tensor on CPU, shape (N,4) normalized cx,cy,w,h
      - pred_phrases: List[str]
    """
    assert text_threshold is not None or token_spans is not None, \
        "text_threshold and token_spans should not be None at the same time!"

    device = "cpu" if cpu_only else (device or _pick_device())

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    # Move model & image to target device
    try:
        model = model.to(device)
    except Exception:
        pass
    image = _to_device(image, device)

    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]              # (nq, 4), normalized cx,cy,w,h

    if token_spans is None:
        logits_filt = logits.detach().cpu()
        boxes_filt = boxes.detach().cpu()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # (M, 256)
        boxes_filt = boxes_filt[filt_mask]    # (M, 4)

        # phrases
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases: List[str] = []
        for logit, _box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrases.append(f"{pred_phrase}({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        tokenizer = model.tokenizer
        positive_maps = create_positive_map_from_span(
            tokenizer(caption), token_span=token_spans
        ).to(image.device)

        logits_for_phrases = positive_maps @ logits.T  # (n_phrase, nq)
        all_phrases: List[str] = []
        all_boxes: List[torch.Tensor] = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            if with_logits:
                all_phrases.extend([f"{phrase}({str(v.item())[:4]})" for v in logit_phr[filt_mask]])
            else:
                all_phrases.extend([phrase for _ in range(int(filt_mask.sum()))])
        boxes_filt = torch.cat(all_boxes, dim=0).detach().cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases

# ------------- SAM2 loading & helpers -------------

def load_sam_model(model_path: str, device: Optional[str] = None):
    """
    Builds SAM2 predictor on the chosen device.
    Uses model_path as checkpoint if provided.
    """
    device = device or _pick_device()

    if device == "cuda" and torch.cuda.is_available():
        # enable TF32 on Ampere for speed (safe for inference)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = model_path if model_path else "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def set_prediction_target(model, image: Image.Image):
    """
    Set the image into the SAM2 predictor. Returns BGR np.ndarray for any downstream needs.
    """
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    model.set_image(image_bgr)
    return image_bgr

def get_grounding_masks(image: np.ndarray, boxes_cxcywh) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Convert normalized cx,cy,w,h boxes (torch or numpy) to pixel xyxy rects,
    and return a simple visualization mask and list of rects.
    """
    h, w, _ = image.shape

    # Ensure torch tensor for conversion utilities
    if isinstance(boxes_cxcywh, np.ndarray):
        boxes_t = torch.as_tensor(boxes_cxcywh, dtype=torch.float32)
    else:
        boxes_t = boxes_cxcywh.detach().cpu().to(dtype=torch.float32)

    # Denormalize and convert to xyxy
    scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes_unnorm = boxes_t * scale
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    mask = np.zeros_like(image)
    rects: List[List[int]] = []
    for x0, y0, x1, y1 in boxes_xyxy:
        x0i = int(max(0, min(w - 1, x0)))
        y0i = int(max(0, min(h - 1, y0)))
        x1i = int(max(0, min(w - 1, x1)))
        y1i = int(max(0, min(h - 1, y1)))
        if x1i <= x0i or y1i <= y0i:
            continue
        mask[y0i:y1i, x0i:x1i, :] = 255
        rects.append([x0i, y0i, x1i, y1i])

    return mask, rects

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)