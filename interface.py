import gradio as gr
import os
import numpy as np
from PIL import Image
import logging
import torch

from mechanisms.segmentation_pipe import load_model, load_sam_model, ground_image, sam_seg_rects

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="2025-09-15 %(asctime)s - %(levelname)s - %(message)s")

CURRENTLY_POSITIVE = True


def pick_device():
    """Pick a safe device: CUDA if available, else MPS on Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_boxes_dict(pred_dict):
    """Normalize pred_dict into a dict with 'boxes' key (possibly empty)."""
    if not pred_dict or ("boxes" not in pred_dict) or pred_dict["boxes"] is None:
        return {"boxes": np.zeros((0, 4), dtype=float)}
    boxes = pred_dict["boxes"]
    # move to numpy on CPU if it’s a tensor
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    norm = {"boxes": boxes}
    for k, v in pred_dict.items():
        if k == "boxes":
            continue
        # best effort: move tensors to cpu/numpy for downstream / gradio
        if torch.is_tensor(v):
            try:
                v = v.detach().cpu().numpy()
            except Exception:
                v = v.detach().cpu()
        norm[k] = v
    return norm


def get_select_index(evt: gr.SelectData, image, point_label_state):
    global CURRENTLY_POSITIVE
    # Work on a copy to avoid mutating Gradio's reference
    canvas = image.copy()

    star_path = "./red_star.png" if not CURRENTLY_POSITIVE else "./star.png"
    star_image = Image.open(star_path).convert("RGBA").resize((32, 32))

    x, y = evt.index[0], evt.index[1]
    sx, sy = x - star_image.width // 2, y - star_image.height // 2
    canvas.paste(star_image, (sx, sy), mask=star_image)

    point_label_state["points"].append((x, y))
    point_label_state["labels"].append(1 if CURRENTLY_POSITIVE else 0)
    return canvas, point_label_state


def box_segment(image, text_prompt, box_threshold, text_threshold):
    """
    Returns: (box_image, pred_state_dict)
    pred_state_dict will always include 'boxes' (possibly empty).
    """
    if image is None:
        logging.warning("No primary image provided to box_segment.")
        return None, {"boxes": np.zeros((0, 4), dtype=float)}

    device = pick_device()
    logging.info(f"Loading detector on device: {device}")

    config_file = "./gd_configs/grounding_dino_config.py"
    checkpoint_path = "./checkpoints/groundingdino_swint_ogc.pth"

    try:
        model = load_model(config_file, checkpoint_path).eval().to(device)
    except Exception:
        logging.exception("Failed to load GroundingDINO model or move to device.")
        return image, {"boxes": np.zeros((0, 4), dtype=float)}

    image = image.convert("RGB")
    try:
        image_with_box, size, boxes_filt, pred_phrases, pred_dict = ground_image(
            model,
            text_prompt,
            image,
            box_threshold,
            text_threshold,
            device=device,  # <<< pass device
        )
        pred_dict = _ensure_boxes_dict(pred_dict)
        n_boxes = len(pred_dict["boxes"])
        logging.info(f"Grounding produced {n_boxes} boxes.")
        return (image_with_box if image_with_box is not None else image), pred_dict
    except Exception:
        logging.exception("ground_image failed.")
        return image, {"boxes": np.zeros((0, 4), dtype=float)}


def create_mask_and_cutout(image, mask, color=(255, 255, 255)):
    h, w = mask.shape[-2:]
    mask_image = Image.fromarray(np.uint8(mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)))
    mask_gray = Image.fromarray(np.uint8(mask.reshape(h, w) * 255)).convert("L")

    base_np = np.array(image)
    base_np[mask.reshape(h, w) == 0] = 0
    masked_cutout = Image.fromarray(base_np)
    return mask_gray, masked_cutout


def segment(image, pred_dict, point_label_state):
    """
    Returns a list of PIL images for the Gallery.
    If no boxes or segmentation fails, returns [] (empty gallery).
    """
    try:
        if image is None:
            logging.warning("No image provided to segment.")
            return []

        pred_dict = _ensure_boxes_dict(pred_dict if pred_dict is not None else {})
        boxes = pred_dict["boxes"]
        if boxes is None or len(boxes) == 0:
            logging.info("No boxes available for segmentation; returning empty gallery.")
            return []

        # Points / labels (optional)
        points = np.array(point_label_state.get("points", [])) if point_label_state else None
        labels = np.array(point_label_state.get("labels", [])) if point_label_state else None
        if points is None or len(points) == 0:
            points, labels = None, None  # box-only segmentation mode

        predictor = load_sam_model("./checkpoints/sam2_hiera_large.pt")
        logging.info(f"Running SAM with {len(boxes)} boxes and "
                     f"{(len(points) if points is not None else 0)} point(s).")

        all_masks = sam_seg_rects(
            predictor,
            points,
            labels,
            image,
            boxes
        )

        if all_masks is None or (isinstance(all_masks, (list, tuple)) and len(all_masks) == 0):
            logging.info("Segmentation produced no masks; returning empty gallery.")
            return []

        output_cutouts = []
        for mask in all_masks:
            _, cutout = create_mask_and_cutout(image, mask)
            output_cutouts.append(cutout)
        logging.info(f"Segmentation produced {len(output_cutouts)} cutout(s).")
        return output_cutouts

    except Exception:
        logging.exception("Segmentation pipeline failed.")
        return []


def passthrough(image):
    return image, {"points": [], "labels": []}


def toggle_current_pos():
    global CURRENTLY_POSITIVE
    CURRENTLY_POSITIVE = not CURRENTLY_POSITIVE


with gr.Blocks() as demo:
    with gr.Column():
        prompt = gr.Textbox("Prompt")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    box_thresh = gr.Slider(minimum=0.0, value=0.3, maximum=1.0, label="Box Threshold")
                    text_thresh = gr.Slider(minimum=0.0, value=0.3, maximum=1.0, label="Text Threshold")
                ground_button = gr.Button("Ground Image")
            with gr.Column():
                toggle_pos = gr.Button("Toggle Positive")
                segment_button = gr.Button("Segment Image")
    with gr.Row():
        with gr.Column():
            primary_image = gr.Image(type="pil", interactive=True)

        with gr.Column():
            box_image = gr.Image(type="pil", interactive=False)

        with gr.Column():
            selection_image = gr.Image(type="pil", interactive=False)

        with gr.Column():
            final_image = gr.Gallery(type="pil", interactive=False)

    toggle_pos.click(toggle_current_pos)

    point_label_state = gr.State(value={"points": [], "labels": []})
    box_image.change(passthrough, box_image, [selection_image, point_label_state])

    selection_image.select(get_select_index, [selection_image, point_label_state], [selection_image, point_label_state])

    pred_state = gr.State()
    ground_button.click(box_segment, [primary_image, prompt, box_thresh, text_thresh], [box_image, pred_state])

    segment_button.click(segment, [primary_image, pred_state, point_label_state], final_image)

demo.launch()
