from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import torch

from setup_propainter import PROPAINTER_DIR, propainter_ready
from tracker import JobCancelled

OOM_HINT = (
    "显存不足。请把「处理分辨率」调低后再试，或关闭其他占用 GPU 的程序。"
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_lock = threading.Lock()
_cache: dict | None = None


def _prepare_path() -> None:
    root = str(PROPAINTER_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_models(on_log=None) -> dict:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        if not propainter_ready():
            raise RuntimeError("未安装 ProPainter。请先运行 python backend/setup_propainter.py")
        if on_log:
            on_log("正在加载 ProPainter 模型（首次较慢，之后会复用）…")
        _prepare_path()
        prev = os.getcwd()
        os.chdir(PROPAINTER_DIR)
        try:
            from model.misc import get_device
            from model.modules.flow_comp_raft import RAFT_bi
            from model.propainter import InpaintGenerator
            from model.recurrent_flow_completion import RecurrentFlowCompleteNet
            from utils.download_util import load_file_from_url

            device = get_device()
            weights = PROPAINTER_DIR / "weights"
            url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
            raft_path = load_file_from_url(url=url + "raft-things.pth", model_dir=str(weights), progress=True)
            flow_path = load_file_from_url(
                url=url + "recurrent_flow_completion.pth", model_dir=str(weights), progress=True
            )
            pp_path = load_file_from_url(url=url + "ProPainter.pth", model_dir=str(weights), progress=True)
            fix_raft = RAFT_bi(raft_path, device)
            fix_flow_complete = RecurrentFlowCompleteNet(flow_path)
            for param in fix_flow_complete.parameters():
                param.requires_grad = False
            fix_flow_complete.to(device).eval()
            model = InpaintGenerator(model_path=pp_path).to(device).eval()
            use_half = device.type == "cuda"
            if use_half:
                fix_flow_complete = fix_flow_complete.half()
                model = model.half()
            _cache = {
                "device": device,
                "fix_raft": fix_raft,
                "fix_flow_complete": fix_flow_complete,
                "model": model,
                "use_half": use_half,
            }
        finally:
            os.chdir(prev)
        if on_log:
            on_log("ProPainter 模型已就绪")
        return _cache


def _write_jpeg(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"无法写入 {path}")
    path.write_bytes(buf.tobytes())


def inpaint_folder(
    video_path: str,
    mask_path: str,
    output_dir: str,
    *,
    resize_ratio: float,
    fp16: bool,
    save_fps: int,
    subvideo_length: int,
    on_log,
    cancel_event: Event | None = None,
    raft_iter: int = 20,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    mask_dilation: int = 4,
) -> Path:
    cache = ensure_models(on_log)
    _prepare_path()
    from core.utils import to_tensors
    from inference_propainter import get_ref_index, read_frame_from_videos, read_mask, resize_frames

    device = cache["device"]
    fix_raft = cache["fix_raft"]
    fix_flow_complete = cache["fix_flow_complete"]
    model = cache["model"]
    use_half = bool(fp16) and cache["use_half"]

    try:
        frames, fps, size, video_name = read_frame_from_videos(video_path)
        if resize_ratio != 1.0:
            size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))
        frames, size, out_size = resize_frames(frames, size)
        fps = save_fps if fps is None else fps
        save_root = Path(output_dir) / video_name
        save_root.mkdir(parents=True, exist_ok=True)
        flow_masks, masks_dilated = read_mask(
            mask_path, len(frames), size, flow_mask_dilates=mask_dilation, mask_dilates=mask_dilation
        )
        width, height = size
        frames_inp = [np.array(frame).astype(np.uint8) for frame in frames]
        frames_t = to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks_t = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated_t = to_tensors()(masks_dilated).unsqueeze(0)
        frames_t = frames_t.to(device)
        flow_masks_t = flow_masks_t.to(device)
        masks_dilated_t = masks_dilated_t.to(device)

        video_length = frames_t.size(1)
        on_log(f"Processing: {video_name} [{video_length} frames]...")
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled(None, "propainter")

        with torch.no_grad():
            if frames_t.size(-1) <= 640:
                short_clip_len = 12
            elif frames_t.size(-1) <= 720:
                short_clip_len = 8
            elif frames_t.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            if frames_t.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for index in range(0, video_length, short_clip_len):
                    if cancel_event is not None and cancel_event.is_set():
                        raise JobCancelled(None, "propainter")
                    end_f = min(video_length, index + short_clip_len)
                    if index == 0:
                        flows_f, flows_b = fix_raft(frames_t[:, index:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames_t[:, index - 1 : end_f], iters=raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
            else:
                gt_flows_bi = fix_raft(frames_t, iters=raft_iter)
                torch.cuda.empty_cache()

            if use_half:
                frames_t = frames_t.half()
                flow_masks_t = flow_masks_t.half()
                masks_dilated_t = masks_dilated_t.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for index in range(0, flow_length, subvideo_length):
                    s_f = max(0, index - pad_len)
                    e_f = min(flow_length, index + subvideo_length + pad_len)
                    pad_len_s = max(0, index) - s_f
                    pad_len_e = e_f - min(flow_length, index + subvideo_length)
                    pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks_t[:, s_f : e_f + 1],
                    )
                    pred_flows_bi_sub = fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        flow_masks_t[:, s_f : e_f + 1],
                    )
                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()
                pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
            else:
                pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
                pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
                torch.cuda.empty_cache()

            masked_frames = frames_t * (1 - masks_dilated_t)
            subvideo_length_img_prop = min(100, subvideo_length)
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for index in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, index - pad_len)
                    e_f = min(video_length, index + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, index) - s_f
                    pad_len_e = e_f - min(video_length, index + subvideo_length_img_prop)
                    batch, length, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f : e_f - 1], pred_flows_bi[1][:, s_f : e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                        masked_frames[:, s_f:e_f],
                        pred_flows_bi_sub,
                        masks_dilated_t[:, s_f:e_f],
                        "nearest",
                    )
                    updated_frames_sub = frames_t[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f]) + prop_imgs_sub.view(
                        batch, length, 3, height, width
                    ) * masks_dilated_t[:, s_f:e_f]
                    updated_frames.append(updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                    updated_masks.append(
                        updated_local_masks_sub.view(batch, length, 1, height, width)[
                            :, pad_len_s : e_f - s_f - pad_len_e
                        ]
                    )
                    torch.cuda.empty_cache()
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                batch, length, _, _, _ = masks_dilated_t.size()
                prop_imgs, updated_local_masks = model.img_propagation(
                    masked_frames, pred_flows_bi, masks_dilated_t, "nearest"
                )
                updated_frames = frames_t * (1 - masks_dilated_t) + prop_imgs.view(
                    batch, length, 3, height, width
                ) * masks_dilated_t
                updated_masks = updated_local_masks.view(batch, length, 1, height, width)
                torch.cuda.empty_cache()

        comp_frames: list = [None] * video_length
        neighbor_stride = neighbor_length // 2
        ref_num = subvideo_length // ref_stride if video_length > subvideo_length else -1
        steps = max(1, (video_length + neighbor_stride - 1) // neighbor_stride)
        for step, index in enumerate(range(0, video_length, neighbor_stride)):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled(None, "propainter")
            on_log(f"PROPAINTER_PCT {int(step / steps * 100)}")
            neighbor_ids = list(
                range(max(0, index - neighbor_stride), min(video_length, index + neighbor_stride + 1))
            )
            ref_ids = get_ref_index(index, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )
            with torch.no_grad():
                local_t = len(neighbor_ids)
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, local_t)
                pred_img = pred_img.view(-1, 3, height, width)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = (
                    masks_dilated_t[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                )
                for local_i, frame_idx in enumerate(neighbor_ids):
                    img = np.array(pred_img[local_i]).astype(np.uint8) * binary_masks[local_i] + frames_inp[
                        frame_idx
                    ] * (1 - binary_masks[local_i])
                    if comp_frames[frame_idx] is None:
                        comp_frames[frame_idx] = img
                    else:
                        comp_frames[frame_idx] = (
                            comp_frames[frame_idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        )
                    comp_frames[frame_idx] = comp_frames[frame_idx].astype(np.uint8)
            torch.cuda.empty_cache()

        for idx, frame in enumerate(comp_frames):
            if frame is None:
                continue
            bgr = cv2.cvtColor(cv2.resize(frame, out_size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
            _write_jpeg(save_root / "frames" / f"{idx + 1:06d}.jpg", bgr)
        on_log(f"saved in {save_root}")
        return save_root
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(OOM_HINT) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(OOM_HINT) from exc
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
