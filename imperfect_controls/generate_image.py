"""Single-image ControlNet sampling helpers."""

import sys
from pathlib import Path
from typing import Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from share import *
import config

import einops
import numpy as np
import torch

from cldm.ddim_hacked import DDIMSampler

ControlInput = Union[np.ndarray, torch.Tensor]


def _control_to_bchw(control: ControlInput, device: torch.device) -> torch.Tensor:
    """Return float tensor [1, 3, H, W] on ``device``.

    Accepts numpy RGB ``float`` in ``[0, 1]``, shape ``H×W×3``, or ``torch`` tensor
    ``CHW`` / ``BCHW`` (values same range as training hints).
    """
    if isinstance(control, torch.Tensor):
        t = control.detach().float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.shape[1] != 3:
            raise ValueError("Control tensor must have 3 channels (BCHW or CHW).")
        return t.contiguous().to(device)

    arr = np.asarray(control)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Control ndarray must be H×W×3 RGB (float in [0, 1]).")
    t = torch.from_numpy(arr).unsqueeze(0).to(device)
    return einops.rearrange(t, "b h w c -> b c h w").contiguous().float()


def generate_image(
    model,
    sampler: DDIMSampler,
    control: ControlInput,
    prompt: str,
    *,
    a_prompt: str = "best quality, extremely detailed",
    n_prompt: str = "low quality, blurry, distorted",
    ddim_steps: int = 20,
    guidance_scale: float = 9.0,
    eta: float = 0.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Run DDIM sampling for one control map and caption.

    Parameters
    ----------
    model
        Loaded ``ControlLDM`` on CUDA, ``eval()`` mode.
    sampler
        ``DDIMSampler(model)`` instance (reuse across calls).
    control
        Conditioning image: ``float32`` RGB ``H×W×3`` in ``[0, 1]``, or ``torch`` ``CHW`` / ``BCHW``.
    prompt
        User prompt (before ``a_prompt`` suffix).

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB image ``H×W×3``.
    """
    device = next(model.parameters()).device
    ctrl = _control_to_bchw(control, device)
    _, _, h, w = ctrl.shape

    full_prompt = f"{prompt}, {a_prompt}"

    model.control_scales = [strength] * 13

    cond = {
        "c_concat": [ctrl],
        "c_crossattn": [model.get_learned_conditioning([full_prompt])],
    }
    un_cond = {
        "c_concat": [ctrl],
        "c_crossattn": [model.get_learned_conditioning([n_prompt])],
    }
    shape = (4, h // 8, w // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    with torch.inference_mode():
        samples, _ = sampler.sample(
            ddim_steps,
            1,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x = model.decode_first_stage(samples)

    x = einops.rearrange(x, "b c h w -> b h w c")
    x = (x * 127.5 + 127.5).cpu().numpy()
    x = x.clip(0, 255).astype(np.uint8)[0]
    return x
