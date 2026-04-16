from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton


COMPILE_MODES: dict[str, str | None] = {
    "default": "default",
    "reduce-overhead": "reduce-overhead",
    "max-autotune": "max-autotune",
    "max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
}

ALL_MODES: dict[str, str | None] = {"eager": None, **COMPILE_MODES}


@dataclass
class WorkloadSpec:
    name: str
    domain: str
    tags: tuple[str, ...]
    build: Callable[[], tuple[Any, tuple[Any, ...], dict[str, Any]]]


# ---------------------------------------------------------------------------
# Rotary position embedding helpers
# ---------------------------------------------------------------------------


def _rope_cos_sin(
    seq_len: int, head_dim: int, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cosine and sine rotation matrices for Rotary Position Embeddings (RoPE).

    Uses the standard 10000-base frequency schedule.  The output tensors are
    pre-cast to ``dtype`` so they can be broadcast directly against query/key
    tensors without an extra cast inside the hot loop.

    Args:
        seq_len: Number of sequence positions to pre-compute.
        head_dim: Attention head dimension (must be even).
        device: Target device string (e.g. ``"cuda"``).
        dtype: Output dtype for the returned cos/sin tensors.

    Returns:
        Tuple of ``(cos, sin)`` tensors, each of shape ``[seq_len, head_dim]``.
    """
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


# ---------------------------------------------------------------------------
# LLM kernel functions
# ---------------------------------------------------------------------------


def rmsnorm_fn(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply Root Mean Square Layer Normalization.

    Normalises by RMS instead of mean+variance, omitting the bias term.
    Used in Llama, Qwen, Mistral, and most modern LLMs.

    Args:
        x: Input tensor of shape ``[..., hidden_size]``.
        w: Per-channel scale weights of shape ``[hidden_size]``.
        eps: Small constant added for numerical stability.

    Returns:
        Normalised tensor of the same shape as ``x``.
    """
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * w


def layernorm_fn(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Apply standard Layer Normalization with learnable affine parameters.

    Delegates to ``F.layer_norm`` over the last dimension.  Used in
    BERT-style encoders, GPT-2, and ViT models.

    Args:
        x: Input tensor of shape ``[..., hidden_size]``.
        w: Per-channel scale weights of shape ``[hidden_size]``.
        b: Per-channel bias of shape ``[hidden_size]``.
        eps: Small constant for numerical stability.

    Returns:
        Normalised tensor of the same shape as ``x``.
    """
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def swiglu_mlp_fn(
    x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor
) -> torch.Tensor:
    """SwiGLU MLP block: gated activation followed by a down projection.

    Computes ``(silu(x @ Wg) * (x @ Wu)) @ Wd``.  The fused gate-and-up
    projection is the dominant cost; this workload targets kernel fusion
    of the two GEMMs with the elementwise SiLU and multiply.  Used in
    Llama, Mistral, Qwen, and DeepSeek.

    Args:
        x: Input of shape ``[tokens, hidden_size]``.
        wg: Gate projection weight ``[hidden_size, intermediate_size]``.
        wu: Up projection weight ``[hidden_size, intermediate_size]``.
        wd: Down projection weight ``[intermediate_size, hidden_size]``.

    Returns:
        Output tensor of shape ``[tokens, hidden_size]``.
    """
    gate = F.silu(x @ wg)
    up = x @ wu
    return (gate * up) @ wd


def gelu_mlp_fn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Two-layer MLP with tanh-approximate GELU activation.

    Computes ``gelu(x @ W1) @ W2``.  Used in GPT-2, BERT, and ViT models.
    The tanh approximation matches the implementation used in most
    production frameworks.

    Args:
        x: Input of shape ``[tokens, hidden_size]``.
        w1: Up-projection weight ``[hidden_size, intermediate_size]``.
        w2: Down-projection weight ``[intermediate_size, hidden_size]``.

    Returns:
        Output tensor of shape ``[tokens, hidden_size]``.
    """
    return F.gelu(x @ w1, approximate="tanh") @ w2


def rope_apply_fn(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to a query or key tensor.

    Rotates even/odd dimension pairs using pre-computed cos/sin tables.
    The cos/sin tensors are broadcast over the batch and head dimensions.

    Args:
        x: Query or key tensor of shape ``[batch, seq_len, n_heads, head_dim]``.
        cos: Cosine table of shape ``[seq_len, head_dim]``.
        sin: Sine table of shape ``[seq_len, head_dim]``.

    Returns:
        Rotated tensor of the same shape as ``x``.
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos[None, :, None, ::2] - x_odd * sin[None, :, None, ::2]
    rot_odd = x_even * sin[None, :, None, ::2] + x_odd * cos[None, :, None, ::2]
    out = torch.empty_like(x)
    out[..., ::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


def logits_softmax_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Project hidden states to vocabulary logits and compute softmax probabilities.

    Combines the language-model head (GEMM over the full vocabulary) with
    a softmax, representing the peak memory-bandwidth and compute cost per
    forward pass.  In practice the softmax is fused into cross-entropy
    during training, but this workload isolates the combined projection cost.

    Args:
        x: Hidden states of shape ``[tokens, hidden_size]``.
        w: Vocabulary projection weight ``[hidden_size, vocab_size]``.

    Returns:
        Probability tensor of shape ``[tokens, vocab_size]``.
    """
    return torch.softmax(x @ w, dim=-1)


def qk_norm_fn(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalise query and key tensors along the head dimension (QK-Norm).

    Stabilises attention logit magnitudes for long contexts and large head
    dimensions.  Used in Gemma 2, Chameleon, and other models with deep
    attention stacks.

    Args:
        q: Query tensor of shape ``[batch, n_heads, seq_len, head_dim]``.
        k: Key tensor of shape ``[batch, n_heads, seq_len, head_dim]``.

    Returns:
        Tuple of ``(q_normed, k_normed)``, each the same shape as the input.
    """
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    k_norm = torch.norm(k, dim=-1, keepdim=True)
    return q / q_norm, k / k_norm


def topk_sampling_prep_fn(logits: torch.Tensor) -> torch.Tensor:
    """Top-k sampling: compute softmax, retain top-k candidates, renormalise, then select.

    Models the token-sampling pipeline that follows the LM head during
    autoregressive decode.  Only meaningfully represents decode-phase
    compute (a single new-token position); in the prefill preset it runs
    over all ``b*s`` positions, which overstates realistic cost.

    Args:
        logits: Raw vocabulary logits of shape ``[tokens, vocab_size]``.

    Returns:
        Selected token indices of shape ``[tokens, 1]``.
    """
    probs = torch.softmax(logits, dim=-1)
    top_vals, top_idx = torch.topk(probs, k=32, dim=-1)
    norm = top_vals / top_vals.sum(dim=-1, keepdim=True)
    return torch.gather(top_idx, -1, torch.argmax(norm, dim=-1, keepdim=True))


def moe_router_fn(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixture-of-Experts (MoE) router: project to expert logits and return top-2 dispatch weights.

    Computes expert-selection probabilities via a linear projection and
    softmax, then picks the two highest-scoring experts per token.  Mirrors
    the routing step in Mixtral, DeepSeek-MoE, and Qwen-MoE.

    Args:
        x: Token hidden states of shape ``[tokens, hidden_size]``.
        w: Router weight matrix ``[hidden_size, num_experts]``.

    Returns:
        Tuple of ``(topk_weights, topk_indices)``, each of shape
        ``[tokens, 2]``.
    """
    logits = x @ w
    probs = torch.softmax(logits, dim=-1)
    topk_val, topk_idx = torch.topk(probs, k=2, dim=-1)
    return topk_val, topk_idx


# ---------------------------------------------------------------------------
# VLM kernel functions
# ---------------------------------------------------------------------------


def patch_embed_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """ViT patch embedding: stride-14 conv2d followed by spatial flatten and transpose.

    Splits an image into non-overlapping 14x14 patches and projects each
    patch to an embedding vector.  The spatial dims are then flattened into
    the sequence dimension, producing the token sequence fed to the
    Transformer encoder.  Used in CLIP, SigLIP, InternViT, and similar
    vision encoders.

    Args:
        x: Image batch of shape ``[batch, 3, height, width]``.
        w: Patch embedding weight ``[embed_dim, 3, 14, 14]``.

    Returns:
        Patch token tensor of shape ``[batch, num_patches, embed_dim]``.
    """
    y = F.conv2d(x, w, stride=14)
    b, c, h, w_ = y.shape
    return y.flatten(2).transpose(1, 2).reshape(b, h * w_, c)


def vit_mlp_block_fn(
    x: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """ViT MLP block: pre-LayerNorm -> GELU MLP -> residual addition.

    One complete Feed-Forward Network (FFN) sub-block of a Vision
    Transformer encoder layer.  Benchmarks the fusion opportunity across
    LayerNorm, two GEMMs, GELU activation, and the residual add.

    Args:
        x: Input patch tokens of shape ``[batch, num_patches, embed_dim]``.
        ln_w: LayerNorm scale of shape ``[embed_dim]``.
        ln_b: LayerNorm bias of shape ``[embed_dim]``.
        w1: First linear weight ``[embed_dim, mlp_dim]``.
        w2: Second linear weight ``[mlp_dim, embed_dim]``.

    Returns:
        Output tensor of the same shape as ``x``.
    """
    h = F.layer_norm(x, (x.shape[-1],), ln_w, ln_b)
    y = F.gelu(h @ w1, approximate="tanh") @ w2
    return x + y


def vision_rmsnorm_pool_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMSNorm followed by mean pooling over the sequence (token) dimension.

    Produces a single fixed-size vision embedding from a variable-length
    sequence of patch tokens.  This pattern is used in VLM visual projectors
    to compress high-resolution image tokens into a compact representation
    before feeding them to the language model.

    Args:
        x: Patch token tensor of shape ``[batch, num_tokens, embed_dim]``.
        w: RMSNorm scale weights of shape ``[embed_dim]``.

    Returns:
        Pooled embedding of shape ``[batch, embed_dim]``.
    """
    return rmsnorm_fn(x, w).mean(dim=1)


# ---------------------------------------------------------------------------
# Diffusion kernel functions
# ---------------------------------------------------------------------------


def unet_resblock_fn(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, skip: torch.Tensor
) -> torch.Tensor:
    """U-Net residual block: GroupNorm -> SiLU -> Conv3x3 -> Conv3x3 -> skip add.

    A standard ResNet-style block as used in Stable Diffusion's U-Net
    backbone.  Benchmarks the interplay between spatially-local convolutions
    and channel-normalisation under different compile modes.

    Args:
        x: Feature map of shape ``[batch, channels, height, width]``.
        w1: First conv weight ``[channels, channels, 3, 3]``.
        w2: Second conv weight ``[channels, channels, 3, 3]``.
        skip: Residual (skip connection) tensor of the same shape as ``x``.

    Returns:
        Output feature map of the same shape as ``x``.
    """
    y = F.group_norm(x, num_groups=32)
    y = F.silu(F.conv2d(y, w1, padding=1))
    y = F.conv2d(y, w2, padding=1)
    return y + skip


def vae_upsample_fn(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """VAE decoder upsample block: nearest-neighbour 2x -> SiLU -> Conv3x3 -> Conv3x3.

    Doubles spatial resolution in the VAE decoder path of latent-diffusion
    models (Stable Diffusion, SDXL).  The nearest-neighbour interpolation
    avoids checkerboard artefacts from transposed convolutions.

    Args:
        x: Feature map of shape ``[batch, in_channels, height, width]``.
        w1: First conv weight ``[in_channels, in_channels, 3, 3]``.
        w2: Second conv weight ``[out_channels, in_channels, 3, 3]``.

    Returns:
        Upsampled feature map of shape ``[batch, out_channels, height*2, width*2]``.
    """
    y = F.interpolate(x, scale_factor=2.0, mode="nearest")
    y = F.silu(F.conv2d(y, w1, padding=1))
    return F.conv2d(y, w2, padding=1)


def time_embed_mlp_fn(
    t: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """Sinusoidal timestep embedding MLP used in diffusion models.

    Converts scalar diffusion timesteps into a high-dimensional embedding
    via sinusoidal encoding (as in DDPM / DiT), then projects through a
    two-layer SiLU MLP.  The resulting conditioning vector is typically
    added to residual block feature maps.

    Args:
        t: Timestep scalars of shape ``[batch_size]``, dtype ``float32``.
        w1: First projection weight ``[freq_dim, mlp_dim]``.
        w2: Second projection weight ``[mlp_dim, mlp_dim]``.

    Returns:
        Timestep embedding of shape ``[batch_size, mlp_dim]``.
    """
    half = w1.shape[0] // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    emb = t[:, None] * freqs[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1).to(w1.dtype)
    return F.silu(emb @ w1) @ w2


def cfg_blend_fn(uncond: torch.Tensor, cond: torch.Tensor, scale: float) -> torch.Tensor:
    """Classifier-Free Guidance (CFG) linear blend.

    Computes ``uncond + scale * (cond - uncond)``, combining unconditional
    and conditional noise predictions at each diffusion step.  This is a
    pure elementwise operation on full latent tensors - memory-bandwidth
    bound and sensitive to CUDA-graph capture overhead.

    Args:
        uncond: Unconditional model output ``[batch, channels, h, w]``.
        cond: Conditional model output, same shape as ``uncond``.
        scale: CFG guidance scale (commonly 7.5 for Stable Diffusion).

    Returns:
        Guided noise prediction of the same shape as the inputs.
    """
    return uncond + scale * (cond - uncond)


def diffusion_scheduler_step_fn(
    latents: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Single DDPM / DDIM scheduler denoising step.

    Subtracts the scaled predicted noise from the current latent estimate:
    ``latents - sigma * noise``.  The sigma is broadcast from a per-batch
    scalar to match the spatial latent dimensions.  Represents one step of
    the iterative reverse-diffusion process.

    Args:
        latents: Current noisy latent tensor ``[batch, channels, h, w]``.
        noise: Predicted noise tensor, same shape as ``latents``.
        sigma: Per-sample noise scale of shape ``[batch]``.

    Returns:
        Denoised latent estimate of the same shape as ``latents``.
    """
    return latents - sigma[:, None, None, None] * noise


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def label_smoothed_ce_fn(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 0.1
) -> torch.Tensor:
    """Label-smoothed cross-entropy loss.

    Mixes the hard-target NLL with a uniform smoothing term to prevent
    overconfident predictions.  Implemented manually (rather than via
    ``F.cross_entropy(label_smoothing=...)``), which makes the gather and
    mean operations visible to the compiler for fusion analysis.

    Args:
        logits: Raw vocabulary logits of shape ``[tokens, vocab_size]``.
        targets: Ground-truth token indices of shape ``[tokens]``.
        eps: Smoothing coefficient (fraction of probability mass redistributed
            uniformly across the vocabulary).

    Returns:
        Scalar mean label-smoothed loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    return ((1 - eps) * nll + eps * smooth).mean()


def kl_div_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL divergence for knowledge distillation.

    Computes ``KL(teacher || student)`` in the soft-label sense.  Logits are
    divided by ``temperature`` before softmax to produce softer probability
    distributions; the loss is scaled by ``temperature^2`` to keep the
    gradient magnitude consistent across temperature settings.  Used in
    DistilBERT, MiniLM, and RLHF reward-model distillation.

    Args:
        student_logits: Student model logits of shape ``[tokens, vocab_size]``.
        teacher_logits: Teacher model logits, same shape as ``student_logits``.
        temperature: Softmax temperature (higher produces softer distributions).

    Returns:
        Scalar KL divergence loss (batchmean reduction, temperature-rescaled).
    """
    p = F.softmax(teacher_logits / temperature, dim=-1)
    log_q = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean") * (temperature**2)


def _ppo_clip(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def ppo_clipped_loss_fn(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate policy-gradient objective.

    Computes the pessimistic clipped ratio loss used in Proximal Policy
    Optimization (PPO).  The clip prevents large policy updates when the
    probability ratio ``pi/pi_old`` deviates too far from 1.  Used in
    InstructGPT, Claude's RLHF pipeline, and most RLHF reward-model
    fine-tuning frameworks.

    Args:
        log_probs: Log-probabilities under the current policy, shape ``[tokens]``.
        old_log_probs: Log-probabilities under the reference (old) policy,
            same shape as ``log_probs``.
        advantages: Advantage estimates per token, same shape.
        clip_eps: Clipping threshold for the probability ratio.

    Returns:
        Scalar PPO policy loss (negated, since we minimise).
    """
    return _ppo_clip(log_probs, old_log_probs, advantages, clip_eps)


def dpo_loss_fn(
    chosen_lp: torch.Tensor,
    rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization (DPO) loss.

    Optimises a language model to prefer chosen completions over rejected
    ones without a separate reward model.  Each input is a per-sequence
    log-probability (sum over tokens), and the loss is derived from the
    Bradley-Terry preference model.  Used in Zephyr, OpenHermes, and many
    instruction-following fine-tunes.

    Args:
        chosen_lp: Log-probability of chosen completions under the policy,
            shape ``[batch]``.
        rejected_lp: Log-probability of rejected completions under the policy,
            same shape.
        ref_chosen_lp: Log-probability of chosen completions under the frozen
            reference model, same shape.
        ref_rejected_lp: Log-probability of rejected completions under the
            reference model, same shape.
        beta: KL regularisation coefficient (higher keeps policy closer to
            the reference).

    Returns:
        Scalar DPO loss.
    """
    chosen_reward = beta * (chosen_lp - ref_chosen_lp)
    rejected_reward = beta * (rejected_lp - ref_rejected_lp)
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def grpo_loss_fn(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int = 8,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Group Relative Policy Optimization (GRPO) loss.

    Eliminates the critic/value network by normalising rewards within a group
    of ``group_size`` completions sampled for the same prompt.  The
    normalised group rewards become the advantage estimates fed into a
    standard PPO clipped objective.  Introduced in DeepSeekMath and used in
    DeepSeek-R1 and subsequent reasoning models.

    Total token count must be divisible by ``group_size``.

    Args:
        log_probs: Log-probabilities under the current policy, shape
            ``[batch * group_size]``.
        old_log_probs: Log-probabilities under the old policy, same shape.
        rewards: Scalar reward per completion, same shape.
        group_size: Number of completions sampled per prompt.
        clip_eps: PPO clipping threshold for the probability ratio.

    Returns:
        Scalar GRPO policy loss.
    """
    rewards_grouped = rewards.view(-1, group_size)
    mean = rewards_grouped.mean(dim=-1, keepdim=True)
    std = rewards_grouped.std(dim=-1, keepdim=True) + 1e-8
    advantages = ((rewards_grouped - mean) / std).view(-1)
    return _ppo_clip(log_probs, old_log_probs, advantages, clip_eps)


def contrastive_nce_fn(
    image_feats: torch.Tensor, text_feats: torch.Tensor, temp: float = 0.07
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style) contrastive loss.

    Aligns image and text embeddings by treating each matching pair as a
    positive and all other pairs in the batch as negatives.  L2-normalises
    both modalities before computing the similarity matrix, then averages
    the image-to-text and text-to-image cross-entropy losses.  Used in
    CLIP, SigLIP, ALIGN, and similar vision-language pretraining frameworks.

    Args:
        image_feats: Image embeddings of shape ``[batch, embed_dim]``.
        text_feats: Text embeddings of shape ``[batch, embed_dim]``.
        temp: Logit temperature scale (CLIP uses a learned temperature
            initialised at 0.07).

    Returns:
        Scalar symmetric contrastive loss.
    """
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    logits = image_feats @ text_feats.T / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# Workload registry
# ---------------------------------------------------------------------------


def build_workloads(
    device: str,
    dtype: torch.dtype,
    shape_preset: str = "prefill",
) -> list[WorkloadSpec]:
    """Construct the list of benchmark workloads for a given shape preset.

    Each :class:`WorkloadSpec` wraps a ``build`` callable that, when invoked,
    returns ``(fn, args, kwargs)``.  The caller compiles ``fn`` and then passes
    ``args``/``kwargs`` to the profiler.  Tensors are freshly allocated
    on every ``build()`` call so the compiled function always receives live
    inputs without aliasing.

    Workloads span four domains:

    - **llm** - core Transformer primitives (norms, MLPs, RoPE, sampling, routing).
    - **vlm** - vision encoder operations (patch embedding, ViT MLP block, pooling).
    - **diffusion** - U-Net / VAE / scheduler ops for image-generation models.
    - **rl** - training-time loss functions (CE, KL, PPO, DPO, GRPO, contrastive).

    Args:
        device: Target device string (e.g. ``"cuda"``).
        dtype: Floating-point dtype for all tensors (e.g. ``torch.bfloat16``).
        shape_preset: Either ``"decode"`` (large batch, ``s=1``, simulating
            autoregressive decoding) or ``"prefill"`` (small batch, long
            sequence, representative of prefill and training forward passes).

    Returns:
        List of :class:`WorkloadSpec` instances ready for benchmarking.

    Raises:
        ValueError: If ``shape_preset`` is not ``"decode"`` or ``"prefill"``.
    """
    torch.manual_seed(0)

    if shape_preset not in {"decode", "prefill"}:
        raise ValueError("shape_preset must be 'decode' or 'prefill'")

    h, n_heads, d = 2048, 16, 128
    inter = 5504
    if shape_preset == "decode":
        # Batched autoregressive decode: large batch, single new token
        b, s = 64, 1
        vlm_img, vlm_tokens = 224, 196
        diff_unet_hw, diff_latent_hw = 32, 64
    else:
        # Long-context prefill (also representative of training forward pass)
        b, s = 16, 2048
        vlm_img, vlm_tokens = 448, 1024
        diff_unet_hw, diff_latent_hw = 64, 128

    cos, sin = _rope_cos_sin(s, d, device, dtype)

    def rn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype)

    def rd(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=device, dtype=dtype)

    def ri(n: int, *shape: int) -> torch.Tensor:
        return torch.randint(0, n, shape, device=device)

    return [
        WorkloadSpec("llm_rmsnorm", "llm", ("reduction", "elementwise"),
            lambda: (rmsnorm_fn, (rn(b, s, h), rn(h)), {})),
        WorkloadSpec("llm_layernorm", "llm", ("reduction", "elementwise"),
            lambda: (layernorm_fn, (rn(b, s, h), rn(h), rn(h)), {})),
        WorkloadSpec("llm_swiglu_mlp", "llm", ("gemm", "elementwise", "fused-mlp"),
            lambda: (swiglu_mlp_fn, (rn(b * s, h), rn(h, inter), rn(h, inter), rn(inter, h)), {})),
        WorkloadSpec("llm_gelu_mlp", "llm", ("gemm", "elementwise", "fused-mlp"),
            lambda: (gelu_mlp_fn, (rn(b * s, h), rn(h, inter), rn(inter, h)), {})),
        WorkloadSpec("llm_rope_apply", "llm", ("elementwise", "indexing"),
            lambda: (rope_apply_fn, (rn(b, s, n_heads, d), cos, sin), {})),
        WorkloadSpec("llm_logits_projection_softmax", "llm", ("gemm", "softmax", "vocab"),
            lambda: (logits_softmax_fn, (rn(b * s, h), rn(h, 32768)), {})),
        WorkloadSpec("llm_topk_sampling_prep", "llm", ("softmax", "topk", "sampling"),
            lambda: (topk_sampling_prep_fn, (rn(b * s, 32768),), {})),
        WorkloadSpec("llm_moe_router", "llm", ("gemm", "softmax", "topk"),
            lambda: (moe_router_fn, (rn(b * s, h), rn(h, 64)), {})),
        WorkloadSpec("llm_qk_norm", "llm", ("reduction", "elementwise"),
            lambda: (qk_norm_fn, (rn(b, n_heads, s, d), rn(b, n_heads, s, d)), {})),
        WorkloadSpec("vlm_patch_embedding_conv2d", "vlm", ("conv", "reshape"),
            lambda: (patch_embed_fn, (rn(8, 3, vlm_img, vlm_img), rn(1024, 3, 14, 14)), {})),
        WorkloadSpec("vlm_vit_mlp_block", "vlm", ("layernorm", "gemm", "residual"),
            lambda: (vit_mlp_block_fn,
                (rn(8, vlm_tokens, 1024), rn(1024), rn(1024), rn(1024, 4096), rn(4096, 1024)), {})),
        WorkloadSpec("vlm_vision_rmsnorm_pool", "vlm", ("reduction", "pooling", "elementwise"),
            lambda: (vision_rmsnorm_pool_fn, (rn(8, vlm_tokens, 1024), rn(1024)), {})),
        WorkloadSpec("vlm_contrastive_nce_loss", "vlm", ("loss", "contrastive", "normalization"),
            lambda: (contrastive_nce_fn, (rn(256, 1024), rn(256, 1024)), {})),
        WorkloadSpec("diff_unet_resblock", "diffusion", ("conv", "groupnorm", "residual"),
            lambda: (unet_resblock_fn,
                (rn(8, 320, diff_unet_hw, diff_unet_hw), rn(320, 320, 3, 3),
                 rn(320, 320, 3, 3), rn(8, 320, diff_unet_hw, diff_unet_hw)), {})),
        WorkloadSpec("diff_vae_decoder_upsample", "diffusion", ("upsample", "conv"),
            lambda: (vae_upsample_fn, (rn(8, 256, 64, 64), rn(256, 256, 3, 3), rn(128, 256, 3, 3)), {})),
        WorkloadSpec("diff_time_embedding_mlp", "diffusion", ("embedding", "mlp"),
            # t is float32: sinusoidal freqs require fp32 precision.
            lambda: (time_embed_mlp_fn,
                (torch.rand(8, device=device, dtype=torch.float32), rn(320, 1280), rn(1280, 1280)), {})),
        WorkloadSpec("diff_cfg_blend", "diffusion", ("elementwise", "memory-bound"),
            lambda: (cfg_blend_fn,
                (rn(8, 4, diff_latent_hw, diff_latent_hw), rn(8, 4, diff_latent_hw, diff_latent_hw), 7.5), {})),
        WorkloadSpec("diff_scheduler_step", "diffusion", ("elementwise", "memory-bound"),
            lambda: (diffusion_scheduler_step_fn,
                (rn(8, 4, diff_latent_hw, diff_latent_hw), rn(8, 4, diff_latent_hw, diff_latent_hw), rd(8)), {})),
        WorkloadSpec("llm_label_smoothed_ce_loss", "llm", ("loss", "softmax", "vocab"),
            lambda: (label_smoothed_ce_fn, (rn(b * s, 32768), ri(32768, b * s)), {})),
        WorkloadSpec("llm_kl_divergence_loss", "llm", ("loss", "distillation", "softmax"),
            lambda: (kl_div_fn, (rn(b * s, 32768), rn(b * s, 32768)), {})),
        WorkloadSpec("rl_ppo_clipped_loss", "rl", ("loss", "policy-gradient", "clipping"),
            lambda: (ppo_clipped_loss_fn, (rn(b * s), rn(b * s), rn(b * s)), {})),
        WorkloadSpec("rl_dpo_loss", "rl", ("loss", "preference-optimization"),
            # 512 preference pairs; DPO uses one log-prob per completion, not per token.
            lambda: (dpo_loss_fn, (rn(512), rn(512), rn(512), rn(512)), {})),
        WorkloadSpec("rl_grpo_loss", "rl", ("loss", "policy-gradient", "group-normalization"),
            # 64 prompts x 8 completions = 512 total.
            lambda: (grpo_loss_fn, (rn(512), rn(512), rn(512)), {})),
    ]


# ---------------------------------------------------------------------------
# Profilers
# ---------------------------------------------------------------------------


def _triton_do_bench(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    warmup: int,
    rep: int,
) -> dict[str, float]:
    """Benchmark a function using Triton's ``do_bench`` harness.

    Args:
        fn: Callable to benchmark.
        args: Positional arguments forwarded to ``fn``.
        kwargs: Keyword arguments forwarded to ``fn``.
        warmup: Warm-up time budget in milliseconds passed to Triton.
        rep: Repetition time budget in milliseconds passed to Triton.

    Returns:
        Dict with keys ``median_ms``, ``p25_ms``, ``p99_ms``.
    """
    ms, p25, p99 = triton.testing.do_bench(
        lambda: fn(*args, **kwargs),
        warmup=warmup,
        rep=rep,
        quantiles=[0.5, 0.25, 0.99],
    )
    return {
        "median_ms": float(ms),
        "p25_ms": float(p25),
        "p99_ms": float(p99),
    }


def _compile_fn(fn: Any, mode: str | None) -> Any:
    """Optionally wrap a function with ``torch.compile``.

    If ``mode`` is ``None`` (eager baseline), the original function is returned
    unchanged.  Otherwise ``torch._dynamo.reset()`` is called first to clear
    any cached compilation state, then the function is compiled with
    ``dynamic=False`` to target fixed-shape kernels matching the benchmark
    inputs.

    Note: Each combination runs in a fresh subprocess, so the dynamo reset is
    technically redundant but kept as a defensive measure.

    Args:
        fn: The function to compile.
        mode: ``torch.compile`` mode string (e.g. ``"max-autotune"``) or
            ``None`` for eager execution.

    Returns:
        The compiled function, or ``fn`` unchanged when ``mode`` is ``None``.
    """
    if mode is None:
        return fn
    torch._dynamo.reset()
    return torch.compile(fn, mode=mode, dynamic=False)


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------


def _analyze(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Compute per-mode performance statistics and write a summary JSON file.

    Filters to rows with ``status == "ok"``, then:

    - Ranks each ``(shape_preset, workload)`` pair by median latency.
    - Counts wins (rank-1) and top-2 appearances per mode.
    - Computes geometric-mean speedup relative to the ``"eager"`` baseline.

    Args:
        df: Raw results DataFrame containing all modes.
        out_dir: Directory where ``summary.json`` will be written.

    Returns:
        Summary dict that is also serialised to ``summary.json``.
    """
    ok_df = (
        df[df["status"] == "ok"]
        .drop_duplicates(subset=["shape_preset", "workload", "mode"])
        .copy()
    )

    summary: dict[str, Any] = {
        "num_workloads": int(df["workload"].nunique()),
        "shape_presets": sorted(df["shape_preset"].dropna().unique().tolist()),
        "num_workload_instances": int(
            df[["shape_preset", "workload"]].drop_duplicates().shape[0]
        ),
    }

    ranks = ok_df.groupby(["shape_preset", "workload"])["median_ms"].rank(
        method="min", ascending=True
    )
    ok_df = ok_df.assign(rank=ranks.rename("rank"))

    rank1 = ok_df[ok_df["rank"] == 1.0]
    wins = rank1.groupby("mode")["workload"].nunique().to_dict()
    win_workloads: dict[str, list[str]] = (
        rank1.groupby("mode")["workload"].apply(sorted).apply(list).to_dict()
    )
    avg_rank = ok_df.groupby("mode")["rank"].mean().to_dict()
    top2 = (
        ok_df[ok_df["rank"] <= 2.0].groupby("mode")["workload"].nunique()
        / ok_df[["shape_preset", "workload"]].drop_duplicates().shape[0]
    ).to_dict()

    base = ok_df[ok_df["mode"] == "eager"][
        ["shape_preset", "workload", "median_ms"]
    ].rename(columns={"median_ms": "eager_ms"})
    merged = ok_df.merge(base, on=["shape_preset", "workload"], how="inner")
    merged["speedup_vs_eager"] = merged["eager_ms"] / merged["median_ms"]
    # Clip to a small positive floor before log to guard against NaN
    # latencies from error rows slipping through the status filter.
    gmean_speedup = (
        merged.groupby("mode")["speedup_vs_eager"]
        .apply(lambda x: math.exp(x.clip(lower=1e-9).map(math.log).mean()))
        .to_dict()
    )

    summary["wins"] = {k: int(v) for k, v in wins.items()}
    summary["win_workloads"] = win_workloads
    summary["avg_rank"] = {k: float(v) for k, v in avg_rank.items()}
    summary["top2_rate"] = {k: float(v) for k, v in top2.items()}
    summary["gmean_speedup_vs_eager"] = {k: float(v) for k, v in gmean_speedup.items()}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_report(
    df: pd.DataFrame, summary: dict[str, Any], out_dir: Path, warmup: int, rep: int
) -> None:
    """Render a human-readable Markdown report and write it to disk.

    Produces a ``report.md`` in ``out_dir`` containing GPU / PyTorch setup
    info, aggregate compile-mode statistics from ``summary``, qualitative
    reasoning for why ``max-autotune`` is not always the fastest choice, and
    per-tag average latency tables.

    Args:
        df: Raw results DataFrame.
        summary: Dict returned by :func:`_analyze`.
        out_dir: Directory to write ``report.md``.
        warmup: Warm-up iteration count used during the run (written to the
            setup section for reproducibility).
        rep: Repetition count used during the run (written to the setup
            section).
    """
    compile_df = df[df["status"] == "ok"].copy()
    by_tag = (
        compile_df.groupby(["tag", "mode"])["median_ms"]
        .mean()
        .reset_index()
        .sort_values(["tag", "median_ms"])
    )

    report = [
        "# Torch Compile Mode Study",
        "",
        "## Setup",
        f"- GPU: {torch.cuda.get_device_name(0)}",
        f"- PyTorch: {torch.__version__}",
        "- Compile modes: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs",
        "- Additional baseline: eager",
        f"- Warmup: {warmup}, Rep: {rep}",
        "- Profiler: Triton `do_bench`",
        "",
        "## Aggregate Results",
        "",
        f"- Number of workloads: {summary['num_workloads']}",
        f"- Shape presets: {', '.join(summary['shape_presets'])}",
        f"- Workload instances (shape x workload): {summary['num_workload_instances']}",
        "",
        "Wins (best latency count):",
        *[
            f"- {k}: {v} — {', '.join(summary['win_workloads'].get(k, []))}"
            for k, v in sorted(summary["wins"].items())
        ],
        "",
        "Average rank (lower is better):",
        *[f"- {k}: {v:.3f}" for k, v in sorted(summary["avg_rank"].items())],
        "",
        "Geometric mean speedup vs `eager`:",
        *[
            f"- {k}: {v:.3f}x"
            for k, v in sorted(summary["gmean_speedup_vs_eager"].items())
        ],
        "",
        "## Why `max-autotune` Is Not Always Best",
        "",
        "1. `max-autotune` targets kernel-level candidates and often prioritizes steady-state throughput for GEMM-heavy regions;",
        "   this can miss wins for launch-bound, memory-bound, or reduction-heavy graphs where overhead dominates.",
        "2. CUDA graph capture can help stable static workloads, but it can also reduce flexibility for workloads with buffer updates",
        "   or index-heavy/control-flow-heavy operations, where `max-autotune-no-cudagraphs` can be faster.",
        "3. `reduce-overhead` can outperform when kernels are already near-optimal and host launch overhead is a meaningful fraction",
        "   of total latency (small or medium kernels, many lightweight ops).",
        "",
        "## Heuristic Recommendations",
        "",
        "- Prefer `max-autotune` for large GEMM/attention-dominated static-shape blocks.",
        "- Prefer `max-autotune-no-cudagraphs` for stateful updates (`kv` cache writes), heavy indexing, or graph-capture friction.",
        "- Prefer `reduce-overhead` for launch-bound chains of small/medium ops (norm + elementwise + light projections).",
        "- Start with `default` when shape dynamism is expected, then A/B test against `reduce-overhead` and",
        "  `max-autotune-no-cudagraphs` on representative production shapes.",
        "",
        "## Tag-Level Averages (Lower Is Better)",
        "",
    ]

    for tag, gdf in by_tag.groupby("tag"):
        best_row = gdf.nsmallest(1, "median_ms").iloc[0]
        report.append(
            f"- {tag}: best `{best_row['mode']}` ({best_row['median_ms']:.3f} ms)"
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PRESET_COLORS: dict[str, str] = {"prefill": "#1f77b4", "decode": "#2ca02c"}
_MODE_ORDER: list[str] = [
    "eager",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


def _plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate per-workload and geomean bar charts from benchmark results.

    For each workload, produces a grouped bar chart with compile modes on the
    x-axis and one bar per shape preset (blue for prefill, green for decode).
    A final summary chart shows the geometric-mean speedup vs eager across all
    workloads for each mode and preset.

    Charts are saved as PNG files in ``out_dir/plots/``.

    Args:
        df: Raw results DataFrame (``status == "ok"`` rows will be selected).
        out_dir: Root output directory; plots are written to ``out_dir/plots/``.
    """
    ok_df = (
        df[df["status"] == "ok"]
        .drop_duplicates(subset=["shape_preset", "workload", "mode"])
        .copy()
    )
    if ok_df.empty:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    presets = sorted(ok_df["shape_preset"].unique())
    modes = [m for m in _MODE_ORDER if m in ok_df["mode"].unique()]

    # --- Per-workload charts ---
    for workload, wdf in ok_df.groupby("workload"):
        _plot_workload_chart(wdf, workload, presets, modes, plots_dir)

    # --- Geomean speedup chart ---
    _plot_geomean_chart(ok_df, presets, modes, plots_dir)


def _plot_workload_chart(
    wdf: pd.DataFrame,
    workload: str,
    presets: list[str],
    modes: list[str],
    plots_dir: Path,
) -> None:
    """Plot a single workload's median latency as a grouped bar chart.

    Args:
        wdf: DataFrame filtered to a single workload.
        workload: Workload name (used for title and filename).
        presets: Ordered list of shape presets.
        modes: Ordered list of compile modes.
        plots_dir: Directory to save the chart.
    """
    pivot = wdf.pivot_table(
        index="mode", columns="shape_preset", values="median_ms", aggfunc="first"
    )

    x = np.arange(len(modes))
    n_presets = len(presets)
    width = 0.8 / n_presets

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, preset in enumerate(presets):
        if preset not in pivot.columns:
            continue
        vals = [pivot.loc[m, preset] if m in pivot.index else 0.0 for m in modes]
        offset = (i - (n_presets - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width, label=preset, color=_PRESET_COLORS.get(preset)
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title(workload)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{workload}.png", dpi=150)
    plt.close(fig)


def _plot_geomean_chart(
    ok_df: pd.DataFrame,
    presets: list[str],
    modes: list[str],
    plots_dir: Path,
) -> None:
    """Plot geometric-mean speedup vs eager for each mode and shape preset.

    Args:
        ok_df: DataFrame of all successful benchmark rows.
        presets: Ordered list of shape presets.
        modes: Ordered list of compile modes.
        plots_dir: Directory to save the chart.
    """
    base = ok_df[ok_df["mode"] == "eager"][
        ["shape_preset", "workload", "median_ms"]
    ].rename(columns={"median_ms": "eager_ms"})
    merged = ok_df.merge(base, on=["shape_preset", "workload"], how="inner")
    merged["speedup"] = merged["eager_ms"] / merged["median_ms"]

    gmean = (
        merged.groupby(["shape_preset", "mode"])["speedup"]
        .apply(lambda s: math.exp(s.clip(lower=1e-9).map(math.log).mean()))
        .reset_index(name="gmean_speedup")
    )
    pivot = gmean.pivot_table(
        index="mode", columns="shape_preset", values="gmean_speedup", aggfunc="first"
    )

    x = np.arange(len(modes))
    n_presets = len(presets)
    width = 0.8 / n_presets

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, preset in enumerate(presets):
        if preset not in pivot.columns:
            continue
        vals = [pivot.loc[m, preset] if m in pivot.index else 0.0 for m in modes]
        offset = (i - (n_presets - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width, label=preset, color=_PRESET_COLORS.get(preset)
        )
        ax.bar_label(bars, fmt="%.3fx", fontsize=8, padding=2)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Geometric mean speedup vs eager")
    ax.set_title("Geomean Speedup vs Eager (all workloads)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "geomean_speedup.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subprocess worker and orchestration
# ---------------------------------------------------------------------------


def _error_rows_for_workload(
    wl: WorkloadSpec,
    shape_preset: str,
    mode_name: str,
    err: str,
) -> list[dict[str, Any]]:
    """Build error-status CSV rows for every tag of a workload.

    Called when a subprocess fails (timeout, non-zero exit code, or unhandled
    exception) so the output DataFrame always contains a complete row per tag
    with ``NaN`` latency columns and the error string rather than silently
    missing entries.

    Args:
        wl: The :class:`WorkloadSpec` that failed.
        shape_preset: Shape preset string (e.g. ``"decode"``).
        mode_name: Compile mode label (e.g. ``"max-autotune"``).
        err: Error message or formatted traceback string.

    Returns:
        List of row dicts, one per tag.
    """
    nan = float("nan")
    return [
        {
            "workload": wl.name,
            "shape_preset": shape_preset,
            "domain": wl.domain,
            "tag": tag,
            "mode": mode_name,
            "status": "error",
            "error": err,
            "elapsed_wall_s": 0.0,
            "median_ms": nan,
            "p25_ms": nan,
            "p99_ms": nan,
        }
        for tag in wl.tags
    ]


def _worker_combo(
    queue: mp.Queue,
    workload_index: int,
    shape_preset: str,
    mode_name: str,
    mode_value: str | None,
    dtype_name: str,
    warmup: int,
    rep: int,
) -> None:
    """Subprocess entry point: compile, warm up, and profile one workload/mode combination.

    Runs inside a freshly spawned process to guarantee CUDA context and
    compile-cache isolation between modes.  Results - or the exception
    traceback on failure - are sent back to the parent via ``queue``.

    The function performs several warmup calls before handing off to the
    profiler.  For ``max-autotune``, the first few calls may still be
    compilation passes rather than steady-state kernel launches; extra warmup
    iterations ensure the profiler measures only compiled throughput.

    Args:
        queue: Multiprocessing queue used to return results to the parent
            process.
        workload_index: Index into the list returned by
            :func:`build_workloads`.
        shape_preset: Shape preset string (e.g. ``"decode"``).
        mode_name: Human-readable compile mode label (used in output rows).
        mode_value: ``torch.compile`` mode string, or ``None`` for eager.
        dtype_name: String name of the torch dtype (e.g. ``"bfloat16"``).
        warmup: Number of warm-up iterations passed to the profiler.
        rep: Number of timed repetitions passed to the profiler.
    """
    try:
        # Use tensorfloat 32 dtype
        torch.set_float32_matmul_precision("high")
        workloads = build_workloads(
            device="cuda",
            dtype=getattr(torch, dtype_name),
            shape_preset=shape_preset,
        )
        wl = workloads[workload_index]
        fn, fn_args, fn_kwargs = wl.build()

        compiled = _compile_fn(fn, mode_value)

        # Run several warmup iterations before profiling.  For max-autotune
        # the first call triggers autotuning and may consume most of a
        # single-pass warmup budget inside do_bench, leaving the profiler
        # measuring partially-compiled kernels.  Twenty passes reliably reach
        # steady state for all modes tested.
        for _ in range(20):
            compiled(*fn_args, **fn_kwargs)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        metrics = _triton_do_bench(compiled, fn_args, fn_kwargs, warmup, rep)
        elapsed_s = time.perf_counter() - t0

        rows: list[dict[str, Any]] = [
            {
                "workload": wl.name,
                "shape_preset": shape_preset,
                "domain": wl.domain,
                "tag": tag,
                "mode": mode_name,
                "status": "ok",
                "elapsed_wall_s": elapsed_s,
                **metrics,
            }
            for tag in wl.tags
        ]

        queue.put({"rows": rows})
    except Exception as exc:
        tb = traceback.format_exc(limit=5)
        queue.put({"error": f"{type(exc).__name__}: {exc}\n{tb}"})


def _run_combo_in_subprocess(
    workload_index: int,
    shape_preset: str,
    mode_name: str,
    mode_value: str | None,
    wl: WorkloadSpec,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Spawn a child process to benchmark one (workload, mode) combination.

    Each combination runs in its own ``spawn``-context process If the
    process exceeds ``args.timeout_s`` seconds, is killed, or raises an
    unhandled exception, error rows are returned in place of benchmark rows so
    the outer loop can continue.

    Args:
        workload_index: Index of the workload in the list from
            :func:`build_workloads`.
        shape_preset: Shape preset string (e.g. ``"decode"``).
        mode_name: Human-readable compile mode label.
        mode_value: ``torch.compile`` mode string, or ``None`` for eager.
        wl: :class:`WorkloadSpec` for ``workload_index`` - used only for
            constructing error rows; the worker reconstructs it independently.
        args: Parsed CLI arguments (supplies ``timeout_s``, ``warmup``,
            ``rep``, ``dtype``).

    Returns:
        List of row dicts suitable for appending to the results DataFrame.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker_combo,
        args=(
            queue,
            workload_index,
            shape_preset,
            mode_name,
            mode_value,
            args.dtype,
            args.warmup,
            args.rep,
        ),
    )
    proc.start()
    proc.join(timeout=args.timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return _error_rows_for_workload(
            wl, shape_preset, mode_name, f"TimeoutError: exceeded {args.timeout_s}s"
        )

    if proc.exitcode != 0:
        return _error_rows_for_workload(
            wl,
            shape_preset,
            mode_name,
            f"SubprocessError: worker exited with code {proc.exitcode}",
        )

    if queue.empty():
        return _error_rows_for_workload(
            wl, shape_preset, mode_name, "RuntimeError: worker returned no payload"
        )

    payload = queue.get_nowait()
    if "error" in payload:
        return _error_rows_for_workload(wl, shape_preset, mode_name, payload["error"])
    return payload["rows"]


def run(args: argparse.Namespace) -> None:
    """Run the full benchmark study and write results to disk.

    Iterates over all ``(shape_preset x workload x compile_mode)`` combinations,
    running each in an isolated subprocess via :func:`_run_combo_in_subprocess`.
    Writes three output files to ``args.output_dir``:

    - ``raw_results.csv`` - one row per ``(shape_preset, workload, mode, tag)``.
    - ``summary.json`` - aggregate statistics (wins, ranks, speedups, gaps).
    - ``report.md`` - human-readable Markdown summary.

    If ``--analyze-only`` is set, benchmarking is skipped and the summary and
    report are recomputed from an existing ``raw_results.csv``.

    Args:
        args: Parsed CLI arguments from :func:`parse_args`.

    Raises:
        AssertionError: If CUDA is not available.
        FileNotFoundError: If ``--analyze-only`` is set but no CSV exists.
        ValueError: If an unrecognised shape preset is requested.
    """
    assert torch.cuda.is_available(), "CUDA is required for this study"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    if args.analyze_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw results file: {raw_path}")
        df = pd.read_csv(raw_path)
        summary = _analyze(df, out_dir)
        _write_report(df, summary, out_dir, args.warmup, args.rep)
        _plot_results(df, out_dir)
        print(f"Recomputed summary from {raw_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved report: {report_path}")
        return

    shape_presets = [s.strip() for s in args.shape_presets.split(",") if s.strip()]
    valid_shape_presets = {"decode", "prefill"}
    invalid = [s for s in shape_presets if s not in valid_shape_presets]
    if invalid:
        raise ValueError(
            f"Invalid shape preset(s): {invalid}. Valid values: {sorted(valid_shape_presets)}"
        )

    rows: list[dict[str, Any]] = []

    for shape_preset in shape_presets:
        # Use device="cpu" here: we only need workload names and count for
        # progress reporting; the actual benchmarking runs in subprocesses
        # that always use CUDA.
        workloads = build_workloads(device="cpu", dtype=torch.bfloat16, shape_preset=shape_preset)

        for i, wl in enumerate(workloads, start=1):
            print(f"[{shape_preset}] [{i:02d}/{len(workloads)}] workload={wl.name}")

            for mode_name, mode_value in ALL_MODES.items():
                combo_rows = _run_combo_in_subprocess(
                    i - 1, shape_preset, mode_name, mode_value, wl, args
                )
                rows.extend(combo_rows)
                if combo_rows and combo_rows[0]["status"] != "ok":
                    err = combo_rows[0].get("error", "unknown error")
                    print(
                        f"  FAILED workload={wl.name} shape_preset={shape_preset}"
                        f" mode={mode_name}: {err}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)

    summary = _analyze(df, out_dir)
    _write_report(df, summary, out_dir, args.warmup, args.rep)
    _plot_results(df, out_dir)

    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the compile-mode study.

    Returns:
        Populated :class:`argparse.Namespace` with fields:
        ``output_dir`` (str), ``dtype`` (str), ``warmup`` (int),
        ``rep`` (int), ``timeout_s`` (int), ``shape_presets`` (str),
        ``analyze_only`` (bool).
    """
    parser = argparse.ArgumentParser(description="Torch compile-mode comparative study")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/torch_compile_mode_study",
        help="Directory for CSV/JSON/Markdown outputs",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=1000)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument(
        "--shape-presets",
        type=str,
        default="decode,prefill",
        help="Comma-separated shape presets to benchmark: decode,prefill",
    )
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
