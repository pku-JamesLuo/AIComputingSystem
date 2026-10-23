# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend Gated DeltaNet kernels backed by the fused ``torch_npu`` operator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult

_MAX_RECURRENT_TOKENS = 8
_L2_NORM_EPS = 1e-6


def _normalize_qk(x: torch.Tensor) -> torch.Tensor:
    """Match TokenSpeed's FP32 GDN normalization and round once to BF16."""
    x_float = x.float()
    return (
        x_float
        * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + _L2_NORM_EPS)
    ).to(torch.bfloat16)


def _run_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    *,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    state_indices: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Invoke the native recurrence with its required contiguous dtypes."""
    return torch_npu.npu_recurrent_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.to(torch.bfloat16).contiguous(),
        state,
        beta=beta.to(torch.bfloat16).contiguous(),
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=state_indices,
        g=g.float().contiguous(),
        gk=None,
    )


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
) -> GdnChunkPrefillResult:
    """Run varlen GDN prefill through the Ascend recurrent fusion.

    The native operator accepts at most eight tokens per sequence. Longer
    prefills therefore advance the same per-request state in eight-token
    segments. TokenSpeed's public state layout is already the native K-last
    layout ``[request, value_head, value_dim, key_dim]``.
    """
    if q.shape[0] != 1 or q.dim() != 4:
        raise ValueError(f"Ascend GDN prefill expects q shaped [1, T, H, K], got {q.shape}")
    if not qk_l2norm:
        raise ValueError("Ascend GDN requires Q/K L2 normalization")
    if output_h:
        raise ValueError("Ascend GDN prefill does not expose intermediate checkpoints")

    scale = q.shape[-1] ** -0.5 if scale is None else scale
    q = _normalize_qk(q)
    k = _normalize_qk(k)
    state = initial_state.to(torch.bfloat16).contiguous().clone()
    out = torch.empty_like(v)

    # Prefill is eager on Ascend. One D2H copy gives stable Python bounds for
    # all segments and avoids synchronizing once per segment.
    bounds = cu_seqlens.to(device="cpu", dtype=torch.int64).tolist()
    if len(bounds) != initial_state.shape[0] + 1:
        raise ValueError(
            "cu_seqlens and initial_state disagree on request count: "
            f"{len(bounds) - 1} vs {initial_state.shape[0]}"
        )

    for request, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        for chunk_start in range(start, end, _MAX_RECURRENT_TOKENS):
            chunk_end = min(chunk_start + _MAX_RECURRENT_TOKENS, end)
            chunk_tokens = chunk_end - chunk_start
            state_indices = torch.full(
                (chunk_tokens,), request, dtype=torch.int32, device=q.device
            )
            actual_seq_lengths = torch.tensor(
                [chunk_tokens], dtype=torch.int32, device=q.device
            )
            chunk_out = _run_recurrent(
                q[0, chunk_start:chunk_end],
                k[0, chunk_start:chunk_end],
                v[0, chunk_start:chunk_end],
                state,
                beta=beta[0, chunk_start:chunk_end],
                scale=scale,
                actual_seq_lengths=actual_seq_lengths,
                state_indices=state_indices,
                g=g[0, chunk_start:chunk_end],
            )
            out[0, chunk_start:chunk_end].copy_(chunk_out)

    final_state = state.to(initial_state.dtype) if output_final_state else None
    return GdnChunkPrefillResult(out=out, final_state=final_state)


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Run one GDN decode step and publish each request's updated state."""
    if q.dim() != 4 or q.shape[1] != 1:
        raise ValueError(f"Ascend GDN decode expects q shaped [B, 1, H, K], got {q.shape}")
    if not use_qk_l2norm:
        raise ValueError("Ascend GDN requires Q/K L2 normalization")

    batch = q.shape[0]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    safe_read = initial_state_indices.clamp_min(0).long()
    valid_read = initial_state_indices >= 0
    state = initial_state.index_select(0, safe_read).to(torch.bfloat16).contiguous()
    state.masked_fill_(~valid_read[:, None, None, None], 0)

    gate = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    output = _run_recurrent(
        _normalize_qk(q).reshape(batch, q.shape[2], q.shape[3]),
        _normalize_qk(k).reshape(batch, k.shape[2], k.shape[3]),
        v.reshape(batch, v.shape[2], v.shape[3]),
        state,
        beta=b.sigmoid().reshape(batch, v.shape[2]),
        scale=scale,
        actual_seq_lengths=torch.ones(batch, dtype=torch.int32, device=q.device),
        state_indices=torch.arange(batch, dtype=torch.int32, device=q.device),
        g=gate.reshape(batch, v.shape[2]),
    )

    write_indices = (
        initial_state_indices
        if output_state_indices is None
        else output_state_indices
    )
    safe_write = write_indices.clamp_min(0).long()
    valid_write = write_indices >= 0
    old_rows = initial_state.index_select(0, safe_write)
    new_rows = state.to(initial_state.dtype)
    rows = torch.where(valid_write[:, None, None, None], new_rows, old_rows)
    initial_state.index_copy_(0, safe_write, rows)
    return output.reshape(batch, 1, v.shape[2], v.shape[3]).to(q.dtype)
