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

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)


def _normalized(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = _normalized(q)
    k = _normalized(k)
    state = state.float().transpose(-2, -1).clone()
    outputs = []
    for token in range(q.shape[1]):
        state = state * torch.exp(g[:, token].float())[..., None, None]
        delta = beta[:, token].float()[..., None] * (
            v[:, token].float()
            - torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        )
        state = state + k[:, token][..., None] * delta[..., None, :]
        outputs.append(
            scale * torch.einsum("bhk,bhkv->bhv", q[:, token], state)
        )
    return torch.stack(outputs, dim=1), state.transpose(-2, -1)


@pytest.mark.parametrize("lengths", [[1], [9], [3, 11]])
def test_gdn_prefill_matches_fp32_recurrence(lengths: list[int]) -> None:
    torch.manual_seed(sum(lengths))
    heads = 2
    head_dim = 128
    total = sum(lengths)
    q = torch.randn(1, total, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q) * 0.25
    g = -torch.rand(1, total, heads, dtype=torch.float32)
    beta = torch.rand(1, total, heads, dtype=torch.bfloat16)
    state = torch.randn(len(lengths), heads, head_dim, head_dim) * 0.01
    scale = head_dim**-0.5

    expected_out = torch.empty_like(v)
    expected_states = []
    start = 0
    for request, length in enumerate(lengths):
        end = start + length
        output, final_state = _reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            state[request : request + 1],
            scale,
        )
        expected_out[:, start:end] = output.to(v.dtype)
        expected_states.append(final_state[0])
        start = end

    cumulative = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="npu",
    )
    result = gdn_chunk_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        scale=scale,
        initial_state=state.npu(),
        cu_seqlens=cumulative,
        qk_l2norm=True,
        solution="torch_npu",
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        result.out.cpu(), expected_out, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        result.final_state.cpu(),
        torch.stack(expected_states),
        rtol=2e-2,
        atol=3e-2,
    )


def test_gdn_decode_updates_only_each_requests_state_across_steps() -> None:
    torch.manual_seed(17)
    batch = 3
    live = 2
    steps = 3
    heads = 2
    head_dim = 128
    q = torch.randn(batch, steps, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q) * 0.25
    a = torch.randn(batch, steps, heads)
    b = torch.randn(batch, steps, heads)
    A_log = torch.randn(heads)
    dt_bias = torch.randn(heads)
    pool = torch.randn(5, heads, head_dim, head_dim) * 0.01
    original = pool.clone()
    read_indices = torch.tensor([1, 3, -1], dtype=torch.int32)
    scale = head_dim**-0.5

    gate = -torch.exp(A_log) * F.softplus(a[:live] + dt_bias)
    expected_out, expected_state = _reference(
        q[:live],
        k[:live],
        v[:live],
        gate,
        b[:live].sigmoid(),
        pool[read_indices[:live].long()],
        scale,
    )

    device_pool = pool.npu()
    actual_steps = []
    for token in range(steps):
        actual_steps.append(
            gdn_decode_step(
                q[:, token : token + 1].npu(),
                k[:, token : token + 1].npu(),
                v[:, token : token + 1].npu(),
                A_log=A_log.npu(),
                a=a[:, token : token + 1].npu(),
                dt_bias=dt_bias.npu(),
                b=b[:, token : token + 1].npu(),
                initial_state=device_pool,
                initial_state_indices=read_indices.npu(),
                scale=scale,
                solution="torch_npu",
            )[:live]
        )
    torch.npu.synchronize()
    actual_out = torch.cat(actual_steps, dim=1).cpu()
    actual_pool = device_pool.cpu()

    torch.testing.assert_close(
        actual_out, expected_out.to(actual_out.dtype), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_pool[read_indices[:live].long()],
        expected_state.to(actual_pool.dtype),
        rtol=2e-2,
        atol=3e-2,
    )
    # The padded request maps to no state, and unrelated requests stay isolated.
    torch.testing.assert_close(actual_pool[[0, 2, 4]], original[[0, 2, 4]])
