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

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.layernorm import rmsnorm
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend RMSNorm tests require an NPU"
)


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm(dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    weight = torch.randn(1024, device="npu", dtype=dtype)
    expected = _reference(x, weight, 1e-6)

    out = torch.empty_like(x)
    result = rmsnorm(x, weight, 1e-6, out=out)

    assert result is out
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_rmsnorm_updates_residual(dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(1024, device="npu", dtype=dtype)
    expected_residual = x + residual
    expected = _reference(expected_residual, weight, 1e-6)

    result, residual_out = rmsnorm(x, weight, 1e-6, residual=residual)

    assert residual_out is residual
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


def test_gemma_rmsnorm_uses_effective_weight_and_updates_residual() -> None:
    torch.manual_seed(23)
    width = 1024
    eps = 1e-6
    layer = GemmaRMSNorm(width, eps=eps).to(device="npu", dtype=torch.bfloat16)
    checkpoint_weight = torch.randn(width, dtype=torch.bfloat16, device="npu")
    layer._weight_loader(layer.weight, checkpoint_weight)
    x = torch.randn(7, width, dtype=torch.bfloat16, device="npu")
    residual = torch.randn_like(x)
    expected_residual = x + residual
    expected = _reference(expected_residual, checkpoint_weight + 1, eps)

    result, fp8_result, updated_residual = layer.add_norm_with_fp8(
        x, residual, None
    )

    assert fp8_result is None
    assert updated_residual is residual
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("sigmoid_gate", [False, True])
def test_gated_rmsnorm_matches_reference(
    norm_before_gate: bool, sigmoid_gate: bool
) -> None:
    torch.manual_seed(29)
    x = torch.randn(5, 2, 128, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, dtype=torch.bfloat16)
    gate = (
        torch.sigmoid(z.float())
        if sigmoid_gate
        else torch.nn.functional.silu(z.float())
    )
    norm_input = x.float() if norm_before_gate else x.float() * gate
    expected = _reference(norm_input, weight, 1e-6).to(x.dtype)
    if norm_before_gate:
        expected = (expected.float() * gate).to(x.dtype)

    actual = gated_rmsnorm(
        x.npu(),
        weight.npu(),
        z.npu(),
        eps=1e-6,
        group_size=128,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
        weights_independent=True,
    )

    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)
