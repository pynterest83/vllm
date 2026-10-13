# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA recurrent-state storage dtype.

``--mamba-ssm-cache-dtype`` selects the recurrent-state storage dtype ("auto"
keeps float32). The layer's allocation and the model-level dtype used for the
hybrid block-size derivation must agree, or the attention page and the KDA
state page are sized from different numbers.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)


def _configs(ssm_dtype: str):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto", mamba_ssm_cache_dtype=ssm_dtype
    )
    return model_config, cache_config


@pytest.mark.parametrize(
    "ssm_dtype,expected",
    [
        ("auto", torch.float32),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_layer_and_model_state_dtype_agree(ssm_dtype, expected):
    model_config, cache_config = _configs(ssm_dtype)
    layer = object.__new__(Glm5NextLinearAttention)
    layer.model_config = model_config
    layer.cache_config = cache_config
    vllm_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)

    conv_dtype, state_dtype = layer.get_state_dtype()
    assert conv_dtype == torch.bfloat16
    assert state_dtype == expected
    for cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert cls.get_mamba_state_dtype_from_config(vllm_config) == (
            conv_dtype,
            state_dtype,
        )


def test_float16_state_rejected():
    model_config, cache_config = _configs("float16")
    vllm_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)
    with pytest.raises(ValueError, match="KDA recurrent state"):
        Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
