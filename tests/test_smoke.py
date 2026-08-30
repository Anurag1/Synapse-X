import tempfile

import torch

from synapse_x_core.orchestrator import SynapseX, text_to_tensor, tensor_to_text


def test_text_round_trip():
    text = "hello synapse"
    encoded = text_to_tensor(text)
    assert encoded.shape == (1, 128)
    assert tensor_to_text(encoded) == text


def test_engine_initializes_without_external_model_call():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = SynapseX(skill_library_path=tmpdir)
        assert engine.skills == {}
        assert engine.meta_mind.model_name == "llama3"


def test_tensor_dtype():
    encoded = text_to_tensor("abc")
    assert encoded.dtype == torch.float32
