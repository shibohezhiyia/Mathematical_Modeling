from core.semantic_model_compiler import OfflineSemanticBackend, SemanticCompilerConfig


def test_offline_backend_is_deterministic_and_empty():
    assert OfflineSemanticBackend().complete([]) == '{"hypotheses":[],"questions":[]}'
    assert SemanticCompilerConfig(provider="offline", model_name="offline").validate().public()["provider"] == "offline"
