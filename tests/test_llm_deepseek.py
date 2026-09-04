import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.semantic_model_compiler import SemanticCompilerConfig
from extensions.llm_analyzer import (
    LLMClient,
    LLMConfig,
    AnalysisPromptBuilder,
    attach_multimodal_images,
    get_default_configs,
)
from web.app import _normalize_llm_images, app, user_sessions


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_deepseek_default_and_chat_completion_use_official_openai_endpoint(monkeypatch):
    defaults = get_default_configs()["deepseek"]
    assert defaults["base_url"] == "https://api.deepseek.com"
    assert defaults["model_name"] == "deepseek-v4-pro"
    assert defaults["model_options"] == ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"]

    captured = {}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return _FakeResponse({"choices": [{"message": {"content": "分析完成"}}]})

    monkeypatch.setattr("extensions.llm_analyzer.requests.post", fake_post)
    client = LLMClient(LLMConfig(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="ds-secret",
        model_name="deepseek-v4-pro",
    ))
    result = client.chat_completion([{"role": "user", "content": "test"}])
    assert result == "分析完成"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ds-secret"
    assert captured["json"]["model"] == "deepseek-v4-pro"
    assert "ds-secret" not in repr(client.config)


def test_deepseek_connection_test_uses_models_endpoint(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return _FakeResponse({"data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-v4-flash"}]})

    monkeypatch.setattr("extensions.llm_analyzer.requests.get", fake_get)
    models = LLMClient(LLMConfig(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="ds-secret",
        model_name="deepseek-v4-pro",
    )).list_models()
    assert models == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert captured["url"] == "https://api.deepseek.com/models"
    assert captured["allow_redirects"] is False


def test_deepseek_validation_requires_key_and_official_host():
    with pytest.raises(ValueError, match="API Key"):
        LLMConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com",
            model_name="deepseek-v4-pro",
        ).validate()
    with pytest.raises(ValueError, match="官方 API 地址"):
        LLMConfig(
            provider="deepseek",
            base_url="https://example.com",
            api_key="secret",
            model_name="deepseek-v4-pro",
        ).validate()


def test_semantic_compiler_accepts_deepseek_without_exposing_key(monkeypatch):
    monkeypatch.setattr(
        "core.semantic_model_compiler.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    config = SemanticCompilerConfig(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model_name="deepseek-v4-pro",
        api_key="ds-secret",
    ).validate()
    assert config.public()["api_key_configured"] is True
    assert "ds-secret" not in repr(config)
    assert "ds-secret" not in json.dumps(config.public())


def test_llm_connection_api_does_not_return_or_persist_key():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        with patch("web.app.LLMClient.list_models", return_value=["deepseek-v4-pro"]):
            response = client.post("/api/llm/test-connection", json={
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com",
                "model_name": "deepseek-v4-pro",
                "api_key": "ds-secret",
            })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["model_available"] is True
    assert "ds-secret" not in response.get_data(as_text=True)


def test_deepseek_controls_are_available_in_both_model_entry_points():
    template = Path("web/templates/index.html").read_text(encoding="utf-8")
    script = Path("web/static/js/app.js").read_text(encoding="utf-8")
    css = Path("web/static/css/style.css").read_text(encoding="utf-8")
    assert template.count('<option value="deepseek">DeepSeek 官方 API</option>') == 2
    assert 'id="research-semantic-test-btn"' in template
    assert 'id="llm-test-btn"' in template
    assert 'id="llm-image-input"' in template
    assert 'id="llm-image-preview"' in template
    assert "deepseek-v4-pro" in template
    assert "deepseek-v4-flash" in template
    assert "deepseek-v4-flash-vision-exp" in template
    assert "testResearchSemanticConnection" in script
    assert "testLLMConnection" in script
    assert "llmImageAttachments" in script
    assert ".settings-panel .form-group label.toggle-label" in css
    assert "mergeRemoteModelPresetOptions" in script
    assert "/api/llm/test-connection" in script


def test_multimodal_messages_append_images_to_last_user_without_mutating_prompt():
    messages = AnalysisPromptBuilder.build_result_prompt({})
    original_text = messages[-1]["content"]
    image = {"name": "题图.png", "data_url": "data:image/png;base64,aGVsbG8="}

    multimodal = attach_multimodal_images(messages, [image])

    assert multimodal is not messages
    assert multimodal[-1]["content"][0] == {"type": "text", "text": original_text}
    assert multimodal[-1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": image["data_url"], "detail": "high"},
    }
    assert messages[-1]["content"] == original_text


def test_llm_image_normalization_enforces_mime_and_size_contract():
    normalized = _normalize_llm_images([{
        "name": "diagram.png",
        "data_url": "data:image/png;base64,aGVsbG8=",
    }])
    assert normalized[0]["mime_type"] == "image/png"
    assert normalized[0]["data_url"] == "data:image/png;base64,aGVsbG8="
    with pytest.raises(ValueError, match="PNG/JPEG/WEBP/GIF"):
        _normalize_llm_images(["data:image/svg+xml;base64,aGVsbG8="])
    with pytest.raises(ValueError, match="base64"):
        _normalize_llm_images(["data:image/png;base64,not-valid!"])


def test_llm_analyze_route_accepts_valid_image_and_returns_count():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        with client.session_transaction() as flask_session:
            sid = flask_session.setdefault("sid", "test-multimodal-session")
            user_sessions[sid] = {"df_info": {"shape": [1, 1]}}
        with patch("web.app.LLMAnalyzer.analyze", return_value="图像分析完成") as analyze:
            response = client.post("/api/llm/analyze", json={
                "analysis_type": "eda",
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com",
                "model_name": "deepseek-v4-flash-vision-exp",
                "api_key": "ds-secret",
                "images": [{"name": "题图.png", "data_url": "data:image/png;base64,aGVsbG8="}],
            })
        assert response.status_code == 200
        assert response.get_json()["image_count"] == 1
        analyze.assert_called_once()
        assert analyze.call_args.kwargs["images"][0]["mime_type"] == "image/png"


def test_llm_analyze_route_rejects_invalid_image_before_starting_thread():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post("/api/llm/analyze", json={
            "analysis_type": "eda",
            "images": [{"data_url": "data:image/svg+xml;base64,aGVsbG8="}],
        })
    assert response.status_code == 400
    assert "PNG/JPEG/WEBP/GIF" in response.get_json()["error"]
