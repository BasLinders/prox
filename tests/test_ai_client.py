import json

from utility import ai_client


def test_parse_conclusion_reads_the_requested_json_shape():
    text = json.dumps({
        "summary": "First paragraph.\n\nSecond paragraph.",
        "key_findings": ["Finding one.", "  ", "Finding two."],
        "next_steps": ["Do this."],
    })
    parsed = ai_client.parse_conclusion(text)
    assert parsed == {
        "summary": "First paragraph.\n\nSecond paragraph.",
        "key_findings": ["Finding one.", "Finding two."],
        "next_steps": ["Do this."],
    }


def test_parse_conclusion_accepts_a_fenced_json_block():
    text = '```json\n{"summary": "S", "key_findings": [], "next_steps": ["N"]}\n```'
    assert ai_client.parse_conclusion(text)["next_steps"] == ["N"]


def test_parse_conclusion_falls_back_to_plain_text():
    parsed = ai_client.parse_conclusion("Not JSON at all.")
    assert parsed == {"summary": "Not JSON at all.", "key_findings": [], "next_steps": []}


def test_parse_conclusion_falls_back_on_json_of_another_shape():
    parsed = ai_client.parse_conclusion('["a", "b"]')
    assert parsed["summary"] == '["a", "b"]'
    assert parsed["next_steps"] == []


def test_prompt_template_formats_for_every_language():
    for name in ai_client.LANGUAGES.values():
        prompt = ai_client._PROMPT_TEMPLATE.format(language=name, language_upper=name.upper())
        assert name.upper() in prompt
        assert prompt.rstrip().endswith("Data:")


def test_generate_conclusion_without_a_key_fails_cleanly(monkeypatch):
    monkeypatch.setattr(ai_client, "get_api_key", lambda: "")
    result = ai_client.generate_conclusion({"log_summary": {}})
    assert result["ok"] is False
    assert result["error_kind"] == "not_configured"
    assert result["models_tried"] == []
