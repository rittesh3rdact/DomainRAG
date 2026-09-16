from app.rag.agent import NO_CONTEXT_MESSAGE, build_system_instruction


def test_role_description_is_embedded_verbatim():
    role = "an assistant covering finance, law, and sports"
    instruction = build_system_instruction(role)
    assert role in instruction


def test_fallback_message_is_embedded():
    instruction = build_system_instruction("a domain assistant")
    assert NO_CONTEXT_MESSAGE in instruction


def test_instructs_against_markdown_and_bracket_citations():
    instruction = build_system_instruction("a domain assistant")
    assert "no markdown" in instruction.lower()
    assert "[source:" not in instruction
