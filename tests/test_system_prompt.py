from core.prompt import compose_system_prompt
from code_agent import build_code_agent_system_prompt, SECURITY_SYSTEM_PROMPT


def test_compose_system_prompt_with_sections_and_env():
    prompt = compose_system_prompt(
        "Base",
        extra_sections=["Second", "Third"],
        environment={"cwd": "/workspace", "date": "2025-09-30"},
    )

    parts = prompt.split("\n\n")
    assert parts[0] == "Base"
    assert parts[1] == "Second"
    assert parts[2] == "Third"
    assert parts[3].startswith("<env>")
    assert "cwd: /workspace" in parts[3]


def test_build_code_agent_system_prompt_includes_security_prompt():
    prompt = build_code_agent_system_prompt(
        base_prompt="Base",
        extra_sections=["Project Details"],
        environment={"platform": "darwin"},
    )

    assert "Base" in prompt
    assert SECURITY_SYSTEM_PROMPT.strip() in prompt
    assert "Project Details" in prompt
    assert "platform: darwin" in prompt
