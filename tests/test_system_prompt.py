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
