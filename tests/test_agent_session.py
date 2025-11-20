from agent.session import CodeAgentSession


def test_parse_tool_response_reports_json_errors():
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "abc",
                            "function": {
                                "name": "read",
                                "arguments": "{invalid json}",
                            },
                        }
                    ],
                }
            }
        ]
    }

    parsed = CodeAgentSession._parse_tool_response(response)

    assert parsed["tool_calls"] == []
    errors = parsed["tool_call_errors"]
    assert errors, "Expected tool call errors to be reported"
    assert errors[0]["id"] == "abc"
    assert errors[0]["name"] == "read"
    assert "Failed to parse" in errors[0]["message"]
