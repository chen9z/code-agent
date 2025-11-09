from core.emission import EmitEvent, create_emit_event


def test_emit_event_preserves_string_representation():
    event = create_emit_event(
        "plan",
        "Run lint",
        payload={
            "step": 1,
            "display": [("args", "--check"), ("note", "auto")],
        },
    )

    assert isinstance(event, EmitEvent)
    assert isinstance(event, str)
    assert event.kind == "plan"
    assert event.body == "Run lint"
    assert ("args", "--check") in event.display
    assert ("note", "auto") in event.display
    assert event.payload == {
        "step": 1,
        "display": [("args", "--check"), ("note", "auto")],
    }
    assert str(event).startswith("[plan] Run lint")


def test_emit_event_supports_flag_metadata():
    event = create_emit_event(
        "tool",
        "echo",
        payload={"display": [("preview truncated", None)]},
    )

    assert "preview truncated" in str(event)
    assert event.display[0][0] == "preview truncated"
    assert event.display[0][1] is None
