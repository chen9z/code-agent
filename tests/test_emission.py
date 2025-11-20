from ui.emission import EmitEvent, create_emit_event


def test_emit_event_preserves_string_representation():
    event = create_emit_event(
        "plan",
        "Run lint",
        payload={
            "step": 1,
            "display": "args: --check\nnote: auto",
        },
    )

    assert isinstance(event, EmitEvent)
    assert str(event).startswith("[plan] Run lint")
    assert event.kind == "plan"
    assert event.body == "Run lint"
    assert event.display[0][0] == "result"
    assert event.display[0][1] == "args: --check\nnote: auto"
    assert event.payload == {
        "step": 1,
        "display": "args: --check\nnote: auto",
    }
    assert str(event).startswith("[plan] Run lint")


def test_emit_event_supports_flag_metadata():
    event = create_emit_event(
        "tool",
        "echo",
        payload={"display": "preview truncated"},
    )

    assert "preview truncated" in str(event)
    assert event.display[0][0] == "result"
    assert event.display[0][1] == "preview truncated"
