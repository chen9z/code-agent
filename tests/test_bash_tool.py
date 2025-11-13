import time

import pytest

from tools.bash import BACKGROUND_SHELLS, BashOutputTool, BashTool, KillBashTool


@pytest.fixture(autouse=True)
def clear_background_shells():
    try:
        yield
    finally:
        # Ensure no stray processes remain after tests.
        for shell in list(BACKGROUND_SHELLS.values()):
            try:
                shell.process.terminate()
            except Exception:
                pass
            shell.close()
        BACKGROUND_SHELLS.clear()


def test_bash_sync_executes_command():
    result = BashTool().execute(command="echo hello")

    assert result["status"] == "success"
    assert result["content"] == "hello"
    assert result["data"]["exit_code"] == 0
    assert result["data"]["stdout"].strip() == "hello"
    assert result["data"]["timed_out"] is False


def test_bash_background_and_output():
    script = "python -c \"import time; print('start'); time.sleep(0.1); print('done')\""
    start = BashTool().execute(command=script, run_in_background=True)

    assert start["status"] == "success"
    shell_id = start["data"]["shell_id"]

    collected = ""
    shell_state = None
    for _ in range(40):
        chunk = BashOutputTool().execute(bash_id=shell_id)
        assert chunk["status"] == "success"
        collected += chunk["data"]["stdout"]
        shell_state = chunk["data"]["shell_status"]
        if shell_state == "completed":
            break
        time.sleep(0.05)

    assert shell_state == "completed"
    assert "start" in collected
    assert "done" in collected


def test_bash_output_filter_consumes_non_matches():
    start = BashTool().execute(
        command="python -c \"print('alpha'); print('beta')\"",
        run_in_background=True,
    )
    shell_id = start["data"]["shell_id"]

    filtered = None
    for _ in range(20):
        candidate = BashOutputTool().execute(bash_id=shell_id, filter="alpha")
        assert candidate["status"] == "success"
        if candidate["data"]["stdout"]:
            filtered = candidate
            break
        time.sleep(0.05)

    assert filtered is not None
    assert filtered["data"]["stdout"].strip() == "alpha"

    remaining = None
    for _ in range(20):
        probe = BashOutputTool().execute(bash_id=shell_id)
        assert probe["status"] == "success"
        if probe["data"]["shell_status"] == "completed" and probe["data"]["stdout"] == "":
            remaining = probe
            break
        time.sleep(0.05)

    assert remaining is not None
    assert remaining["data"]["stdout"] == ""
    assert remaining["data"]["shell_status"] == "completed"


def test_bash_output_invalid_regex():
    start = BashTool().execute(
        command="python -c \"print('line')\"",
        run_in_background=True,
    )
    shell_id = start["data"]["shell_id"]

    time.sleep(0.05)
    result = BashOutputTool().execute(bash_id=shell_id, filter="[")
    assert result["status"] == "error"


def test_kill_bash_terminates_process():
    start = BashTool().execute(
        command="python -c \"import time; print('begin'); time.sleep(5)\"",
        run_in_background=True,
    )
    shell_id = start["data"]["shell_id"]

    time.sleep(0.2)
    kill_result = KillBashTool().execute(shell_id=shell_id)

    assert kill_result["status"] == "success"
    assert kill_result["data"]["shell_status"] == "terminated"
    assert "stdout" in kill_result["data"]
    assert shell_id not in BACKGROUND_SHELLS


def test_kill_bash_unknown_id():
    result = KillBashTool().execute(shell_id="missing")
    assert result["status"] == "error"
