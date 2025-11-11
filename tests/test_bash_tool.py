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
    status = None
    for _ in range(40):
        chunk = BashOutputTool().execute(bash_id=shell_id)
        assert "error" not in chunk
        collected += chunk.get("stdout", "")
        status = chunk.get("status")
        if status == "completed":
            break
        time.sleep(0.05)

    assert status == "completed"
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
        assert "error" not in candidate
        if candidate.get("stdout"):
            filtered = candidate
            break
        time.sleep(0.05)

    assert filtered is not None
    assert filtered["stdout"].strip() == "alpha"

    remaining = BashOutputTool().execute(bash_id=shell_id)
    assert "error" not in remaining
    assert remaining["stdout"] == ""
    assert remaining["status"] == "completed"


def test_bash_output_invalid_regex():
    start = BashTool().execute(
        command="python -c \"print('line')\"",
        run_in_background=True,
    )
    shell_id = start["data"]["shell_id"]

    time.sleep(0.05)
    result = BashOutputTool().execute(bash_id=shell_id, filter="[")
    assert "error" in result


def test_kill_bash_terminates_process():
    start = BashTool().execute(
        command="python -c \"import time; print('begin'); time.sleep(5)\"",
        run_in_background=True,
    )
    shell_id = start["data"]["shell_id"]

    time.sleep(0.2)
    kill_result = KillBashTool().execute(shell_id=shell_id)

    assert "error" not in kill_result
    assert kill_result["status"] == "terminated"
    assert "stdout" in kill_result
    assert shell_id not in BACKGROUND_SHELLS


def test_kill_bash_unknown_id():
    result = KillBashTool().execute(shell_id="missing")
    assert "error" in result
