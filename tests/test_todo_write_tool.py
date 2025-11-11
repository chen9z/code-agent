from tools.todo_write import TodoWriteTool


def test_todo_write_formats_summary():
    tool = TodoWriteTool()
    todos = [
        {"content": "Review code", "status": "in_progress", "activeForm": "Reviewing code"},
        {"content": "Update registry", "status": "pending", "activeForm": "Updating registry"},
        {"content": "Write tests", "status": "completed", "activeForm": "Writing tests"},
    ]

    result = tool.execute(todos=todos)

    assert "error" not in result
    assert "counts" not in result
    assert result["todos"] == todos

    summary_lines = result["result"].splitlines()
    assert summary_lines[0] == "In Progress (1):"
    assert summary_lines[1] == "- Review code (Reviewing code)"
    assert "Pending" in summary_lines[2]
    assert summary_lines[-2] == "Completed (1):"
    assert summary_lines[-1] == "- Write tests (Writing tests)"


def test_todo_write_requires_single_in_progress():
    tool = TodoWriteTool()
    todos = [
        {"content": "Task A", "status": "pending", "activeForm": "Working on Task A"},
        {"content": "Task B", "status": "pending", "activeForm": "Working on Task B"},
    ]

    result = tool.execute(todos=todos)

    assert "error" in result
    assert "in_progress" in result["error"]


def test_todo_write_validates_required_fields():
    tool = TodoWriteTool()

    result = tool.execute(todos=[{"content": "Task", "status": "in_progress"}])

    assert "error" in result
    assert "activeForm" in result["error"]


def test_todo_write_accepts_all_completed_snapshot():
    tool = TodoWriteTool()
    todos = [
        {"content": "Wrap up work", "status": "completed", "activeForm": "Wrapping up"},
        {"content": "Share summary", "status": "completed", "activeForm": "Summarizing"},
    ]

    result = tool.execute(todos=todos)

    assert "error" not in result
    assert "counts" not in result
    assert result["todos"] == todos


def test_todo_write_rejects_zero_in_progress_with_pending_items():
    tool = TodoWriteTool()
    todos = [
        {"content": "Plan work", "status": "pending", "activeForm": "Planning"},
        {"content": "Consider ideas", "status": "completed", "activeForm": "Considering"},
    ]

    result = tool.execute(todos=todos)

    assert "error" in result
    assert "Exactly one todo" in result["error"]
