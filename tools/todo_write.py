from __future__ import annotations

from typing import Any, Dict, List

from tools.base import BaseTool


_ALLOWED_STATUSES = {"pending", "in_progress", "completed"}


def _build_display_entries(todos: List[Dict[str, str]]) -> str:
    if not todos:
        return ""

    lines: List[str] = []
    for todo in todos:
        status = todo.get("status", "").lower()
        if status == "completed":
            marker = "[x]"
        elif status == "in_progress":
            marker = "[-]"
        else:
            marker = "[ ]"
        content = todo.get("content", "").strip()
        active_form = todo.get("activeForm", "").strip()
        suffix = f" ({active_form})" if active_form else ""
        line = f"- {marker} {content}{suffix}".rstrip()
        lines.append(line)

    return "\n".join(lines)


def _error_display(message: str) -> str:
    text = str(message or "").strip()
    return text or "Unknown error"


class TodoWriteTool(BaseTool):
    """Structured todo list tool for tracking coding tasks during a session."""

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def description(self) -> str:
        return '''Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Examples of When to Use the Todo List

<example>
User: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!
Assistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.
*Creates todo list with the following items:*
1. Creating dark mode toggle component in Settings page
2. Adding dark mode state management (context/store)
3. Implementing CSS-in-JS styles for dark theme
4. Updating existing components to support theme switching
5. Running tests and build process, addressing any failures or errors that occur
*Begins working on the first task*

<reasoning>
The assistant used the todo list because:
1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes
2. The user explicitly requested tests and build be run afterward
3. The assistant inferred that tests and build need to pass by adding "Ensure tests and build succeed" as the final task
</reasoning>
</example>

<example>
User: Help me rename the function getCwd to getCurrentWorkingDirectory across my project
Assistant: Let me first search through your codebase to find all occurrences of 'getCwd'.
*Uses grep or search tools to locate all instances of getCwd in the codebase*
Assistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.
*Creates todo list with specific items for each file that needs updating*

<reasoning>
The assistant used the todo list because:
1. First, the assistant searched to understand the scope of the task
2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps
3. The todo list helps ensure every instance is tracked and updated systematically
4. This approach prevents missing any occurrences and maintains code consistency
</reasoning>
</example>


<example>
User: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.
Assistant: I'll help implement these features. First, let's add all the features to the todo list.
*Creates a todo list breaking down each feature into specific tasks based on the project architecture*
Assistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.

<reasoning>
The assistant used the todo list because:
1. The user provided multiple complex features to implement in a comma separated list
2. The todo list helps organize these large features into manageable tasks
3. This approach allows for tracking progress across the entire implementation
</reasoning>
</example>

<example>
User: Can you help optimize my React application? It's rendering slowly and has performance issues.</user>
Assistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.
*Reviews component structure, render patterns, state management, and data fetching*
Assistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.
*Creates todo list with items like: 1) Implementing memoization for expensive calculations in ProductList, 2) Adding virtualization for long lists in Dashboard, 3) Optimizing image loading in Gallery component, 4) Fixing state update loops in ShoppingCart, 5) Reviewing bundle size and implementing code splitting*
Let's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>

<reasoning>
The assistant used the todo list because:
1. First, the assistant examined the codebase to identify specific performance issues
2. Based on this analysis, it identified multiple optimization opportunities
3. Performance optimization is a non-trivial task requiring multiple steps
4. The todo list helps methodically track improvements across different components
5. This systematic approach ensures all performance bottlenecks are addressed
</reasoning>
</example>

## Examples of When NOT to Use the Todo List

<example>
User: How do I print 'Hello World' in Python?
Assistant: In Python, you can print "Hello World" with this simple code:

python
print("Hello World")

This will output the text "Hello World" to the console when executed.</assistant>

<reasoning>
The assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.
</reasoning>
</example>

<example>
User: What does the git status command do?
Assistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.

<reasoning>
The assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.
</reasoning>
</example>

<example>
User: Can you add a comment to the calculateTotal function to explain what it does?
Assistant: Sure, let me add a comment to the calculateTotal function to explain what it does.
* Uses the Edit tool to add a comment to the calculateTotal function *

<reasoning>
The assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.
</reasoning>
</example>

<example>
User: Run npm install for me and tell me what happens.
Assistant: I'll run the npm install command for you.

*Executes: npm install*

The command completed successfully. Here's the output:
[Output of npm install command]

All dependencies have been installed according to your package.json file.

<reasoning>
The assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.
</reasoning>
</example>

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

   **IMPORTANT**: Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done (e.g., "Run tests", "Build the project")
   - activeForm: The present continuous form shown during execution (e.g., "Running tests", "Building the project")

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names
   - Always provide both forms:
     - content: "Fix authentication bug"
     - activeForm: "Fixing authentication bug"

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
'''
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["todos"],
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "content",
                            "status",
                            "activeForm"
                        ],
                        "properties": {
                            "status": {
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed"
                                ],
                                "type": "string"
                            },
                            "content": {
                                "type": "string",
                                "minLength": 1
                            },
                            "activeForm": {
                                "type": "string",
                                "minLength": 1
                            }
                        },
                    },
                    "description": "The updated todo list"
                }
            }
        }

    def execute(self, *, todos: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            normalized = self._validate_and_normalize(todos)
            formatted = self._format_summary(normalized)
            data = {
                "todos": normalized,
                "display": _build_display_entries(normalized),
            }
            return {
                "status": "success",
                "content": formatted,
                "data": data,
            }
        except Exception as exc:  # pragma: no cover - exercised via tests
            message = str(exc)
            return {
                "status": "error",
                "content": message,
                "data": {
                    "error": message,
                    "display": _error_display(message),
                },
            }

    def _validate_and_normalize(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        if not isinstance(todos, list):
            raise TypeError("todos must be a list of todo objects")
        if not todos:
            raise ValueError("todos list cannot be empty")

        normalized: List[Dict[str, str]] = []
        in_progress_count = 0
        status_counts = {key: 0 for key in _ALLOWED_STATUSES}
        for index, raw in enumerate(todos):
            if not isinstance(raw, dict):
                raise TypeError(f"todo entry at index {index} must be an object")
            try:
                status = str(raw["status"])  # type: ignore[index]
                content = str(raw["content"])  # type: ignore[index]
                active_form = str(raw["activeForm"])  # type: ignore[index]
            except KeyError as missing:
                raise KeyError(f"todo entry at index {index} is missing required field '{missing.args[0]}'")

            status = status.strip()
            content = content.strip()
            active_form = active_form.strip()

            if status not in _ALLOWED_STATUSES:
                raise ValueError(
                    f"todo entry at index {index} has invalid status '{status}'. "
                    f"Allowed values: {sorted(_ALLOWED_STATUSES)}"
                )
            if not content:
                raise ValueError(f"todo entry at index {index} must have non-empty content")
            if not active_form:
                raise ValueError(f"todo entry at index {index} must have non-empty activeForm")

            if status == "in_progress":
                in_progress_count += 1
            status_counts[status] += 1

            normalized.append({
                "status": status,
                "content": content,
                "activeForm": active_form,
            })

        if in_progress_count == 1:
            return normalized

        completed_only = (
            in_progress_count == 0
            and status_counts["pending"] == 0
            and status_counts["completed"] == len(normalized)
        )

        if completed_only:
            return normalized

        raise ValueError(
            "Exactly one todo must have status 'in_progress'. Found "
            f"{in_progress_count} entries marked in_progress."
        )

    @staticmethod
    def _format_summary(todos: List[Dict[str, str]]) -> str:
        buckets: Dict[str, List[Dict[str, str]]] = {status: [] for status in _ALLOWED_STATUSES}
        for todo in todos:
            buckets[todo["status"]].append(todo)

        lines: List[str] = []
        status_order = ["in_progress", "pending", "completed"]
        for status in status_order:
            items = buckets[status]
            if not items:
                continue
            header = status.replace("_", " ").title()
            lines.append(f"{header} ({len(items)}):")
            for todo in items:
                lines.append(f"- {todo['content']} ({todo['activeForm']})")
        return "\n".join(lines)
