"""
SWE-bench Purple Agent - Solves GitHub issues using LLM reasoning.

This agent:
1. Receives raw task data from Green Agent (issue description, repo info)
2. Builds LLM prompts with its own strategy
3. Explores codebase via bash commands
4. Generates patches to fix issues
5. Handles patch retry on failure
"""

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
import litellm
from dotenv import load_dotenv
import json

load_dotenv()


# Response format keys
RESPONSE_KEY = "action"
CONTENT_KEY = "content"

# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing bugs in a repository.

You will receive a GitHub issue description and must fix it by exploring the codebase and/or submitting a patch.

## Response Format

You MUST respond with a single JSON object in one of these formats:

1. To explore the codebase or fetch context (run a bash command):
   - Format: {"action": "bash", "content": "<shell command>"}
   - Example: {"action": "bash", "content": "ls sklearn/metrics"}
   - Outputs from the command will be returned to you.
   - Only read-only commands are allowed; do not modify files yet.

2. To submit your fix (unified diff format):
   - Format: {"action": "patch", "content": "<unified diff>"}
   - Example: {"action": "patch", "content": "```
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 context line
-old line to remove
+new line to add
 context line
```"}
   - You may generate the patch as a minimal diff; it will be executed for you and output returned to you.

## Important Rules

- Respond with ONLY the JSON object. No explanations, no markdown, no extra text.
- Use bash commands to explore: ls, cat, grep, find, git log, git diff, etc.
- The codebase is READ-ONLY. You cannot modify files with bash commands.
- When ready, submit a patch in unified diff format (like `git diff` output).
- If your patch fails, you'll receive the error. Analyze it and try again.

## Execution Output Format

After every bash command that you submit, you'll receive its execution output as follows:
{
  "cwd": "/workspace/repo/current/directory",
  "stdout": "command output...",
  "stderr": "any errors..."
}

Use the 'cwd' key to track your current location. You can use `cd` to navigate within the repo but never outside it.

Here are the Issue Details:

"""


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.messages = []
        self.task_data = None

    def _parse_task_data(self, input_text: str) -> dict | None:
        """Parse raw task data from Green Agent."""
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            return None

    def _parse_bash_result(self, input_text: str) -> dict | None:
        """Parse bash result from Green Agent."""
        try:
            data = json.loads(input_text)
            if "cwd" in data and ("stdout" in data or "stderr" in data):
                return data
            return None
        except json.JSONDecodeError:
            return None

    def _parse_patch_failure(self, input_text: str) -> dict | None:
        """Parse patch failure feedback from Green Agent."""
        try:
            data = json.loads(input_text)
            if data.get("patch_failed"):
                return data
            return None
        except json.JSONDecodeError:
            return None

    def _parse_error_feedback(self, input_text: str) -> dict | None:
        """Parse error feedback from Green Agent."""
        try:
            data = json.loads(input_text)
            if "error" in data:
                return data
            return None
        except json.JSONDecodeError:
            return None

    def _build_initial_prompt(self, task_data: dict) -> str:
        """Build the initial user prompt from task data."""
        problem_statement = task_data.get("problem_statement", "No description provided")
        hints = task_data.get("hints_text", "")
        repo = task_data.get("repo", "unknown")
        python_version = task_data.get("python_version", "3.9")
        instance_id = task_data.get("instance_id", "unknown")
        fail_to_pass = task_data.get("fail_to_pass", [])

        prompt = f"""Current Working Directory (cwd): {repo}
Python Version: {python_version}

## Issue Description

{problem_statement}

"""
        if hints:
            prompt += f"""## Additional Context (from issue discussion)

{hints}

"""
        if fail_to_pass:
            prompt += f"""## Tests to Fix

The following tests are currently failing and should pass after your fix:
{chr(10).join(f'- {test}' for test in fail_to_pass[:5])}
{'...' if len(fail_to_pass) > 5 else ''}

"""

        return prompt

    def _format_bash_result_for_llm(self, result: dict) -> str:
        """Format bash result for the LLM."""
        cwd = result.get("cwd", "/workspace/repo")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        parts = [f"Current directory: {cwd}"]

        if stdout:
            # Truncate very long outputs
            if len(stdout) > 8000:
                stdout = stdout[:8000] + "\n... [output truncated]"
            parts.append(f"Output:\n{stdout}")

        if stderr:
            parts.append(f"Errors:\n{stderr}")

        if not stdout and not stderr:
            parts.append("(no output)")

        return "\n\n".join(parts)

    def _format_patch_failure_for_llm(self, result: dict) -> str:
        """Format patch failure for the LLM to retry."""
        stderr = result.get("stderr", "Unknown error")
        cwd = result.get("cwd", "/workspace/repo")

        return f"""Patch application FAILED.

Current directory: {cwd}

Error details:
{stderr}

Please analyze the error and submit a corrected patch. Common issues:
- Incorrect file path in the diff header
- Wrong line numbers (the code may have changed)
- Missing context lines
- Whitespace issues

Try using `cat` to view the exact current content of the file, then create a new patch."""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Handle messages from Green Agent.

        Messages can be:
        1. Initial task data (JSON with problem_statement, repo, etc.)
        2. Bash command result (JSON with cwd, stdout, stderr)
        3. Patch failure feedback (JSON with patch_failed, stderr)
        4. Error feedback (JSON with error message)
        """
        input_text = get_message_text(message)
        print(f"input > {input_text[:200]}...")

        await updater.update_status(
            TaskState.working, new_agent_text_message("Processing...")
        )

        # Determine message type and format appropriately
        user_content = None

        # Check if this is initial task data
        task_data = self._parse_task_data(input_text)
        if task_data and "problem_statement" in task_data:
            # Initial task - build prompt and start fresh conversation
            self.task_data = task_data
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            user_content = self._build_initial_prompt(task_data)

        # Check if this is a bash result
        elif bash_result := self._parse_bash_result(input_text):
            user_content = self._format_bash_result_for_llm(bash_result)

        # Check if this is a patch failure
        elif patch_failure := self._parse_patch_failure(input_text):
            user_content = self._format_patch_failure_for_llm(patch_failure)

        # Check if this is an error feedback
        elif error_feedback := self._parse_error_feedback(input_text):
            user_content = f"Error: {error_feedback.get('message', error_feedback.get('error', 'Unknown error'))}\n\nPlease respond with valid JSON: {{\"action\": \"bash\"|\"patch\", \"content\": \"...\"}}"

        # Fallback - treat as raw text
        else:
            user_content = input_text

        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_content})

        # Call LLM
        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        try:
            completion = litellm.completion(
                model="openrouter/tngtech/deepseek-r1t-chimera:free",
                messages=self.messages,
                response_format={"type": "json_object"}
            )
            response = completion.choices[0].message.content
        except Exception as e:
            # Handle LLM errors
            await updater.add_artifact(
                name="error",
                parts=[Part(root=TextPart(text=f"LLM error: {str(e)}"))],
            )
            return

        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": response})
        print(f"response > {response[:200]}...")

        # Parse and return response
        try:
            response_json = json.loads(response)
            action = response_json.get(RESPONSE_KEY, "unknown")
            content = response_json.get(CONTENT_KEY, "")

            await updater.add_artifact(
                name=action,
                parts=[Part(root=TextPart(text=content))],
            )
        except json.JSONDecodeError:
            # If LLM didn't return valid JSON, try to extract it
            await updater.add_artifact(
                name="error",
                parts=[Part(root=TextPart(text=f"Invalid JSON response: {response[:500]}"))],
            )