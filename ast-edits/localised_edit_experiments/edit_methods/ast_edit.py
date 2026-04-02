"""AST-aware JSON edit operations method."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from localised_edit_experiments.data_models import ApplyError, ParseError


AstNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

VALID_OPERATIONS = {
    "replace_function_body",
    "replace_function",
    "add_before",
    "add_after",
    "delete",
    "add_import",
    "replace_imports",
    "add_method",
    "replace_global",
}


@dataclass
class AstEditOp:
    """A single AST edit operation."""

    operation: str
    target: str | None = None
    content: str | None = None
    position: str | None = None


class AstEditMethod:
    """Edit method using AST-aware JSON edit operations."""

    def system_prompt(self) -> str:
        """Return system prompt for AST edit operations."""
        return (
            "You are a code editing assistant. When asked to edit code, "
            "output a JSON array of edit operations in a ```json fenced block.\n\n"
            "Supported operations:\n"
            '- "replace_function_body": Replace only the body of a function, '
            "keeping its signature. The content MUST include proper indentation "
            "(e.g., 4 spaces for top-level functions).\n"
            '- "replace_function": Replace an entire function definition '
            "including its signature. Content must be the complete function.\n"
            '- "add_before": Add code before a function/class.\n'
            '- "add_after": Add code after a function/class.\n'
            '- "delete": Remove a function/class entirely.\n'
            '- "add_import": Add an import statement (no target needed).\n'
            '- "replace_imports": Replace the entire import block at the '
            "top of the file. No target needed, content is the full "
            "import section.\n"
            '- "add_method": Add a method to a class. Target is the '
            "class name, content is the method code (with proper "
            "indentation). The method is added at the end of the class.\n"
            '- "replace_global": Replace a module-level (top-level, '
            "unindented) variable assignment. Target is the variable "
            "name. This does NOT work for class or instance attributes "
            "— to change those, use replace_function_body on the "
            "__init__ method instead.\n\n"
            "Each operation is an object with: operation, target (function/class "
            "name), content (the code).\n\n"
            "For methods inside classes, use dotted names like "
            '"ClassName.method_name" to target the correct method.\n\n'
            "Important notes:\n"
            "- For dataclasses, do NOT use replace_function_body — it "
            "will delete properties and methods. Use replace_function "
            "to replace the entire dataclass definition instead.\n"
            "- Do not emit duplicate operations on the same target.\n\n"
            "Example:\n"
            "```json\n"
            "[\n"
            '  {"operation": "replace_function_body", '
            '"target": "MyClass.calculate",\n'
            '   "content": "    return a * b"}\n'
            "]\n"
            "```\n\n"
            "Make ONLY the change requested. "
            "Do not include any explanation outside the JSON block."
        )

    def user_prompt(self, original_code: str, instruction: str) -> str:
        """Build user prompt with original code and instruction."""
        return (
            f"Edit the following code according to the instruction.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Original code:\n```python\n{original_code}\n```"
        )

    def parse(self, llm_output: str) -> list[AstEditOp]:
        """Extract JSON array of edit ops from ```json fenced block."""
        pattern = r"```json\s*\n(.*?)```"
        matches = list(re.finditer(pattern, llm_output, re.DOTALL))
        if not matches:
            raise ParseError("No ```json fenced code block found in output")

        # Use the last JSON block — the LLM may self-correct
        match = matches[-1]
        try:
            ops_data = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e

        if not isinstance(ops_data, list):
            raise ParseError("Expected a JSON array of operations")

        ops: list[AstEditOp] = []
        for item in ops_data:
            if not isinstance(item, dict) or "operation" not in item:
                raise ParseError(f"Invalid operation: {item}")
            if item["operation"] not in VALID_OPERATIONS:
                raise ParseError(
                    f"Unknown operation: {item['operation']}. Valid: {VALID_OPERATIONS}"
                )
            ops.append(
                AstEditOp(
                    operation=item["operation"],
                    target=item.get("target"),
                    content=item.get("content"),
                    position=item.get("position"),
                )
            )
        return ops

    def apply(self, original_code: str, parsed_edit: list[AstEditOp]) -> str:
        """Apply AST edit operations to original code."""
        lines = original_code.splitlines()

        # Process operations in reverse line order to preserve positions
        ops_with_pos = self._resolve_positions(original_code, parsed_edit)
        ops_with_pos.sort(key=lambda x: x[0], reverse=True)

        for _pos, op in ops_with_pos:
            lines = self._apply_single_op(lines, original_code, op)

        return "\n".join(lines)

    def _resolve_positions(
        self, code: str, ops: list[AstEditOp]
    ) -> list[tuple[int, AstEditOp]]:
        """Resolve each op to a line position for ordering."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ApplyError(f"Cannot parse original code: {e}") from e

        result: list[tuple[int, AstEditOp]] = []
        for op in ops:
            if op.operation in ("add_import", "replace_imports"):
                result.append((0, op))
                continue

            if op.operation == "replace_global":
                pos = self._find_global_assignment(
                    code, op.target or ""
                )
                if pos is None:
                    raise ApplyError(
                        f"Global assignment not found: {op.target!r}"
                    )
                result.append((pos, op))
                continue

            if op.operation == "add_method":
                node = self._find_node(tree, op.target or "")
                if node is None:
                    raise ApplyError(f"Target not found: {op.target!r}")
                end = node.end_lineno or node.lineno
                result.append((end, op))
                continue

            node = self._find_node(tree, op.target or "")
            if node is None:
                raise ApplyError(f"Target not found: {op.target!r}")
            result.append((node.lineno, op))

        return result

    _NO_TARGET_OPS: frozenset[str] = frozenset(
        {"add_import", "replace_imports", "replace_global"}
    )

    def _apply_single_op(
        self, lines: list[str], original_code: str, op: AstEditOp
    ) -> list[str]:
        """Apply a single edit operation."""
        if op.operation in self._NO_TARGET_OPS:
            return self._apply_targetless_op(lines, op)

        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError as e:
            raise ApplyError(f"Cannot parse code during edit: {e}") from e

        node = self._find_node(tree, op.target or "")
        if node is None:
            raise ApplyError(f"Target not found: {op.target!r}")
        return self._apply_node_op(lines, node, op)

    def _apply_targetless_op(
        self, lines: list[str], op: AstEditOp
    ) -> list[str]:
        """Apply operations that don't need an AST node target."""
        content = op.content or ""
        if op.operation == "add_import":
            return self._add_import(lines, content)
        if op.operation == "replace_imports":
            return self._replace_imports(lines, content)
        if op.operation == "replace_global":
            return self._replace_global(lines, op.target or "", content)
        raise ApplyError(f"Unhandled targetless operation: {op.operation}")

    def _apply_node_op(
        self, lines: list[str], node: AstNode, op: AstEditOp
    ) -> list[str]:
        """Apply operations that target an AST node."""
        content = op.content or ""
        if op.operation == "replace_function_body":
            return self._replace_function_body(lines, node, content)
        if op.operation == "replace_function":
            return self._replace_function(lines, node, content)
        if op.operation == "add_before":
            return self._add_before(lines, node, content)
        if op.operation == "add_after":
            return self._add_after(lines, node, content)
        if op.operation == "delete":
            return self._delete_node(lines, node)
        if op.operation == "add_method":
            return self._add_method(lines, node, content)
        raise ApplyError(f"Unhandled operation: {op.operation}")

    def _find_node(self, tree: ast.Module, name: str) -> AstNode | None:
        """Find a function or class by name in the AST.

        Supports dotted names like 'ClassName.method_name' to target
        a method within a specific class.
        """
        if "." in name:
            class_name, method_name = name.split(".", 1)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in ast.walk(node):
                        if isinstance(
                            child, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and child.name == method_name:
                            return child
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name:
                    return node
            elif isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def _add_import(self, lines: list[str], content: str) -> list[str]:
        """Add import statement at the top of the file."""
        # Find last import line
        last_import = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                last_import = i

        insert_pos = last_import + 1 if last_import >= 0 else 0
        new_lines = content.splitlines()
        return lines[:insert_pos] + new_lines + lines[insert_pos:]

    def _replace_function_body(
        self,
        lines: list[str],
        node: AstNode,
        content: str,
    ) -> list[str]:
        """Replace the body of a function, keeping its signature and docstring."""
        end_line = node.end_lineno  # 1-indexed
        if end_line is None:
            raise ApplyError("Cannot determine function end line")

        # Preserve the docstring if present: skip body[0] when it's a
        # string-constant expression (the standard docstring pattern).
        body_start_idx = 0
        first_stmt = node.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
            and len(node.body) > 1
        ):
            body_start_idx = 1

        body_start = node.body[body_start_idx].lineno - 1  # 0-indexed

        # Detect indentation from first body line
        indent = self._get_indent(lines[body_start])
        content_lines = content.splitlines()

        # Check if content is already indented to the correct level
        if content_lines and self._get_indent(content_lines[0]) == indent:
            new_body_lines = content_lines
        else:
            new_body_lines = self._reindent(content_lines, indent)

        return lines[:body_start] + new_body_lines + lines[end_line:]

    def _replace_function(
        self,
        lines: list[str],
        node: AstNode,
        content: str,
    ) -> list[str]:
        """Replace entire function/class definition."""
        start = node.lineno - 1
        # Check for decorators
        if node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        end_line = node.end_lineno
        if end_line is None:
            raise ApplyError("Cannot determine function end line")

        # Detect the indentation of the original definition
        original_indent = self._get_indent(lines[node.lineno - 1])
        new_lines = content.splitlines()
        if new_lines and self._get_indent(new_lines[0]) != original_indent:
            new_lines = self._reindent(new_lines, original_indent)

        return lines[:start] + new_lines + lines[end_line:]

    def _add_before(
        self,
        lines: list[str],
        node: AstNode,
        content: str,
    ) -> list[str]:
        """Add code before a function/class."""
        start = node.lineno - 1
        if node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        new_lines = content.splitlines()
        return lines[:start] + new_lines + [""] + lines[start:]

    def _add_after(
        self,
        lines: list[str],
        node: AstNode,
        content: str,
    ) -> list[str]:
        """Add code after a function/class."""
        end_line = node.end_lineno
        if end_line is None:
            raise ApplyError("Cannot determine function end line")
        new_lines = content.splitlines()
        return [*lines[:end_line], "", *new_lines, *lines[end_line:]]

    def _delete_node(
        self,
        lines: list[str],
        node: AstNode,
    ) -> list[str]:
        """Delete a function/class definition."""
        start = node.lineno - 1
        if node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        end_line = node.end_lineno
        if end_line is None:
            raise ApplyError("Cannot determine function end line")
        return lines[:start] + lines[end_line:]

    def _replace_imports(self, lines: list[str], content: str) -> list[str]:
        """Replace the entire import block at the top of the file."""
        first_import = -1
        last_import = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                if first_import == -1:
                    first_import = i
                last_import = i

        if first_import == -1:
            # No imports found, insert at top
            new_lines = content.splitlines()
            return [*new_lines, "", *lines]

        new_lines = content.splitlines()
        return [*lines[:first_import], *new_lines, *lines[last_import + 1 :]]

    def _add_method(
        self,
        lines: list[str],
        node: AstNode,
        content: str,
    ) -> list[str]:
        """Add a method at the end of a class body."""
        if not isinstance(node, ast.ClassDef):
            raise ApplyError(
                f"add_method target must be a class, got {type(node).__name__}"
            )
        end_line = node.end_lineno
        if end_line is None:
            raise ApplyError("Cannot determine class end line")

        # Detect class body indentation
        indent = self._get_indent(lines[node.body[0].lineno - 1])
        content_lines = content.splitlines()

        # Check if content is already indented correctly
        if content_lines and self._get_indent(content_lines[0]) == indent:
            new_method_lines = content_lines
        else:
            new_method_lines = [indent + ln for ln in content_lines]

        # Insert before end of class with a blank line separator
        return [*lines[:end_line], "", *new_method_lines, *lines[end_line:]]

    def _replace_global(
        self, lines: list[str], target: str, content: str
    ) -> list[str]:
        """Replace a module-level variable assignment."""
        prefixes = (f"{target} =", f"{target}:")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(prefixes) and line[0] not in (" ", "\t"):
                new_lines = content.splitlines()
                return [*lines[:i], *new_lines, *lines[i + 1 :]]
        raise ApplyError(f"Global assignment not found: {target!r}")

    def _find_global_assignment(self, code: str, target: str) -> int | None:
        """Find the line number of a module-level assignment."""
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            prefixes = (f"{target} =", f"{target}:")
            if stripped.startswith(prefixes) and line[0] not in (" ", "\t"):
                return i
        return None

    def _reindent(self, content_lines: list[str], target_indent: str) -> list[str]:
        """Re-indent content lines to match a target indentation level."""
        if not content_lines:
            return content_lines
        old_indent = self._get_indent(content_lines[0])
        result = []
        for line in content_lines:
            if not line.strip():
                result.append(line)
            elif line.startswith(old_indent):
                result.append(target_indent + line[len(old_indent):])
            else:
                result.append(target_indent + line.lstrip())
        return result

    def _get_indent(self, line: str) -> str:
        """Extract leading whitespace from a line."""
        return line[: len(line) - len(line.lstrip())]
