"""Benchmark tasks for localised code editing experiments.

Each task uses a realistic Python file (100-1500 lines) with a small,
localised edit. This tests whether edit methods can handle large files
where most of the code should remain unchanged.
"""

from __future__ import annotations

from localised_edit_experiments.data_models import (
    EditDifficulty,
    EditTask,
    EditType,
)
from localised_edit_experiments.task_fixtures import (
    build_giant_module,
    build_huge_module,
    build_large_module,
    build_massive_module,
    build_medium_module,
    build_small_module,
    build_xlarge_module,
)


def _make_task(
    task_id: str,
    description: str,
    base_code: str,
    old_fragment: str,
    new_fragment: str,
    difficulty: EditDifficulty,
    edit_type: EditType,
    test_code: str = "",
) -> EditTask:
    """Create a task by replacing a fragment in base code."""
    if old_fragment not in base_code:
        raise ValueError(
            f"Task {task_id}: old_fragment not found in base code: "
            f"{old_fragment[:80]!r}"
        )
    if base_code.count(old_fragment) > 1:
        raise ValueError(
            f"Task {task_id}: old_fragment found multiple times in base code"
        )
    expected_code = base_code.replace(old_fragment, new_fragment, 1)
    return EditTask(
        task_id=task_id,
        description=description,
        original_code=base_code,
        expected_code=expected_code,
        difficulty=difficulty,
        edit_type=edit_type,
        test_code=test_code,
    )


_NO_TAMPER_INSTRUCTION = (
    "\n\nCRITICAL: You MUST leave all code outside the requested "
    "changes COMPLETELY UNCHANGED. Do not reformat, rename, "
    "reorder, restyle, or restructure ANY code that is not "
    "directly required by this task. Do not change whitespace, "
    "quotes, string formatting, variable naming conventions, or "
    "function signatures in unrelated code. Every line you did "
    "not need to touch must be identical to the original."
)



def _build_tasks() -> list[EditTask]:
    """Build all benchmark tasks."""
    small = build_small_module()
    medium = build_medium_module()
    large = build_large_module()
    xlarge = build_xlarge_module()
    huge = build_huge_module()

    tasks: list[EditTask] = []

    # === EASY (5 tasks) — small/medium files, single-line or trivial edits ===

    # 1. Fix off-by-one in chunk_list (small module, ~100 lines)
    tasks.append(
        _make_task(
            task_id="easy_off_by_one_01",
            description=(
                "Fix the off-by-one bug in chunk_list: when the list length "
                "is not evenly divisible by chunk_size, the last chunk is "
                "dropped. The range step should use chunk_size correctly. "
                "Change the slice to include the remainder."
            ),
            base_code=small.replace(
                "def chunk_list(items: list, chunk_size: int) -> list[list]:\n"
                '    """Split a list into chunks of specified size."""\n'
                "    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]",
                "def chunk_list(items: list, chunk_size: int) -> list[list]:\n"
                '    """Split a list into chunks of specified size."""\n'
                "    return [items[i : i + chunk_size] for i in range(0, len(items) - 1, chunk_size)]",
            ),
            old_fragment="range(0, len(items) - 1, chunk_size)",
            new_fragment="range(0, len(items), chunk_size)",
            difficulty=EditDifficulty.EASY,
            edit_type=EditType.BUG_FIX,
            test_code=(
                "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
                "assert chunk_list([1, 2, 3], 2) == [[1, 2], [3]]\n"
                "assert chunk_list([], 3) == []\n"
                "assert chunk_list([1], 1) == [[1]]\n"
            ),
        )
    )

    # 2. Fix wrong comparison in validate_email (small module)
    tasks.append(
        _make_task(
            task_id="easy_wrong_operator_02",
            description=(
                "Fix the email validation: the check 'if \"@\" not in email' "
                "should return False, but currently returns True."
            ),
            base_code=small.replace(
                "def validate_email(email: str) -> bool:\n"
                '    """Basic email validation."""\n'
                '    if "@" not in email:\n'
                "        return False",
                "def validate_email(email: str) -> bool:\n"
                '    """Basic email validation."""\n'
                '    if "@" not in email:\n'
                "        return True",
            ),
            old_fragment=('    if "@" not in email:\n        return True'),
            new_fragment=('    if "@" not in email:\n        return False'),
            difficulty=EditDifficulty.EASY,
            edit_type=EditType.BUG_FIX,
            test_code=(
                "assert validate_email('no-at-sign') is False\n"
                "assert validate_email('test@example.com') is True\n"
                "assert validate_email('') is False\n"
            ),
        )
    )

    # 3. Add missing return in compute_hash (medium module, ~230 lines)
    tasks.append(
        _make_task(
            task_id="easy_missing_return_03",
            description=(
                "The compute_hash function is missing the return statement. "
                "It computes the hash but doesn't return it. Add the return."
            ),
            base_code=medium.replace(
                "def compute_hash(data: str) -> str:\n"
                '    """Compute SHA-256 hash of string data."""\n'
                "    return hashlib.sha256(data.encode()).hexdigest()",
                "def compute_hash(data: str) -> str:\n"
                '    """Compute SHA-256 hash of string data."""\n'
                "    hashlib.sha256(data.encode()).hexdigest()",
            ),
            old_fragment="    hashlib.sha256(data.encode()).hexdigest()",
            new_fragment="    return hashlib.sha256(data.encode()).hexdigest()",
            difficulty=EditDifficulty.EASY,
            edit_type=EditType.BUG_FIX,
            test_code=(
                "result = compute_hash('test')\n"
                "assert isinstance(result, str)\n"
                "assert len(result) == 64\n"
                "assert result == compute_hash('test')\n"
            ),
        )
    )

    # 4. Add missing import (medium module — remove 'import time', ask to add it back)
    tasks.append(
        _make_task(
            task_id="easy_add_import_04",
            description=(
                "The module uses time.time() and time.sleep() but is missing "
                "'import time'. Add the missing import."
            ),
            base_code=medium.replace("import time\n", ""),
            old_fragment="import os\nfrom dataclasses",
            new_fragment="import os\nimport time\nfrom dataclasses",
            difficulty=EditDifficulty.EASY,
            edit_type=EditType.BUG_FIX,
            test_code=(
                "# LRUCache.put uses time.time(); will fail if import time is missing\n"
                "cache = LRUCache(max_size=2)\n"
                "cache.put('key', 'value')\n"
                "assert cache.get('key') == 'value'\n"
            ),
        )
    )

    # 5. Rename variable in truncate_string (small module)
    tasks.append(
        _make_task(
            task_id="easy_rename_var_05",
            description=(
                "Rename the parameter 'max_length' to 'limit' in the "
                "truncate_string function (both the parameter and all usages "
                "within the function body)."
            ),
            base_code=small,
            old_fragment=(
                "def truncate_string(text: str, max_length: int = 100) -> str:\n"
                '    """Truncate a string to max_length with ellipsis."""\n'
                "    if len(text) <= max_length:\n"
                "        return text\n"
                '    return text[: max_length - 3] + "..."'
            ),
            new_fragment=(
                "def truncate_string(text: str, limit: int = 100) -> str:\n"
                '    """Truncate a string to limit with ellipsis."""\n'
                "    if len(text) <= limit:\n"
                "        return text\n"
                '    return text[: limit - 3] + "..."'
            ),
            difficulty=EditDifficulty.EASY,
            edit_type=EditType.REFACTOR,
            test_code=(
                "import inspect as _inspect\n"
                "_sig = _inspect.signature(truncate_string)\n"
                "assert 'limit' in _sig.parameters, 'Missing limit parameter'\n"
                "assert 'max_length' not in _sig.parameters, 'Old param still exists'\n"
                "assert truncate_string('hello', limit=10) == 'hello'\n"
                "assert truncate_string('a' * 20, limit=10) == 'a' * 7 + '...'\n"
            ),
        )
    )

    # === MEDIUM (7 tasks) — medium/large files, multi-line edits ===

    # 6. Add exception handling in parse_key_value (large module, ~360 lines)
    tasks.append(
        _make_task(
            task_id="medium_exception_handling_06",
            description=(
                "Wrap the parse_key_value function body in a try/except "
                "that catches ValueError and returns a tuple of "
                "empty strings ('', '') instead of raising."
            ),
            base_code=large,
            old_fragment=(
                'def parse_key_value(line: str, delimiter: str = "=") -> tuple[str, str]:\n'
                '    """Parse a key=value line into a tuple."""\n'
                "    if delimiter not in line:\n"
                '        raise ValueError(f"Delimiter {delimiter!r} not found in line")\n'
                "    key, _, value = line.partition(delimiter)\n"
                "    return key.strip(), value.strip()"
            ),
            new_fragment=(
                'def parse_key_value(line: str, delimiter: str = "=") -> tuple[str, str]:\n'
                '    """Parse a key=value line into a tuple."""\n'
                "    try:\n"
                "        if delimiter not in line:\n"
                '            raise ValueError(f"Delimiter {delimiter!r} not found in line")\n'
                "        key, _, value = line.partition(delimiter)\n"
                "        return key.strip(), value.strip()\n"
                "    except ValueError:\n"
                '        return "", ""'
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "assert parse_key_value('key=value') == ('key', 'value')\n"
                "assert parse_key_value('key:value', ':') == ('key', 'value')\n"
                "result = parse_key_value('no-delimiter')\n"
                "assert result == ('', ''), f\"Expected ('', ''), got {result}\"\n"
            ),
        )
    )

    # 7. Add default parameter to LRUCache.__init__ (large module)
    tasks.append(
        _make_task(
            task_id="medium_default_param_07",
            description=(
                "Add a 'ttl_seconds' parameter (default 3600) to the "
                "LRUCache.__init__ method and store it as self._ttl."
            ),
            base_code=large,
            old_fragment=(
                "    def __init__(self, max_size: int = 128) -> None:\n"
                "        self._max_size = max_size\n"
                "        self._cache: dict[str, tuple[Any, float]] = {}"
            ),
            new_fragment=(
                "    def __init__(self, max_size: int = 128, ttl_seconds: int = 3600) -> None:\n"
                "        self._max_size = max_size\n"
                "        self._ttl = ttl_seconds\n"
                "        self._cache: dict[str, tuple[Any, float]] = {}"
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "cache = LRUCache()\n"
                "assert hasattr(cache, '_ttl'), 'LRUCache missing _ttl attribute'\n"
                "assert cache._ttl == 3600, f'Default _ttl should be 3600, got {cache._ttl}'\n"
                "cache2 = LRUCache(ttl_seconds=120)\n"
                "assert cache2._ttl == 120\n"
            ),
        )
    )

    # 8. Extract helper from process_record error handling (large module)
    tasks.append(
        _make_task(
            task_id="medium_extract_helper_08",
            description=(
                "In the DataProcessor.process_record method, extract the "
                "error handling logic (the except block that checks "
                "_error_handlers) into a private method called "
                "_handle_step_error that takes (current_data, exc, result) "
                "and returns (data, should_continue) tuple."
            ),
            base_code=large,
            old_fragment=(
                "            except Exception as exc:\n"
                "                handler = self._error_handlers.get(type(exc))\n"
                "                if handler:\n"
                "                    current_data = handler(current_data, exc)\n"
                "                else:\n"
                "                    result.status = TaskStatus.FAILED\n"
                "                    result.error_message = str(exc)\n"
                '                    self._metrics["failed"] += 1\n'
                "                    return result"
            ),
            new_fragment=(
                "            except Exception as exc:\n"
                "                current_data, should_continue = self._handle_step_error(\n"
                "                    current_data, exc, result\n"
                "                )\n"
                "                if not should_continue:\n"
                "                    return result"
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.REFACTOR,
            test_code=(
                "_config = ServiceConfig()\n"
                "\n"
                "# Happy path still works\n"
                "def _step(data):\n"
                "    data['processed'] = True\n"
                "    return data\n"
                "_proc = DataProcessor(_config)\n"
                "_proc.add_step(_step)\n"
                "_r = _proc.process_record({'id': 'test1'})\n"
                "assert _r.status == TaskStatus.COMPLETED\n"
                "assert _r.output_data['processed'] is True\n"
                "\n"
                "# Verify _handle_step_error is referenced in the code\n"
                "assert '_handle_step_error' in _code_under_test, "
                "'Expected _handle_step_error in refactored code'\n"
                "\n"
                "# Verify original except block pattern was extracted\n"
                "assert 'should_continue' in _code_under_test, "
                "'Expected should_continue pattern'\n"
            ),
        )
    )

    # 9. Add type hints to merge_dicts (xlarge module, ~520 lines)
    tasks.append(
        _make_task(
            task_id="medium_type_hints_09",
            description=(
                "Add more specific type hints to merge_dicts: change "
                "the parameter types from 'dict' to 'dict[str, Any]' "
                "and the return type from 'dict' to 'dict[str, Any]'."
            ),
            base_code=xlarge,
            old_fragment=(
                "def merge_dicts(base: dict, override: dict) -> dict:\n"
                '    """Deep merge two dictionaries."""\n'
                "    result = base.copy()"
            ),
            new_fragment=(
                "def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:\n"
                '    """Deep merge two dictionaries."""\n'
                "    result = base.copy()"
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.STYLE,
            test_code=(
                "# Behavior unchanged\n"
                "assert merge_dicts({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}\n"
                "assert merge_dicts({'a': {'x': 1}}, {'a': {'y': 2}}) == {'a': {'x': 1, 'y': 2}}\n"
                "\n"
                "# Check annotations are dict[str, Any]\n"
                "import ast as _ast\n"
                "_tree = _ast.parse(_code_under_test)\n"
                "_found = False\n"
                "for _node in _ast.walk(_tree):\n"
                "    if isinstance(_node, _ast.FunctionDef) and _node.name == 'merge_dicts':\n"
                "        _ann = _ast.unparse(_node.args.args[0].annotation)\n"
                "        assert 'str' in _ann, f'Expected dict[str, Any], got {_ann}'\n"
                "        assert 'Any' in _ann, f'Expected dict[str, Any], got {_ann}'\n"
                "        _ret = _ast.unparse(_node.returns)\n"
                "        assert 'str' in _ret and 'Any' in _ret, "
                "f'Expected dict[str, Any] return, got {_ret}'\n"
                "        _found = True\n"
                "        break\n"
                "assert _found, 'merge_dicts not found'\n"
            ),
        )
    )

    # 10. Change list to set in _access_order (xlarge module)
    tasks.append(
        _make_task(
            task_id="medium_change_data_structure_10",
            description=(
                "In the LRUCache class, change _access_order from a list "
                "to use an OrderedDict for O(1) operations. Replace "
                "'self._access_order: list[str] = []' with "
                "'self._access_order: dict[str, None] = {}' and update "
                "the get() method to use 'del self._access_order[key]' "
                "and 'self._access_order[key] = None' instead of "
                "list .remove() and .append()."
            ),
            base_code=xlarge,
            old_fragment=(
                "        self._access_order: list[str] = []\n"
                "        self._hits = 0\n"
                "        self._misses = 0\n"
                "\n"
                "    def get(self, key: str) -> Any | None:\n"
                '        """Get value from cache."""\n'
                "        if key in self._cache:\n"
                "            self._hits += 1\n"
                "            self._access_order.remove(key)\n"
                "            self._access_order.append(key)\n"
                "            return self._cache[key][0]"
            ),
            new_fragment=(
                "        self._access_order: dict[str, None] = {}\n"
                "        self._hits = 0\n"
                "        self._misses = 0\n"
                "\n"
                "    def get(self, key: str) -> Any | None:\n"
                '        """Get value from cache."""\n'
                "        if key in self._cache:\n"
                "            self._hits += 1\n"
                "            del self._access_order[key]\n"
                "            self._access_order[key] = None\n"
                "            return self._cache[key][0]"
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.REFACTOR,
            test_code=(
                "# Structural: _access_order should be initialized as dict\n"
                "cache = LRUCache(max_size=3)\n"
                "assert isinstance(cache._access_order, dict), "
                "f'Expected dict, got {type(cache._access_order)}'\n"
                "\n"
                "# Verify get() uses dict ops in the source\n"
                "assert 'del self._access_order[key]' in _code_under_test, "
                "'Expected del dict op in get()'\n"
                "assert 'self._access_order[key] = None' in _code_under_test, "
                "'Expected dict assignment in get()'\n"
            ),
        )
    )

    # 11. Add new method to RecordValidator (large module)
    tasks.append(
        _make_task(
            task_id="medium_add_method_11",
            description=(
                "Add a new method 'validate_batch' to the RecordValidator "
                "class that takes a list of records and returns a list of "
                "tuples (record, bool, list[str]) for each record "
                "(the record, whether it's valid, and the errors). "
                "Add it right after the 'errors' property."
            ),
            base_code=large,
            old_fragment=(
                "    @property\n"
                "    def errors(self) -> list[str]:\n"
                '        """Return list of validation errors from last validate call."""\n'
                "        return self._errors.copy()"
            ),
            new_fragment=(
                "    @property\n"
                "    def errors(self) -> list[str]:\n"
                '        """Return list of validation errors from last validate call."""\n'
                "        return self._errors.copy()\n"
                "\n"
                "    def validate_batch(\n"
                "        self, records: list[dict[str, Any]]\n"
                "    ) -> list[tuple[dict[str, Any], bool, list[str]]]:\n"
                '        """Validate a batch of records."""\n'
                "        results = []\n"
                "        for record in records:\n"
                "            is_valid = self.validate(record)\n"
                "            results.append((record, is_valid, self.errors))\n"
                "        return results"
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_v = RecordValidator()\n"
                "_v.require('name')\n"
                "_v.optional('age')\n"
                "assert hasattr(_v, 'validate_batch'), 'Missing validate_batch'\n"
                "\n"
                "_records = [\n"
                "    {'name': 'Alice', 'age': 30},\n"
                "    {'wrong_field': 'bad'},\n"
                "    {'name': 'Bob'},\n"
                "]\n"
                "_results = _v.validate_batch(_records)\n"
                "assert len(_results) == 3\n"
                "assert _results[0][1] is True\n"
                "assert _results[1][1] is False\n"
                "assert _results[2][1] is True\n"
            ),
        )
    )

    # 12. Add @staticmethod decorator to format_duration (medium module)
    tasks.append(
        _make_task(
            task_id="medium_add_decorator_12",
            description=(
                "The Formatter.format_bytes method doesn't use any "
                "instance state (self is unused). Convert it to a "
                "@staticmethod by adding the decorator and removing "
                "the 'self' parameter."
            ),
            base_code=medium.replace(
                "\ndef format_duration(seconds: float) -> str:",
                "\nclass Formatter:\n    @staticmethod\n    def format_duration(seconds: float) -> str:",
            ).replace(
                '    """Format duration in human-readable format."""\n'
                "    if seconds < 1:\n"
                '        return f"{seconds * 1000:.0f}ms"\n'
                "    if seconds < 60:\n"
                '        return f"{seconds:.1f}s"\n'
                "    minutes = int(seconds // 60)\n"
                "    remaining = seconds % 60\n"
                '    return f"{minutes}m {remaining:.0f}s"',
                '        """Format duration in human-readable format."""\n'
                "        if seconds < 1:\n"
                '            return f"{seconds * 1000:.0f}ms"\n'
                "        if seconds < 60:\n"
                '            return f"{seconds:.1f}s"\n'
                "        minutes = int(seconds // 60)\n"
                "        remaining = seconds % 60\n"
                '        return f"{minutes}m {remaining:.0f}s"\n'
                "\n"
                "    def format_bytes(self, num_bytes: int) -> str:\n"
                '        """Format bytes in human-readable format."""\n'
                '        for unit in ["B", "KB", "MB", "GB"]:\n'
                "            if num_bytes < 1024:\n"
                '                return f"{num_bytes:.1f}{unit}"\n'
                "            num_bytes /= 1024\n"
                '        return f"{num_bytes:.1f}TB"',
            ),
            old_fragment=(
                "    def format_bytes(self, num_bytes: int) -> str:\n"
                '        """Format bytes in human-readable format."""'
            ),
            new_fragment=(
                "    @staticmethod\n"
                "    def format_bytes(num_bytes: int) -> str:\n"
                '        """Format bytes in human-readable format."""'
            ),
            difficulty=EditDifficulty.MEDIUM,
            edit_type=EditType.STYLE,
            test_code=(
                "assert isinstance(Formatter.__dict__['format_bytes'], staticmethod), "
                "'format_bytes should be a staticmethod'\n"
                "# Can call without instance\n"
                "result = Formatter.format_bytes(1024)\n"
                "assert 'KB' in result or 'B' in result\n"
            ),
        )
    )

    # === HARD (4 tasks) — xlarge/huge files, complex multi-line edits ===

    # 13. Fix multi-line logic in _should_retry (xlarge module, ~520 lines)
    tasks.append(
        _make_task(
            task_id="hard_fix_retry_logic_13",
            description=(
                "Fix the retry logic in APIClient._should_retry: it should "
                "also check that the attempt count is less than max_retries "
                "AND the status code is retryable AND the method is "
                "idempotent (GET, PUT, DELETE but not POST). Add a "
                "'method' parameter and the idempotency check."
            ),
            base_code=xlarge,
            old_fragment=(
                "    def _should_retry(self, status_code: int, attempt: int) -> bool:\n"
                '        """Determine if request should be retried."""\n'
                "        if attempt >= self.config.max_retries:\n"
                "            return False\n"
                "        return status_code in RETRY_STATUS_CODES"
            ),
            new_fragment=(
                '    def _should_retry(self, status_code: int, attempt: int, method: str = "GET") -> bool:\n'
                '        """Determine if request should be retried."""\n'
                "        if attempt >= self.config.max_retries:\n"
                "            return False\n"
                "        if status_code not in RETRY_STATUS_CODES:\n"
                "            return False\n"
                '        idempotent_methods = {"GET", "PUT", "DELETE", "HEAD", "OPTIONS"}\n'
                "        return method.upper() in idempotent_methods"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.BUG_FIX,
            test_code=(
                "import inspect as _inspect\n"
                "_config = ServiceConfig(max_retries=3)\n"
                "_client = APIClient(_config)\n"
                "_sig = _inspect.signature(_client._should_retry)\n"
                "assert 'method' in _sig.parameters, 'Missing method parameter'\n"
                "\n"
                "# Idempotent methods should be retried on retryable status\n"
                "assert _client._should_retry(500, 0, 'GET') is True\n"
                "assert _client._should_retry(429, 1, 'PUT') is True\n"
                "assert _client._should_retry(503, 0, 'DELETE') is True\n"
                "\n"
                "# POST is not idempotent - should NOT retry\n"
                "assert _client._should_retry(500, 0, 'POST') is False\n"
                "\n"
                "# Max retries exceeded\n"
                "assert _client._should_retry(500, 3, 'GET') is False\n"
                "\n"
                "# Non-retryable status code\n"
                "assert _client._should_retry(404, 0, 'GET') is False\n"
            ),
        )
    )

    # 14. Refactor process_record to use match/case (huge module, ~770 lines)
    tasks.append(
        _make_task(
            task_id="hard_refactor_dispatch_14",
            description=(
                "Refactor the EventBus.publish method to log different "
                "message levels based on the event_type prefix: events "
                "starting with 'task_failed' should use logger.warning, "
                "'task_completed' should use logger.info, and the generic "
                "exception handler should use logger.error instead of "
                "logger.exception. Also add a 'count' field to the event "
                "dict (set to the event log length + 1) so each event "
                "carries its sequence number."
            ),
            base_code=huge,
            old_fragment=(
                "    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:\n"
                '        """Publish an event to all subscribers."""\n'
                "        event = {\n"
                '            "type": event_type,\n'
                '            "data": data or {},\n'
                '            "timestamp": datetime.now().isoformat(),\n'
                "        }\n"
                "        self._event_log.append(event)\n"
                "        if len(self._event_log) > self._max_log_size:\n"
                "            self._event_log = self._event_log[-self._max_log_size :]\n"
                "\n"
                "        for handler in self._subscribers.get(event_type, []):\n"
                "            try:\n"
                "                handler(event)\n"
                "            except Exception:\n"
                '                logger.exception("Error in event handler for %s", event_type)'
            ),
            new_fragment=(
                "    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:\n"
                '        """Publish an event to all subscribers."""\n'
                "        event = {\n"
                '            "type": event_type,\n'
                '            "data": data or {},\n'
                '            "timestamp": datetime.now().isoformat(),\n'
                '            "count": len(self._event_log) + 1,\n'
                "        }\n"
                "        self._event_log.append(event)\n"
                "        if len(self._event_log) > self._max_log_size:\n"
                "            self._event_log = self._event_log[-self._max_log_size :]\n"
                "\n"
                '        if event_type.startswith("task_failed"):\n'
                '            logger.warning("Event: %s", event_type)\n'
                '        elif event_type.startswith("task_completed"):\n'
                '            logger.info("Event: %s", event_type)\n'
                "\n"
                "        for handler in self._subscribers.get(event_type, []):\n"
                "            try:\n"
                "                handler(event)\n"
                "            except Exception:\n"
                '                logger.error("Error in event handler for %s", event_type)'
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.REFACTOR,
            test_code=(
                "_bus = EventBus()\n"
                "_received = []\n"
                "_bus.subscribe('task_completed', lambda e: _received.append(e))\n"
                "_bus.publish('task_completed', {'id': '1'})\n"
                "\n"
                "assert len(_received) == 1\n"
                "assert 'count' in _received[0], 'Event should have count field'\n"
                "\n"
                "_log = _bus.get_event_log()\n"
                "assert len(_log) == 1\n"
                "assert _log[0]['type'] == 'task_completed'\n"
                "assert 'count' in _log[0]\n"
            ),
        )
    )

    # 15. Reorder imports and add missing one (huge module)
    tasks.append(
        _make_task(
            task_id="hard_reorder_imports_15",
            description=(
                "The imports are out of order. Reorder to: __future__ first, "
                "then standard library alphabetically, then third-party. "
                "Also add 'from collections import OrderedDict' after the "
                "standard library imports (it's needed by the LRUCache)."
            ),
            base_code=huge.replace(
                "import hashlib\n"
                "import json\n"
                "import logging\n"
                "import os\n"
                "import time\n"
                "from dataclasses import dataclass, field\n"
                "from datetime import datetime\n"
                "from enum import Enum\n"
                "from pathlib import Path\n"
                "from typing import Any",
                "import json\n"
                "import os\n"
                "import hashlib\n"
                "import time\n"
                "import logging\n"
                "from enum import Enum\n"
                "from typing import Any\n"
                "from dataclasses import dataclass, field\n"
                "from pathlib import Path\n"
                "from datetime import datetime",
            ),
            old_fragment=(
                "import json\n"
                "import os\n"
                "import hashlib\n"
                "import time\n"
                "import logging\n"
                "from enum import Enum\n"
                "from typing import Any\n"
                "from dataclasses import dataclass, field\n"
                "from pathlib import Path\n"
                "from datetime import datetime"
            ),
            new_fragment=(
                "import hashlib\n"
                "import json\n"
                "import logging\n"
                "import os\n"
                "import time\n"
                "from collections import OrderedDict\n"
                "from dataclasses import dataclass, field\n"
                "from datetime import datetime\n"
                "from enum import Enum\n"
                "from pathlib import Path\n"
                "from typing import Any"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.STYLE,
            test_code=(
                "# OrderedDict should be available in namespace\n"
                "from collections import OrderedDict as _OD\n"
                "assert OrderedDict is _OD\n"
                "\n"
                "# Check import ordering via AST\n"
                "import ast as _ast\n"
                "_tree = _ast.parse(_code_under_test)\n"
                "_import_names = []\n"
                "for _node in _ast.iter_child_nodes(_tree):\n"
                "    if isinstance(_node, _ast.Import):\n"
                "        for _alias in _node.names:\n"
                "            _import_names.append(_alias.name)\n"
                "if 'hashlib' in _import_names and 'json' in _import_names:\n"
                "    assert _import_names.index('hashlib') < _import_names.index('json'), "
                "'Imports not alphabetical'\n"
            ),
        )
    )

    # 16. Add comprehensive error handling to apply_all in MigrationManager (huge module)
    tasks.append(
        _make_task(
            task_id="hard_error_handling_16",
            description=(
                "Add comprehensive error handling to MigrationManager.apply_all: "
                "wrap each migration in try/except, log the error, record "
                "which migration failed, and rollback all previously applied "
                "migrations in this batch on failure. Return the original "
                "data if any migration fails."
            ),
            base_code=huge,
            old_fragment=(
                "    def apply_all(self, data: dict) -> dict:\n"
                '        """Apply all pending migrations."""\n'
                "        result = data.copy()\n"
                "        for migration in self._migrations:\n"
                '            if migration["name"] not in self._applied:\n'
                '                result = migration["up"](result)\n'
                '                self._applied.add(migration["name"])\n'
                '                logger.info("Applied migration: %s", migration["name"])\n'
                "        return result"
            ),
            new_fragment=(
                "    def apply_all(self, data: dict) -> dict:\n"
                '        """Apply all pending migrations."""\n'
                "        result = data.copy()\n"
                "        applied_in_batch: list[dict[str, Any]] = []\n"
                "        for migration in self._migrations:\n"
                '            if migration["name"] not in self._applied:\n'
                "                try:\n"
                '                    result = migration["up"](result)\n'
                '                    self._applied.add(migration["name"])\n'
                "                    applied_in_batch.append(migration)\n"
                '                    logger.info("Applied migration: %s", migration["name"])\n'
                "                except Exception:\n"
                "                    logger.error(\n"
                '                        "Migration failed: %s, rolling back batch",\n'
                '                        migration["name"],\n'
                "                    )\n"
                "                    for prev in reversed(applied_in_batch):\n"
                '                        result = prev["down"](result)\n'
                '                        self._applied.discard(prev["name"])\n'
                "                    return data\n"
                "        return result"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_mm = MigrationManager()\n"
                "\n"
                "def _up1(data):\n"
                "    data['v1'] = True\n"
                "    return data\n"
                "def _down1(data):\n"
                "    data.pop('v1', None)\n"
                "    return data\n"
                "def _up2(data):\n"
                "    raise RuntimeError('migration fail')\n"
                "def _down2(data):\n"
                "    return data\n"
                "\n"
                "_mm.register('m1', _up1, _down1)\n"
                "_mm.register('m2', _up2, _down2)\n"
                "\n"
                "_original = {'key': 'value'}\n"
                "_result = _mm.apply_all(_original)\n"
                "\n"
                "# Should return original data since m2 failed and m1 rolled back\n"
                "assert 'v1' not in _result, 'm1 should be rolled back'\n"
                "assert _result == _original, f'Should return original on failure'\n"
                "assert 'm1' not in _mm._applied, 'm1 should be rolled back'\n"
            ),
        )
    )

    # === VERY HARD (5 tasks) — huge file, multi-site edits, cross-class logic ===

    # 17. Add TTL-based expiration to LRUCache (2 edit sites: __init__ + get)
    tasks.append(
        _make_task(
            task_id="vhard_cache_ttl_17",
            description=(
                "Add TTL (time-to-live) support to LRUCache. The "
                "constructor should accept a 'ttl_seconds' parameter "
                "(float, default 0.0 meaning no expiry). When get() is "
                "called for an entry that was stored more than "
                "ttl_seconds ago, it should be treated as a cache miss: "
                "remove the stale entry and return None. Note that the "
                "cache already stores (value, timestamp) tuples in "
                "self._cache."
                + _NO_TAMPER_INSTRUCTION
            ),
            base_code=huge,
            old_fragment=(
                "    def __init__(self, max_size: int = 128) -> None:\n"
                "        self._max_size = max_size\n"
                "        self._cache: dict[str, tuple[Any, float]] = {}\n"
                "        self._access_order: list[str] = []\n"
                "        self._hits = 0\n"
                "        self._misses = 0\n"
                "\n"
                "    def get(self, key: str) -> Any | None:\n"
                '        """Get value from cache."""\n'
                "        if key in self._cache:\n"
                "            self._hits += 1\n"
                "            self._access_order.remove(key)\n"
                "            self._access_order.append(key)\n"
                "            return self._cache[key][0]\n"
                "        self._misses += 1\n"
                "        return None"
            ),
            new_fragment=(
                "    def __init__(self, max_size: int = 128, ttl_seconds: float = 0.0) -> None:\n"
                "        self._max_size = max_size\n"
                "        self._ttl = ttl_seconds\n"
                "        self._cache: dict[str, tuple[Any, float]] = {}\n"
                "        self._access_order: list[str] = []\n"
                "        self._hits = 0\n"
                "        self._misses = 0\n"
                "\n"
                "    def get(self, key: str) -> Any | None:\n"
                '        """Get value from cache."""\n'
                "        if key in self._cache:\n"
                "            if self._ttl > 0 and time.time() - self._cache[key][1] > self._ttl:\n"
                "                self.invalidate(key)\n"
                "                self._misses += 1\n"
                "                return None\n"
                "            self._hits += 1\n"
                "            self._access_order.remove(key)\n"
                "            self._access_order.append(key)\n"
                "            return self._cache[key][0]\n"
                "        self._misses += 1\n"
                "        return None"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "import time as _time\n"
                "cache = LRUCache(max_size=10, ttl_seconds=1)\n"
                "cache.put('key', 'value')\n"
                "assert cache.get('key') == 'value', 'Fresh entry should be returned'\n"
                "\n"
                "# Manually expire the entry by backdating its timestamp\n"
                "cache._cache['key'] = ('value', _time.time() - 2)\n"
                "assert cache.get('key') is None, 'Expired entry should return None'\n"
                "assert cache.size == 0, 'Expired entry should be invalidated'\n"
                "\n"
                "# TTL disabled by default\n"
                "cache2 = LRUCache(max_size=10)\n"
                "cache2.put('old', 'data')\n"
                "cache2._cache['old'] = ('data', _time.time() - 99999)\n"
                "assert cache2.get('old') == 'data', 'TTL=0 means no expiry'\n"
            ),
        )
    )

    # 18. Add per-step timing to DataProcessor.process_record
    #     (multi-site within one method: init list, wrap loop body, store metadata)
    tasks.append(
        _make_task(
            task_id="vhard_step_timing_18",
            description=(
                "Add per-step timing instrumentation to "
                "DataProcessor.process_record. After successful "
                "processing, result.metadata should contain a "
                "'step_timings' key holding a list of dicts, one per "
                "pipeline step executed. Each dict must have "
                "'step_index' (0-based int) and 'duration_s' (float, "
                "wall-clock seconds that step took). Timing should "
                "cover both successful steps and steps where the "
                "error handler recovered."
                + _NO_TAMPER_INSTRUCTION
            ),
            base_code=huge,
            old_fragment=(
                "        current_data = record.copy()\n"
                "        for step in self._pipeline:\n"
                "            try:\n"
                "                current_data = step(current_data)\n"
                "            except Exception as exc:\n"
                "                handler = self._error_handlers.get(type(exc))\n"
                "                if handler:\n"
                "                    current_data = handler(current_data, exc)\n"
                "                else:\n"
                "                    result.status = TaskStatus.FAILED\n"
                "                    result.error_message = str(exc)\n"
                '                    self._metrics["failed"] += 1\n'
                "                    return result\n"
                "\n"
                "        result.status = TaskStatus.COMPLETED\n"
                "        result.output_data = current_data\n"
                "        result.finished_at = datetime.now()\n"
                "        result.duration_seconds = (\n"
                "            result.finished_at - result.started_at\n"
                "        ).total_seconds()\n"
                "        result.output_hash = compute_hash(json.dumps(current_data, default=str))\n"
                '        self._metrics["processed"] += 1\n'
                "        return result"
            ),
            new_fragment=(
                "        current_data = record.copy()\n"
                "        step_timings: list[dict[str, Any]] = []\n"
                "        for i, step in enumerate(self._pipeline):\n"
                "            step_start = time.time()\n"
                "            try:\n"
                "                current_data = step(current_data)\n"
                "            except Exception as exc:\n"
                "                handler = self._error_handlers.get(type(exc))\n"
                "                if handler:\n"
                "                    current_data = handler(current_data, exc)\n"
                "                else:\n"
                "                    result.status = TaskStatus.FAILED\n"
                "                    result.error_message = str(exc)\n"
                '                    self._metrics["failed"] += 1\n'
                "                    return result\n"
                "            step_timings.append(\n"
                '                {"step_index": i, "duration_s": time.time() - step_start}\n'
                "            )\n"
                "\n"
                "        result.status = TaskStatus.COMPLETED\n"
                "        result.output_data = current_data\n"
                "        result.finished_at = datetime.now()\n"
                "        result.duration_seconds = (\n"
                "            result.finished_at - result.started_at\n"
                "        ).total_seconds()\n"
                "        result.output_hash = compute_hash(json.dumps(current_data, default=str))\n"
                '        result.metadata["step_timings"] = step_timings\n'
                '        self._metrics["processed"] += 1\n'
                "        return result"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_config = ServiceConfig()\n"
                "_proc = DataProcessor(_config)\n"
                "_proc.add_step(lambda d: d)\n"
                "_proc.add_step(lambda d: {**d, 'step2': True})\n"
                "_proc.add_step(lambda d: d)\n"
                "_r = _proc.process_record({'id': 'timing_test'})\n"
                "assert _r.status == TaskStatus.COMPLETED\n"
                "assert 'step_timings' in _r.metadata, 'Missing step_timings'\n"
                "_timings = _r.metadata['step_timings']\n"
                "assert len(_timings) == 3, f'Expected 3 steps, got {len(_timings)}'\n"
                "assert _timings[0]['step_index'] == 0\n"
                "assert _timings[1]['step_index'] == 1\n"
                "assert _timings[2]['step_index'] == 2\n"
                "assert all(t['duration_s'] >= 0 for t in _timings)\n"
            ),
        )
    )

    # 19. Add typed event registration + validation to EventBus
    #     (3 edit sites: __init__, new method, modify publish)
    tasks.append(
        _make_task(
            task_id="vhard_typed_events_19",
            description=(
                "Add event type registration and validation to EventBus. "
                "Callers should be able to register event types with "
                "required data fields via a register_event_type("
                "event_type, required_fields) method. When publish() is "
                "called for a registered event type, it must validate "
                "that all required fields are present in the data dict. "
                "If any are missing, raise ValueError with a message "
                "that includes the missing field names. Unregistered "
                "event types should continue to work without validation. "
                "Passing None as data to a registered event type should "
                "be treated as an empty dict for validation purposes."
                + _NO_TAMPER_INSTRUCTION
            ),
            base_code=huge,
            old_fragment=(
                "    def __init__(self) -> None:\n"
                "        self._subscribers: dict[str, list[callable]] = {}\n"
                "        self._event_log: list[dict[str, Any]] = []\n"
                "        self._max_log_size = 1000\n"
                "\n"
                "    def subscribe(self, event_type: str, handler: callable) -> None:\n"
                '        """Subscribe a handler to an event type."""\n'
                "        if event_type not in self._subscribers:\n"
                "            self._subscribers[event_type] = []\n"
                "        self._subscribers[event_type].append(handler)\n"
                "\n"
                "    def unsubscribe(self, event_type: str, handler: callable) -> None:\n"
                '        """Unsubscribe a handler from an event type."""\n'
                "        if event_type in self._subscribers:\n"
                "            self._subscribers[event_type] = [\n"
                "                h for h in self._subscribers[event_type] if h != handler\n"
                "            ]\n"
                "\n"
                "    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:\n"
                '        """Publish an event to all subscribers."""\n'
                "        event = {"
            ),
            new_fragment=(
                "    def __init__(self) -> None:\n"
                "        self._subscribers: dict[str, list[callable]] = {}\n"
                "        self._event_log: list[dict[str, Any]] = []\n"
                "        self._max_log_size = 1000\n"
                "        self._registered_types: dict[str, list[str]] = {}\n"
                "\n"
                "    def subscribe(self, event_type: str, handler: callable) -> None:\n"
                '        """Subscribe a handler to an event type."""\n'
                "        if event_type not in self._subscribers:\n"
                "            self._subscribers[event_type] = []\n"
                "        self._subscribers[event_type].append(handler)\n"
                "\n"
                "    def register_event_type(\n"
                "        self, event_type: str, required_fields: list[str]\n"
                "    ) -> None:\n"
                '        """Register an event type with required data fields."""\n'
                "        self._registered_types[event_type] = required_fields\n"
                "\n"
                "    def unsubscribe(self, event_type: str, handler: callable) -> None:\n"
                '        """Unsubscribe a handler from an event type."""\n'
                "        if event_type in self._subscribers:\n"
                "            self._subscribers[event_type] = [\n"
                "                h for h in self._subscribers[event_type] if h != handler\n"
                "            ]\n"
                "\n"
                "    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:\n"
                '        """Publish an event to all subscribers."""\n'
                "        if event_type in self._registered_types:\n"
                "            actual_data = data or {}\n"
                "            missing = [\n"
                "                f for f in self._registered_types[event_type]\n"
                "                if f not in actual_data\n"
                "            ]\n"
                "            if missing:\n"
                "                raise ValueError(\n"
                '                    f"Missing required fields for {event_type!r}: {missing}"\n'
                "                )\n"
                "        event = {"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_bus = EventBus()\n"
                "_bus.register_event_type('user_login', ['user_id', 'ip_address'])\n"
                "\n"
                "# Valid event works\n"
                "_bus.publish('user_login', {'user_id': '123', 'ip_address': '1.2.3.4'})\n"
                "assert len(_bus.get_event_log()) == 1\n"
                "\n"
                "# Missing required field raises ValueError\n"
                "_raised = False\n"
                "try:\n"
                "    _bus.publish('user_login', {'user_id': '123'})\n"
                "except ValueError as _e:\n"
                "    _raised = True\n"
                "    assert 'ip_address' in str(_e), f'Error should mention missing field: {_e}'\n"
                "assert _raised, 'Should have raised ValueError for missing field'\n"
                "\n"
                "# None data treated as empty dict\n"
                "_raised2 = False\n"
                "try:\n"
                "    _bus.publish('user_login', None)\n"
                "except ValueError:\n"
                "    _raised2 = True\n"
                "assert _raised2, 'Should raise ValueError when data is None'\n"
                "\n"
                "# Unregistered event type still works without validation\n"
                "_bus.publish('random_event', {'anything': 'goes'})\n"
                "_bus.publish('random_event')  # None data OK for unregistered\n"
                "assert len(_bus.get_event_log()) == 3\n"
            ),
        )
    )

    # 20. Add priority-based scheduling to TaskScheduler
    #     (3 edit sites: submit signature, pending structure, tick ordering)
    tasks.append(
        _make_task(
            task_id="vhard_priority_scheduler_20",
            description=(
                "Add priority-based scheduling to TaskScheduler. The "
                "submit() method should accept an optional 'priority' "
                "parameter (int, default 0, higher means more urgent). "
                "When tick() processes pending tasks, it should process "
                "higher-priority tasks first. Tasks with equal priority "
                "should maintain their original submission order (FIFO)."
                + _NO_TAMPER_INSTRUCTION
            ),
            base_code=huge,
            old_fragment=(
                "    def submit(self, task: dict[str, Any]) -> str:\n"
                '        """Submit a task for processing. Returns task ID."""\n'
                "        task_id = compute_hash(json.dumps(task, default=str) + str(time.time()))\n"
                '        self._pending.append({"id": task_id, "data": task, "submitted_at": time.time()})\n'
                '        self._event_bus.publish("task_submitted", {"task_id": task_id})\n'
                "        return task_id\n"
                "\n"
                "    def _can_start_more(self) -> bool:\n"
                '        """Check if more tasks can be started."""\n'
                "        return len(self._running) < self._max_concurrent\n"
                "\n"
                "    def tick(self, processor: DataProcessor) -> list[ProcessingResult]:\n"
                '        """Process pending tasks. Returns newly completed results."""\n'
                "        new_results = []"
            ),
            new_fragment=(
                "    def submit(self, task: dict[str, Any], priority: int = 0) -> str:\n"
                '        """Submit a task for processing. Returns task ID."""\n'
                "        task_id = compute_hash(json.dumps(task, default=str) + str(time.time()))\n"
                "        self._pending.append({\n"
                '            "id": task_id, "data": task,\n'
                '            "submitted_at": time.time(), "priority": priority,\n'
                "        })\n"
                '        self._event_bus.publish("task_submitted", {"task_id": task_id})\n'
                "        return task_id\n"
                "\n"
                "    def _can_start_more(self) -> bool:\n"
                '        """Check if more tasks can be started."""\n'
                "        return len(self._running) < self._max_concurrent\n"
                "\n"
                "    def tick(self, processor: DataProcessor) -> list[ProcessingResult]:\n"
                '        """Process pending tasks. Returns newly completed results."""\n'
                '        self._pending.sort(key=lambda t: t["priority"], reverse=True)\n'
                "        new_results = []"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_config = ServiceConfig()\n"
                "_proc = DataProcessor(_config)\n"
                "_proc.add_step(lambda d: d)\n"
                "\n"
                "_sched = TaskScheduler(max_concurrent=10)\n"
                "_sched.submit({'id': 'low'}, priority=1)\n"
                "_sched.submit({'id': 'high'}, priority=10)\n"
                "_sched.submit({'id': 'medium'}, priority=5)\n"
                "\n"
                "_results = _sched.tick(_proc)\n"
                "assert len(_results) == 3\n"
                "# Highest priority first\n"
                "assert _results[0].task_id == 'high'\n"
                "assert _results[1].task_id == 'medium'\n"
                "assert _results[2].task_id == 'low'\n"
                "\n"
                "# Default priority is 0\n"
                "_sched2 = TaskScheduler()\n"
                "_sched2.submit({'id': 'a'})\n"
                "_sched2.submit({'id': 'urgent'}, priority=99)\n"
                "_sched2.submit({'id': 'b'})\n"
                "_r2 = _sched2.tick(_proc)\n"
                "assert _r2[0].task_id == 'urgent'\n"
            ),
        )
    )

    # 21. Wire MigrationManager to publish events via EventBus
    #     (3 edit sites: __init__, apply_all, rollback)
    tasks.append(
        _make_task(
            task_id="vhard_migration_events_21",
            description=(
                "Wire MigrationManager to optionally publish lifecycle "
                "events through an EventBus. The constructor should "
                "accept an optional 'event_bus' parameter (default "
                "None). When a migration is successfully applied, "
                "publish a 'migration_applied' event with "
                "{'name': migration_name} as data. When a migration is "
                "successfully rolled back, publish a "
                "'migration_rolled_back' event with "
                "{'name': migration_name} as data. When no event bus "
                "is provided, the manager should behave exactly as "
                "before."
                + _NO_TAMPER_INSTRUCTION
            ),
            base_code=huge,
            old_fragment=(
                "    def __init__(self) -> None:\n"
                "        self._migrations: list[dict[str, Any]] = []\n"
                "        self._applied: set[str] = set()\n"
                "\n"
                "    def register(self, name: str, up_fn: callable, down_fn: callable) -> None:\n"
                '        """Register a migration."""\n'
                "        self._migrations.append({\n"
                '            "name": name,\n'
                '            "up": up_fn,\n'
                '            "down": down_fn,\n'
                "        })\n"
                "\n"
                "    def apply_all(self, data: dict) -> dict:\n"
                '        """Apply all pending migrations."""\n'
                "        result = data.copy()\n"
                "        for migration in self._migrations:\n"
                '            if migration["name"] not in self._applied:\n'
                '                result = migration["up"](result)\n'
                '                self._applied.add(migration["name"])\n'
                '                logger.info("Applied migration: %s", migration["name"])\n'
                "        return result\n"
                "\n"
                "    def rollback(self, data: dict, count: int = 1) -> dict:\n"
                '        """Rollback the last N migrations."""\n'
                "        result = data.copy()\n"
                "        applied_list = [\n"
                "            m for m in reversed(self._migrations)\n"
                '            if m["name"] in self._applied\n'
                "        ]\n"
                "        for migration in applied_list[:count]:\n"
                '            result = migration["down"](result)\n'
                '            self._applied.discard(migration["name"])\n'
                '            logger.info("Rolled back migration: %s", migration["name"])'
            ),
            new_fragment=(
                "    def __init__(self, event_bus: EventBus | None = None) -> None:\n"
                "        self._migrations: list[dict[str, Any]] = []\n"
                "        self._applied: set[str] = set()\n"
                "        self._event_bus = event_bus\n"
                "\n"
                "    def register(self, name: str, up_fn: callable, down_fn: callable) -> None:\n"
                '        """Register a migration."""\n'
                "        self._migrations.append({\n"
                '            "name": name,\n'
                '            "up": up_fn,\n'
                '            "down": down_fn,\n'
                "        })\n"
                "\n"
                "    def apply_all(self, data: dict) -> dict:\n"
                '        """Apply all pending migrations."""\n'
                "        result = data.copy()\n"
                "        for migration in self._migrations:\n"
                '            if migration["name"] not in self._applied:\n'
                '                result = migration["up"](result)\n'
                '                self._applied.add(migration["name"])\n'
                '                logger.info("Applied migration: %s", migration["name"])\n'
                "                if self._event_bus is not None:\n"
                "                    self._event_bus.publish(\n"
                '                        "migration_applied", {"name": migration["name"]}\n'
                "                    )\n"
                "        return result\n"
                "\n"
                "    def rollback(self, data: dict, count: int = 1) -> dict:\n"
                '        """Rollback the last N migrations."""\n'
                "        result = data.copy()\n"
                "        applied_list = [\n"
                "            m for m in reversed(self._migrations)\n"
                '            if m["name"] in self._applied\n'
                "        ]\n"
                "        for migration in applied_list[:count]:\n"
                '            result = migration["down"](result)\n'
                '            self._applied.discard(migration["name"])\n'
                '            logger.info("Rolled back migration: %s", migration["name"])\n'
                "            if self._event_bus is not None:\n"
                "                self._event_bus.publish(\n"
                '                    "migration_rolled_back", {"name": migration["name"]}\n'
                "                )"
            ),
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_bus = EventBus()\n"
                "_mm = MigrationManager(event_bus=_bus)\n"
                "\n"
                "def _up1(data):\n"
                "    data['v1'] = True\n"
                "    return data\n"
                "def _down1(data):\n"
                "    data.pop('v1', None)\n"
                "    return data\n"
                "def _up2(data):\n"
                "    data['v2'] = True\n"
                "    return data\n"
                "def _down2(data):\n"
                "    data.pop('v2', None)\n"
                "    return data\n"
                "\n"
                "_mm.register('m1', _up1, _down1)\n"
                "_mm.register('m2', _up2, _down2)\n"
                "\n"
                "# Apply all and check events\n"
                "_result = _mm.apply_all({'key': 'value'})\n"
                "assert _result['v1'] is True\n"
                "assert _result['v2'] is True\n"
                "_applied_events = _bus.get_event_log('migration_applied')\n"
                "assert len(_applied_events) == 2, f'Expected 2 applied events, got {len(_applied_events)}'\n"
                "assert _applied_events[0]['data']['name'] == 'm1'\n"
                "assert _applied_events[1]['data']['name'] == 'm2'\n"
                "\n"
                "# Rollback and check events\n"
                "_mm.rollback(_result, count=1)\n"
                "_rb_events = _bus.get_event_log('migration_rolled_back')\n"
                "assert len(_rb_events) == 1, f'Expected 1 rollback event, got {len(_rb_events)}'\n"
                "assert _rb_events[0]['data']['name'] == 'm2'\n"
                "\n"
                "# Without event bus, no errors\n"
                "_mm2 = MigrationManager()\n"
                "_mm2.register('m1', _up1, _down1)\n"
                "_mm2.apply_all({'key': 'value'})\n"
            ),
        )
    )

    # === EXTREME (3 tasks) — massive file (~1800 lines), 4-6 scattered edit sites ===

    massive = build_massive_module()

    # 22. Wire MetricsCollector into LRUCache, DataProcessor, APIClient, EventBus
    #     Edit sites: ~line 177 (LRUCache.__init__), ~line 184 (get), ~line 194 (put),
    #                 ~line 294 (DataProcessor.__init__), ~line 312 (process_record),
    #                 ~line 368 (APIClient.__init__), ~line 420 (APIClient.get),
    #                 ~line 473 (EventBus.__init__), ~line 491 (publish)
    #     Span: lines 177-507 (~330 lines between first and last edit)
    _massive_metrics = massive
    # LRUCache: add optional collector, report hits/misses
    _massive_metrics = _massive_metrics.replace(
        "    def __init__(self, max_size: int = 128) -> None:\n"
        "        self._max_size = max_size\n"
        "        self._cache: dict[str, tuple[Any, float]] = {}",
        "    def __init__(self, max_size: int = 128, "
        "metrics: MetricsCollector | None = None) -> None:\n"
        "        self._max_size = max_size\n"
        "        self._metrics_collector = metrics\n"
        "        self._cache: dict[str, tuple[Any, float]] = {}",
    )
    _massive_metrics = _massive_metrics.replace(
        "        if key in self._cache:\n"
        "            self._hits += 1\n"
        "            self._access_order.remove(key)\n"
        "            self._access_order.append(key)\n"
        "            return self._cache[key][0]\n"
        "        self._misses += 1\n"
        "        return None",
        "        if key in self._cache:\n"
        "            self._hits += 1\n"
        "            if self._metrics_collector is not None:\n"
        "                self._metrics_collector.increment('cache.hits')\n"
        "            self._access_order.remove(key)\n"
        "            self._access_order.append(key)\n"
        "            return self._cache[key][0]\n"
        "        self._misses += 1\n"
        "        if self._metrics_collector is not None:\n"
        "            self._metrics_collector.increment('cache.misses')\n"
        "        return None",
    )
    # DataProcessor: add optional collector, report processed/failed counts
    _massive_metrics = _massive_metrics.replace(
        "    def __init__(self, config: ServiceConfig) -> None:\n"
        "        self.config = config\n"
        "        self._pipeline: list[callable] = []\n"
        "        self._error_handlers: dict[type, callable] = {}",
        "    def __init__(self, config: ServiceConfig, "
        "metrics: MetricsCollector | None = None) -> None:\n"
        "        self.config = config\n"
        "        self._metrics_collector = metrics\n"
        "        self._pipeline: list[callable] = []\n"
        "        self._error_handlers: dict[type, callable] = {}",
    )
    _massive_metrics = _massive_metrics.replace(
        '        self._metrics["processed"] += 1\n'
        "        return result",
        '        self._metrics["processed"] += 1\n'
        "        if self._metrics_collector is not None:\n"
        "            self._metrics_collector.increment('processor.processed')\n"
        "        return result",
    )
    _massive_metrics = _massive_metrics.replace(
        "                    result.status = TaskStatus.FAILED\n"
        "                    result.error_message = str(exc)\n"
        '                    self._metrics["failed"] += 1\n'
        "                    return result",
        "                    result.status = TaskStatus.FAILED\n"
        "                    result.error_message = str(exc)\n"
        '                    self._metrics["failed"] += 1\n'
        "                    if self._metrics_collector is not None:\n"
        "                        self._metrics_collector.increment('processor.failed')\n"
        "                    return result",
    )
    # EventBus: add optional collector, count published events
    _massive_metrics = _massive_metrics.replace(
        "    def __init__(self) -> None:\n"
        "        self._subscribers: dict[str, list[callable]] = {}\n"
        "        self._event_log: list[dict[str, Any]] = []\n"
        "        self._max_log_size = 1000",
        "    def __init__(self, metrics: MetricsCollector | None = None) -> None:\n"
        "        self._subscribers: dict[str, list[callable]] = {}\n"
        "        self._event_log: list[dict[str, Any]] = []\n"
        "        self._max_log_size = 1000\n"
        "        self._metrics_collector = metrics",
    )
    _massive_metrics = _massive_metrics.replace(
        '                logger.exception("Error in event handler for %s", event_type)',
        '                logger.exception("Error in event handler for %s", event_type)\n'
        "        if self._metrics_collector is not None:\n"
        "            self._metrics_collector.increment('events.published')",
    )
    tasks.append(
        EditTask(
            task_id="extreme_metrics_wiring_22",
            description=(
                "Wire the existing MetricsCollector class into three "
                "other classes as an optional dependency. Each class's "
                "constructor should accept an optional 'metrics' "
                "parameter (MetricsCollector | None, default None).\n\n"
                "LRUCache: increment 'cache.hits' on each cache hit "
                "in get(), and 'cache.misses' on each miss.\n\n"
                "DataProcessor: increment 'processor.processed' when a "
                "record completes successfully, and 'processor.failed' "
                "when processing fails with an unhandled error.\n\n"
                "EventBus: increment 'events.published' each time an "
                "event is published (after handlers run).\n\n"
                "When no MetricsCollector is provided, all classes "
                "should behave exactly as before."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=massive,
            expected_code=_massive_metrics,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_mc = MetricsCollector()\n"
                "\n"
                "# LRUCache metrics\n"
                "_cache = LRUCache(max_size=10, metrics=_mc)\n"
                "_cache.put('a', 1)\n"
                "_cache.put('b', 2)\n"
                "assert _cache.get('a') == 1\n"
                "assert _cache.get('missing') is None\n"
                "assert _mc.get_counter('cache.hits') == 1\n"
                "assert _mc.get_counter('cache.misses') == 1\n"
                "\n"
                "# DataProcessor metrics\n"
                "_proc = DataProcessor(ServiceConfig(), metrics=_mc)\n"
                "_proc.add_step(lambda d: d)\n"
                "_proc.process_record({'id': 'ok'})\n"
                "assert _mc.get_counter('processor.processed') == 1\n"
                "\n"
                "def _fail_step(d):\n"
                "    raise RuntimeError('boom')\n"
                "_proc2 = DataProcessor(ServiceConfig(), metrics=_mc)\n"
                "_proc2.add_step(_fail_step)\n"
                "_proc2.process_record({'id': 'fail'})\n"
                "assert _mc.get_counter('processor.failed') == 1\n"
                "\n"
                "# EventBus metrics\n"
                "_bus = EventBus(metrics=_mc)\n"
                "_bus.publish('test_event')\n"
                "_bus.publish('test_event')\n"
                "assert _mc.get_counter('events.published') == 2\n"
                "\n"
                "# Without metrics, no errors\n"
                "_cache2 = LRUCache(max_size=5)\n"
                "_cache2.put('x', 1)\n"
                "_cache2.get('x')\n"
                "_cache2.get('nope')\n"
            ),
        )
    )

    # 23. Add request tracing (trace_id) through the processing pipeline
    #     Edit sites: ProcessingResult dataclass (~line 80), DataProcessor.process_record
    #     (~line 312), TaskScheduler.submit (~line 605), TaskScheduler.tick (~line 616),
    #     EventBus.publish (~line 491)
    #     Span: lines 80-635 (~555 lines between first and last edit)
    _massive_trace = massive
    # Add trace_id field to ProcessingResult
    _massive_trace = _massive_trace.replace(
        '    input_hash: str = ""\n'
        '    output_hash: str = ""\n'
        "    metadata: dict[str, Any] = field(default_factory=dict)",
        '    input_hash: str = ""\n'
        '    output_hash: str = ""\n'
        '    trace_id: str = ""\n'
        "    metadata: dict[str, Any] = field(default_factory=dict)",
    )
    # DataProcessor.process_record: accept and store trace_id
    _massive_trace = _massive_trace.replace(
        "    def process_record(self, record: dict[str, Any]) -> ProcessingResult:\n"
        '        """Process a single record through the pipeline."""\n'
        '        task_id = record.get("id", compute_hash(json.dumps(record)))\n'
        "        started = datetime.now()\n"
        "        result = ProcessingResult(\n"
        "            task_id=task_id,\n"
        "            status=TaskStatus.RUNNING,\n"
        "            started_at=started,\n"
        "            input_hash=compute_hash(json.dumps(record)),\n"
        "        )",
        "    def process_record(\n"
        '        self, record: dict[str, Any], trace_id: str = ""\n'
        "    ) -> ProcessingResult:\n"
        '        """Process a single record through the pipeline."""\n'
        '        task_id = record.get("id", compute_hash(json.dumps(record)))\n'
        "        started = datetime.now()\n"
        "        result = ProcessingResult(\n"
        "            task_id=task_id,\n"
        "            status=TaskStatus.RUNNING,\n"
        "            started_at=started,\n"
        "            input_hash=compute_hash(json.dumps(record)),\n"
        "            trace_id=trace_id,\n"
        "        )",
    )
    # TaskScheduler.submit: accept and store trace_id
    _massive_trace = _massive_trace.replace(
        "    def submit(self, task: dict[str, Any]) -> str:\n"
        '        """Submit a task for processing. Returns task ID."""\n'
        "        task_id = compute_hash(json.dumps(task, default=str) + str(time.time()))\n"
        '        self._pending.append({"id": task_id, "data": task, "submitted_at": time.time()})',
        '    def submit(self, task: dict[str, Any], trace_id: str = "") -> str:\n'
        '        """Submit a task for processing. Returns task ID."""\n'
        "        task_id = compute_hash(json.dumps(task, default=str) + str(time.time()))\n"
        "        self._pending.append({\n"
        '            "id": task_id, "data": task,\n'
        '            "submitted_at": time.time(), "trace_id": trace_id,\n'
        "        })",
    )
    # TaskScheduler.tick: pass trace_id to process_record
    _massive_trace = _massive_trace.replace(
        '            result = processor.process_record(task_info["data"])\n'
        "            del self._running[task_id]",
        "            result = processor.process_record(\n"
        '                task_info["data"], trace_id=task_info.get("trace_id", "")\n'
        "            )\n"
        "            del self._running[task_id]",
    )
    tasks.append(
        EditTask(
            task_id="extreme_request_tracing_23",
            description=(
                "Add request tracing support so a trace_id can flow "
                "through the entire processing pipeline.\n\n"
                "1. Add a 'trace_id: str' field (default empty string) "
                "to the ProcessingResult dataclass.\n\n"
                "2. DataProcessor.process_record should accept an "
                "optional 'trace_id' string parameter and store it on "
                "the ProcessingResult it creates.\n\n"
                "3. TaskScheduler.submit should accept an optional "
                "'trace_id' string parameter and store it in the "
                "pending task info.\n\n"
                "4. TaskScheduler.tick should pass each task's "
                "trace_id through to processor.process_record.\n\n"
                "All changes must be backwards-compatible — callers "
                "that don't pass trace_id should still work."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=massive,
            expected_code=_massive_trace,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "# Direct process_record with trace_id\n"
                "_config = ServiceConfig()\n"
                "_proc = DataProcessor(_config)\n"
                "_proc.add_step(lambda d: d)\n"
                "_r = _proc.process_record({'id': 't1'}, trace_id='abc-123')\n"
                "assert _r.trace_id == 'abc-123', f'Expected abc-123, got {_r.trace_id}'\n"
                "assert _r.status == TaskStatus.COMPLETED\n"
                "\n"
                "# Without trace_id, still works\n"
                "_r2 = _proc.process_record({'id': 't2'})\n"
                "assert _r2.trace_id == ''\n"
                "\n"
                "# Through TaskScheduler pipeline\n"
                "_sched = TaskScheduler(max_concurrent=10)\n"
                "_sched.submit({'id': 'traced'}, trace_id='sched-trace-1')\n"
                "_sched.submit({'id': 'untraced'})\n"
                "_results = _sched.tick(_proc)\n"
                "assert len(_results) == 2\n"
                "_traced = [r for r in _results if r.task_id == 'traced'][0]\n"
                "_untraced = [r for r in _results if r.task_id == 'untraced'][0]\n"
                "assert _traced.trace_id == 'sched-trace-1'\n"
                "assert _untraced.trace_id == ''\n"
            ),
        )
    )

    # 24. Add AuditLogger integration to DataProcessor and MigrationManager
    _massive_audit = massive
    # DataProcessor: add optional audit_logger, log process start/complete/fail
    _massive_audit = _massive_audit.replace(
        "    def __init__(self, config: ServiceConfig) -> None:\n"
        "        self.config = config\n"
        "        self._pipeline: list[callable] = []\n"
        "        self._error_handlers: dict[type, callable] = {}\n"
        "        self._metrics: dict[str, int] = {\n"
        '            "processed": 0,\n'
        '            "failed": 0,\n'
        '            "skipped": 0,\n'
        "        }",
        "    def __init__(\n"
        "        self, config: ServiceConfig, audit: AuditLogger | None = None\n"
        "    ) -> None:\n"
        "        self.config = config\n"
        "        self._audit = audit\n"
        "        self._pipeline: list[callable] = []\n"
        "        self._error_handlers: dict[type, callable] = {}\n"
        "        self._metrics: dict[str, int] = {\n"
        '            "processed": 0,\n'
        '            "failed": 0,\n'
        '            "skipped": 0,\n'
        "        }",
    )
    _massive_audit = _massive_audit.replace(
        '        self._metrics["processed"] += 1\n'
        "        return result",
        '        self._metrics["processed"] += 1\n'
        "        if self._audit is not None:\n"
        "            self._audit.log(\n"
        '                "process_record", "DataProcessor", result.task_id,\n'
        '                {"status": "completed"}\n'
        "            )\n"
        "        return result",
    )
    _massive_audit = _massive_audit.replace(
        "                    result.status = TaskStatus.FAILED\n"
        "                    result.error_message = str(exc)\n"
        '                    self._metrics["failed"] += 1\n'
        "                    return result",
        "                    result.status = TaskStatus.FAILED\n"
        "                    result.error_message = str(exc)\n"
        '                    self._metrics["failed"] += 1\n'
        "                    if self._audit is not None:\n"
        "                        self._audit.log(\n"
        '                            "process_record", "DataProcessor",\n'
        '                            result.task_id, {"status": "failed", "error": str(exc)}\n'
        "                        )\n"
        "                    return result",
    )
    # MigrationManager: add optional audit_logger, log apply/rollback
    _massive_audit = _massive_audit.replace(
        "    def __init__(self) -> None:\n"
        "        self._migrations: list[dict[str, Any]] = []\n"
        "        self._applied: set[str] = set()",
        "    def __init__(self, audit: AuditLogger | None = None) -> None:\n"
        "        self._migrations: list[dict[str, Any]] = []\n"
        "        self._applied: set[str] = set()\n"
        "        self._audit = audit",
    )
    _massive_audit = _massive_audit.replace(
        '                logger.info("Applied migration: %s", migration["name"])\n'
        "        return result",
        '                logger.info("Applied migration: %s", migration["name"])\n'
        "                if self._audit is not None:\n"
        "                    self._audit.log(\n"
        '                        "apply_migration", "MigrationManager",\n'
        '                        migration["name"],\n'
        "                    )\n"
        "        return result",
    )
    _massive_audit = _massive_audit.replace(
        '            logger.info("Rolled back migration: %s", migration["name"])',
        '            logger.info("Rolled back migration: %s", migration["name"])\n'
        "            if self._audit is not None:\n"
        "                self._audit.log(\n"
        '                    "rollback_migration", "MigrationManager",\n'
        '                    migration["name"],\n'
        "                )",
    )
    tasks.append(
        EditTask(
            task_id="extreme_audit_integration_24",
            description=(
                "Integrate the existing AuditLogger class into "
                "DataProcessor and MigrationManager as an optional "
                "dependency. Each class's constructor should accept "
                "an optional 'audit' parameter (AuditLogger | None, "
                "default None).\n\n"
                "DataProcessor: log a 'process_record' action (actor "
                "'DataProcessor', resource is the task_id) when a "
                "record completes successfully with details "
                "{'status': 'completed'}, and also when it fails with "
                "details {'status': 'failed', 'error': <message>}.\n\n"
                "MigrationManager: log an 'apply_migration' action "
                "(actor 'MigrationManager', resource is the migration "
                "name) after each successful apply, and a "
                "'rollback_migration' action after each successful "
                "rollback.\n\n"
                "When no AuditLogger is provided, both classes should "
                "behave exactly as before."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=massive,
            expected_code=_massive_audit,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_al = AuditLogger()\n"
                "\n"
                "# DataProcessor audit\n"
                "_proc = DataProcessor(ServiceConfig(), audit=_al)\n"
                "_proc.add_step(lambda d: d)\n"
                "_proc.process_record({'id': 'rec1'})\n"
                "_entries = _al.search(action='process_record')\n"
                "assert len(_entries) == 1\n"
                "assert _entries[0].resource == 'rec1'\n"
                "assert _entries[0].details['status'] == 'completed'\n"
                "\n"
                "# DataProcessor audit on failure\n"
                "def _boom(d):\n"
                "    raise RuntimeError('fail')\n"
                "_proc2 = DataProcessor(ServiceConfig(), audit=_al)\n"
                "_proc2.add_step(_boom)\n"
                "_proc2.process_record({'id': 'rec2'})\n"
                "_fail_entries = _al.search(action='process_record', resource='rec2')\n"
                "assert len(_fail_entries) == 1\n"
                "assert _fail_entries[0].details['status'] == 'failed'\n"
                "assert 'fail' in _fail_entries[0].details['error']\n"
                "\n"
                "# MigrationManager audit\n"
                "_mm = MigrationManager(audit=_al)\n"
                "_mm.register('m1', lambda d: {**d, 'v': 1}, lambda d: d)\n"
                "_mm.apply_all({'key': 'val'})\n"
                "_apply_entries = _al.search(action='apply_migration')\n"
                "assert len(_apply_entries) == 1\n"
                "assert _apply_entries[0].resource == 'm1'\n"
                "\n"
                "# Rollback audit\n"
                "_mm.rollback({'key': 'val'}, count=1)\n"
                "_rb_entries = _al.search(action='rollback_migration')\n"
                "assert len(_rb_entries) == 1\n"
                "assert _rb_entries[0].resource == 'm1'\n"
                "\n"
                "# Without audit, no errors\n"
                "_proc3 = DataProcessor(ServiceConfig())\n"
                "_proc3.add_step(lambda d: d)\n"
                "_proc3.process_record({'id': 'no_audit'})\n"
                "_mm2 = MigrationManager()\n"
                "_mm2.register('m1', lambda d: d, lambda d: d)\n"
                "_mm2.apply_all({})\n"
            ),
        )
    )

    # === SUPER-EXTREME (5 tasks) — giant file (~4200 lines), complex multi-site edits ===

    giant = build_giant_module()

    # 25. Wire MetricsCollector into StateMachine, JobScheduler, and EventStore
    #     Spans: ~773 (MetricsCollector) to ~3693 (EventStore) = ~2900 line span
    _giant_metrics = giant
    _giant_metrics = _giant_metrics.replace(
        "    def __init__(self, name: str, initial_state: str) -> None:\n"
        "        self._name = name\n"
        "        self._current_state = initial_state\n"
        "        self._initial_state = initial_state\n"
        "        self._transitions: list[Transition] = []",
        "    def __init__(\n"
        "        self, name: str, initial_state: str,\n"
        "        metrics: MetricsCollector | None = None,\n"
        "    ) -> None:\n"
        "        self._name = name\n"
        "        self._current_state = initial_state\n"
        "        self._initial_state = initial_state\n"
        "        self._transitions: list[Transition] = []\n"
        "        self._metrics_collector = metrics",
    )
    _giant_metrics = _giant_metrics.replace(
        "                self._transition_count += 1\n"
        "\n"
        "                # Run enter callbacks",
        "                self._transition_count += 1\n"
        "                if self._metrics_collector is not None:\n"
        "                    self._metrics_collector.increment('statemachine.transitions')\n"
        "\n"
        "                # Run enter callbacks",
    )
    _giant_metrics = _giant_metrics.replace(
        "    def __init__(self, max_concurrent: int = 4) -> None:\n"
        "        self._max_concurrent = max_concurrent\n"
        "        self._jobs: dict[str, Job] = {}",
        "    def __init__(\n"
        "        self, max_concurrent: int = 4,\n"
        "        metrics: MetricsCollector | None = None,\n"
        "    ) -> None:\n"
        "        self._max_concurrent = max_concurrent\n"
        "        self._metrics_collector = metrics\n"
        "        self._jobs: dict[str, Job] = {}",
    )
    _giant_metrics = _giant_metrics.replace(
        "            self._completed_jobs.add(job.job_id)\n"
        "        except Exception as exc:",
        "            self._completed_jobs.add(job.job_id)\n"
        "            if self._metrics_collector is not None:\n"
        "                self._metrics_collector.increment('jobs.completed')\n"
        "        except Exception as exc:",
    )
    _giant_metrics = _giant_metrics.replace(
        "                job.status = JobStatus.FAILED\n"
        "                self._dead_letter.append(job)",
        "                job.status = JobStatus.FAILED\n"
        "                self._dead_letter.append(job)\n"
        "                if self._metrics_collector is not None:\n"
        "                    self._metrics_collector.increment('jobs.failed')",
    )
    _giant_metrics = _giant_metrics.replace(
        "    def append(self, event: DomainEvent) -> None:\n"
        '        """Append an event to the store."""\n'
        "        self._events.append(event)\n"
        "        self._event_count += 1",
        "    def append(self, event: DomainEvent) -> None:\n"
        '        """Append an event to the store."""\n'
        "        self._events.append(event)\n"
        "        self._event_count += 1\n"
        "        if self._metrics_collector is not None:\n"
        "            self._metrics_collector.increment('eventstore.events')",
    )
    _giant_metrics = _giant_metrics.replace(
        "class EventStore:\n"
        '    """Stores and retrieves domain events."""\n'
        "\n"
        "    def __init__(self) -> None:\n"
        "        self._events: list[DomainEvent] = []\n"
        "        self._snapshots: dict[str, dict[str, Any]] = {}\n"
        "        self._subscribers: dict[str, list[callable]] = {}\n"
        "        self._event_count = 0",
        "class EventStore:\n"
        '    """Stores and retrieves domain events."""\n'
        "\n"
        "    def __init__(self, metrics: MetricsCollector | None = None) -> None:\n"
        "        self._events: list[DomainEvent] = []\n"
        "        self._snapshots: dict[str, dict[str, Any]] = {}\n"
        "        self._subscribers: dict[str, list[callable]] = {}\n"
        "        self._event_count = 0\n"
        "        self._metrics_collector = metrics",
    )
    tasks.append(
        EditTask(
            task_id="super_extreme_metrics_wide_25",
            description=(
                "Wire the existing MetricsCollector class into three "
                "distant classes as an optional dependency. Each "
                "class's constructor should accept an optional "
                "'metrics' parameter (MetricsCollector | None, "
                "default None).\n\n"
                "StateMachine: increment 'statemachine.transitions' "
                "each time a successful state transition occurs.\n\n"
                "JobScheduler: increment 'jobs.completed' when a job "
                "completes successfully, and 'jobs.failed' when a job "
                "exhausts retries and fails permanently.\n\n"
                "EventStore: increment 'eventstore.events' each time "
                "an event is appended.\n\n"
                "When no MetricsCollector is provided, all classes "
                "should behave exactly as before."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=giant,
            expected_code=_giant_metrics,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_mc = MetricsCollector()\n"
                "\n"
                "# StateMachine metrics\n"
                "_sm = StateMachine('test', 'idle', metrics=_mc)\n"
                "_sm.add_transition('idle', 'running', 'start')\n"
                "_sm.add_transition('running', 'done', 'finish')\n"
                "_sm.fire('start')\n"
                "_sm.fire('finish')\n"
                "assert _mc.get_counter('statemachine.transitions') == 2\n"
                "\n"
                "# JobScheduler metrics\n"
                "_js = JobScheduler(metrics=_mc)\n"
                "_js.add_job(Job('j1', 'ok', lambda: 'result'))\n"
                "_js.add_job(Job('j2', 'fail', lambda: (_ for _ in ()).throw(RuntimeError('boom'))))\n"
                "_js.execute_all()\n"
                "assert _mc.get_counter('jobs.completed') == 1\n"
                "assert _mc.get_counter('jobs.failed') == 1\n"
                "\n"
                "# EventStore metrics\n"
                "_es = EventStore(metrics=_mc)\n"
                "_es.append(DomainEvent('test', 'agg1', {'key': 'val'}))\n"
                "_es.append(DomainEvent('test', 'agg1', {'key': 'val2'}))\n"
                "assert _mc.get_counter('eventstore.events') == 2\n"
                "\n"
                "# Without metrics, no errors\n"
                "_sm2 = StateMachine('t', 'a')\n"
                "_sm2.add_transition('a', 'b', 'go')\n"
                "_sm2.fire('go')\n"
            ),
        )
    )

    # 26. Add AuditLogger to PluginRegistry, ConfigManager, and PermissionManager
    #     Spans: ~1747 to ~3094 = ~1350 line span
    _giant_audit = giant
    _giant_audit = _giant_audit.replace(
        "class PluginRegistry:\n"
        '    """Manages plugin registration, dependency resolution, and lifecycle."""\n'
        "\n"
        "    def __init__(self) -> None:\n"
        "        self._plugins: dict[str, PluginInterface] = {}",
        "class PluginRegistry:\n"
        '    """Manages plugin registration, dependency resolution, and lifecycle."""\n'
        "\n"
        "    def __init__(self, audit: AuditLogger | None = None) -> None:\n"
        "        self._audit = audit\n"
        "        self._plugins: dict[str, PluginInterface] = {}",
    )
    _giant_audit = _giant_audit.replace(
        "                plugin.initialize(self._context)\n"
        "                plugin.meta.loaded_at = datetime.now()\n"
        "                loaded.append(name)",
        "                plugin.initialize(self._context)\n"
        "                plugin.meta.loaded_at = datetime.now()\n"
        "                loaded.append(name)\n"
        "                if self._audit is not None:\n"
        '                    self._audit.log("plugin_loaded", "PluginRegistry", name)',
    )
    _giant_audit = _giant_audit.replace(
        "class ConfigManager:\n"
        '    """Manages layered configuration with defaults, env vars, and overrides."""\n'
        "\n"
        "    def __init__(self) -> None:\n"
        "        self._layers: list[ConfigLayer] = []",
        "class ConfigManager:\n"
        '    """Manages layered configuration with defaults, env vars, and overrides."""\n'
        "\n"
        "    def __init__(self, audit: AuditLogger | None = None) -> None:\n"
        "        self._audit = audit\n"
        "        self._layers: list[ConfigLayer] = []",
    )
    _giant_audit = _giant_audit.replace(
        "        self._notify_watchers(key, value)",
        "        self._notify_watchers(key, value)\n"
        "        if self._audit is not None:\n"
        '            self._audit.log("config_override", "ConfigManager", key, {"value": str(value)})',
    )
    _giant_audit = _giant_audit.replace(
        "class PermissionManager:\n"
        '    """Manages roles, permissions, and access control."""\n'
        "\n"
        "    def __init__(self) -> None:\n"
        "        self._roles: dict[str, Role] = {}",
        "class PermissionManager:\n"
        '    """Manages roles, permissions, and access control."""\n'
        "\n"
        "    def __init__(self, audit: AuditLogger | None = None) -> None:\n"
        "        self._audit = audit\n"
        "        self._roles: dict[str, Role] = {}",
    )
    _giant_audit = _giant_audit.replace(
        "        self._log_action(\"assign_role\", user_id, role_name)",
        "        self._log_action(\"assign_role\", user_id, role_name)\n"
        "        if self._audit is not None:\n"
        '            self._audit.log("assign_role", "PermissionManager", user_id, {"role": role_name})',
    )
    tasks.append(
        EditTask(
            task_id="super_extreme_audit_wide_26",
            description=(
                "Integrate the existing AuditLogger class into "
                "PluginRegistry, ConfigManager, and "
                "PermissionManager. Each class's constructor should "
                "accept an optional 'audit' parameter (AuditLogger | "
                "None, default None).\n\n"
                "PluginRegistry: log a 'plugin_loaded' action (actor "
                "'PluginRegistry', resource is the plugin name) each "
                "time a plugin is successfully loaded.\n\n"
                "ConfigManager: log a 'config_override' action (actor "
                "'ConfigManager', resource is the config key, details "
                "{'value': str(value)}) each time set_override is "
                "called.\n\n"
                "PermissionManager: log an 'assign_role' action "
                "(actor 'PermissionManager', resource is the user_id, "
                "details {'role': role_name}) each time a role is "
                "assigned.\n\n"
                "When no AuditLogger is provided, all classes should "
                "behave exactly as before."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=giant,
            expected_code=_giant_audit,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_al = AuditLogger()\n"
                "\n"
                "# PluginRegistry audit\n"
                "_pr = PluginRegistry(audit=_al)\n"
                "_p = PluginInterface(PluginMeta('test_plugin'))\n"
                "_pr.register(_p)\n"
                "_pr.load_all()\n"
                "_entries = _al.search(action='plugin_loaded')\n"
                "assert len(_entries) == 1\n"
                "assert _entries[0].resource == 'test_plugin'\n"
                "\n"
                "# ConfigManager audit\n"
                "_cm = ConfigManager(audit=_al)\n"
                "_cm.set_override('debug', True)\n"
                "_cfg_entries = _al.search(action='config_override')\n"
                "assert len(_cfg_entries) == 1\n"
                "assert _cfg_entries[0].resource == 'debug'\n"
                "assert _cfg_entries[0].details['value'] == 'True'\n"
                "\n"
                "# PermissionManager audit\n"
                "_pm = PermissionManager(audit=_al)\n"
                "_pm.create_role('admin')\n"
                "_pm.assign_role('user1', 'admin')\n"
                "_pm_entries = _al.search(action='assign_role', actor='PermissionManager')\n"
                "assert len(_pm_entries) == 1\n"
                "assert _pm_entries[0].resource == 'user1'\n"
                "assert _pm_entries[0].details['role'] == 'admin'\n"
                "\n"
                "# Without audit, no errors\n"
                "_pr2 = PluginRegistry()\n"
                "_cm2 = ConfigManager()\n"
                "_pm2 = PermissionManager()\n"
            ),
        )
    )

    # 27. Add trace_id to AdvancedPipeline, NotificationService, and MiddlewareChain
    #     Spans: ~2104 to ~4039 = ~1935 line span
    _giant_trace = giant
    _giant_trace = _giant_trace.replace(
        "    def send(\n"
        "        self,\n"
        "        recipient: str,\n"
        "        subject: str,\n"
        "        body: str,\n"
        "        channel_name: str | None = None,\n"
        "    ) -> bool:",
        "    def send(\n"
        "        self,\n"
        "        recipient: str,\n"
        "        subject: str,\n"
        "        body: str,\n"
        "        channel_name: str | None = None,\n"
        "        trace_id: str = \"\",\n"
        "    ) -> bool:",
    )
    _giant_trace = _giant_trace.replace(
        "        status = \"sent\" if success else \"failed\"\n"
        "        self._record_history(recipient, subject, ch_name, status)",
        "        status = \"sent\" if success else \"failed\"\n"
        "        self._record_history(recipient, subject, ch_name, status, trace_id)",
    )
    _giant_trace = _giant_trace.replace(
        "    def _record_history(\n"
        "        self,\n"
        "        recipient: str,\n"
        "        subject: str,\n"
        "        channel: str | None,\n"
        "        status: str,\n"
        "    ) -> None:",
        "    def _record_history(\n"
        "        self,\n"
        "        recipient: str,\n"
        "        subject: str,\n"
        "        channel: str | None,\n"
        "        status: str,\n"
        "        trace_id: str = \"\",\n"
        "    ) -> None:",
    )
    _giant_trace = _giant_trace.replace(
        '            "timestamp": datetime.now().isoformat(),\n'
        "        })\n"
        "        if len(self._history) > self._max_history:",
        '            "timestamp": datetime.now().isoformat(),\n'
        '            "trace_id": trace_id,\n'
        "        })\n"
        "        if len(self._history) > self._max_history:",
    )
    _giant_trace = _giant_trace.replace(
        "    def execute(self, data: dict[str, Any] | None = None) -> PipelineContext:",
        "    def execute(self, data: dict[str, Any] | None = None, trace_id: str = \"\") -> PipelineContext:",
    )
    _giant_trace = _giant_trace.replace(
        "        ctx = PipelineContext(data)\n"
        "        ctx.started_at = datetime.now()\n"
        "        self._execution_count += 1",
        "        ctx = PipelineContext(data)\n"
        "        ctx.metadata[\"trace_id\"] = trace_id\n"
        "        ctx.started_at = datetime.now()\n"
        "        self._execution_count += 1",
    )
    _giant_trace = _giant_trace.replace(
        "    def handle(self, request: dict[str, Any]) -> MiddlewareContext:",
        "    def handle(self, request: dict[str, Any], trace_id: str = \"\") -> MiddlewareContext:",
    )
    _giant_trace = _giant_trace.replace(
        "        ctx = MiddlewareContext(request)\n"
        "        self._total_requests += 1",
        "        ctx = MiddlewareContext(request)\n"
        "        ctx.metadata[\"trace_id\"] = trace_id\n"
        "        self._total_requests += 1",
    )
    tasks.append(
        EditTask(
            task_id="super_extreme_tracing_wide_27",
            description=(
                "Add trace_id support to three distant classes so "
                "requests can be tracked end-to-end.\n\n"
                "NotificationService.send: accept an optional "
                "'trace_id' string parameter (default empty). Pass "
                "it through to _record_history so each history entry "
                "includes a 'trace_id' field.\n\n"
                "AdvancedPipeline.execute: accept an optional "
                "'trace_id' string parameter (default empty). Store "
                "it in ctx.metadata['trace_id'] at the start.\n\n"
                "MiddlewareChain.handle: accept an optional "
                "'trace_id' string parameter (default empty). Store "
                "it in ctx.metadata['trace_id'] at the start.\n\n"
                "All changes must be backwards-compatible."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=giant,
            expected_code=_giant_trace,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "# NotificationService trace_id\n"
                "_ns = NotificationService()\n"
                "_ch = NotificationChannel('email', 'smtp')\n"
                "_ns.add_channel(_ch)\n"
                "_ns.send('user@test.com', 'Hi', 'Body', trace_id='t-123')\n"
                "_hist = _ns.get_history()\n"
                "assert len(_hist) == 1\n"
                "assert _hist[0]['trace_id'] == 't-123', f\"Expected t-123, got {_hist[0].get('trace_id')}\"\n"
                "\n"
                "# Without trace_id\n"
                "_ns.send('user@test.com', 'Hi2', 'Body2')\n"
                "assert _ns.get_history()[-1]['trace_id'] == ''\n"
                "\n"
                "# AdvancedPipeline trace_id\n"
                "_pipe = AdvancedPipeline('test')\n"
                "_pipe.add_stage(PipelineStage('s1', lambda d: d))\n"
                "_ctx = _pipe.execute({'key': 'val'}, trace_id='pipe-456')\n"
                "assert _ctx.metadata.get('trace_id') == 'pipe-456'\n"
                "\n"
                "# Without trace_id\n"
                "_ctx2 = _pipe.execute({'key': 'val'})\n"
                "assert _ctx2.metadata.get('trace_id') == ''\n"
                "\n"
                "# MiddlewareChain trace_id\n"
                "_chain = MiddlewareChain()\n"
                "_chain.set_handler(lambda ctx: None)\n"
                "_mc = _chain.handle({'path': '/test'}, trace_id='mw-789')\n"
                "assert _mc.metadata.get('trace_id') == 'mw-789'\n"
            ),
        )
    )

    # 28. Wire EventBus into WorkflowEngine, SchemaValidator, and ServiceMesh
    #     Spans: ~470 to ~2868 = ~2400 line span
    _giant_events = giant
    _giant_events = _giant_events.replace(
        "    def __init__(self, name: str = \"default\") -> None:\n"
        "        self._name = name\n"
        "        self._steps: list[WorkflowStep] = []\n"
        "        self._variables: dict[str, Any] = {}\n"
        "        self._execution_count = 0",
        "    def __init__(\n"
        "        self, name: str = \"default\",\n"
        "        event_bus: EventBus | None = None,\n"
        "    ) -> None:\n"
        "        self._name = name\n"
        "        self._event_bus = event_bus\n"
        "        self._steps: list[WorkflowStep] = []\n"
        "        self._variables: dict[str, Any] = {}\n"
        "        self._execution_count = 0",
    )
    _giant_events = _giant_events.replace(
        "        result.success = True\n"
        "        result.finished_at = datetime.now()\n"
        "        return result",
        "        result.success = True\n"
        "        result.finished_at = datetime.now()\n"
        "        if self._event_bus is not None:\n"
        "            self._event_bus.publish(\n"
        '                "workflow_completed",\n'
        '                {"workflow": self._name, "steps": result.steps_executed},\n'
        "            )\n"
        "        return result",
    )
    _giant_events = _giant_events.replace(
        "class SchemaValidator:\n"
        '    """Validates data against schemas."""\n'
        "\n"
        "    def __init__(self) -> None:\n"
        "        self._schemas: dict[str, Schema] = {}",
        "class SchemaValidator:\n"
        '    """Validates data against schemas."""\n'
        "\n"
        "    def __init__(self, event_bus: EventBus | None = None) -> None:\n"
        "        self._event_bus = event_bus\n"
        "        self._schemas: dict[str, Schema] = {}",
    )
    _giant_events = _giant_events.replace(
        "        self._validation_count += 1\n"
        "        errors = self._validate_against(data, schema)\n"
        "        self._error_count += len(errors)\n"
        "        return errors",
        "        self._validation_count += 1\n"
        "        errors = self._validate_against(data, schema)\n"
        "        self._error_count += len(errors)\n"
        "        if self._event_bus is not None:\n"
        "            self._event_bus.publish(\n"
        '                "validation_completed",\n'
        '                {"schema": schema_name, "valid": len(errors) == 0, "error_count": len(errors)},\n'
        "            )\n"
        "        return errors",
    )
    _giant_events = _giant_events.replace(
        "class ServiceMesh:\n"
        '    """Manages service discovery, routing, and load balancing."""\n'
        "\n"
        "    def __init__(self) -> None:\n"
        "        self._services: dict[str, list[ServiceEndpoint]] = {}",
        "class ServiceMesh:\n"
        '    """Manages service discovery, routing, and load balancing."""\n'
        "\n"
        "    def __init__(self, event_bus: EventBus | None = None) -> None:\n"
        "        self._event_bus = event_bus\n"
        "        self._services: dict[str, list[ServiceEndpoint]] = {}",
    )
    _giant_events = _giant_events.replace(
        "        if cb:\n"
        "            cb.record_success()\n"
        "\n"
        "        return result",
        "        if cb:\n"
        "            cb.record_success()\n"
        "\n"
        "        if self._event_bus is not None:\n"
        "            self._event_bus.publish(\n"
        '                "request_routed",\n'
        '                {"service": service_name, "endpoint": endpoint.address},\n'
        "            )\n"
        "        return result",
    )
    tasks.append(
        EditTask(
            task_id="super_extreme_eventbus_wide_28",
            description=(
                "Wire the existing EventBus class into WorkflowEngine, "
                "SchemaValidator, and ServiceMesh. Each class's "
                "constructor should accept an optional 'event_bus' "
                "parameter (EventBus | None, default None).\n\n"
                "WorkflowEngine: publish a 'workflow_completed' event "
                "with {'workflow': name, 'steps': steps_executed} when "
                "a workflow completes successfully.\n\n"
                "SchemaValidator: publish a 'validation_completed' "
                "event with {'schema': schema_name, 'valid': bool, "
                "'error_count': int} after each validation.\n\n"
                "ServiceMesh: publish a 'request_routed' event with "
                "{'service': service_name, 'endpoint': address} after "
                "successfully routing a request.\n\n"
                "When no EventBus is provided, all classes should "
                "behave exactly as before."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=giant,
            expected_code=_giant_events,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_bus = EventBus()\n"
                "\n"
                "# WorkflowEngine events\n"
                "_wf = WorkflowEngine('test_wf', event_bus=_bus)\n"
                "_wf.add_step('s1', lambda d: d)\n"
                "_r = _wf.execute({'key': 'val'})\n"
                "assert _r.success\n"
                "_wf_events = _bus.get_event_log('workflow_completed')\n"
                "assert len(_wf_events) == 1\n"
                "assert _wf_events[0]['data']['workflow'] == 'test_wf'\n"
                "\n"
                "# SchemaValidator events\n"
                "_sv = SchemaValidator(event_bus=_bus)\n"
                "_schema = Schema('person')\n"
                "_schema.require('name', str)\n"
                "_sv.register_schema(_schema)\n"
                "_sv.validate({'name': 'Alice'}, 'person')\n"
                "_val_events = _bus.get_event_log('validation_completed')\n"
                "assert len(_val_events) == 1\n"
                "assert _val_events[0]['data']['valid'] is True\n"
                "\n"
                "# ServiceMesh events\n"
                "_mesh = ServiceMesh(event_bus=_bus)\n"
                "_mesh.register_endpoint(ServiceEndpoint('api', 'localhost', 8080))\n"
                "_mesh.route_request('api', {'path': '/test'})\n"
                "_route_events = _bus.get_event_log('request_routed')\n"
                "assert len(_route_events) == 1\n"
                "assert _route_events[0]['data']['service'] == 'api'\n"
                "\n"
                "# Without event_bus\n"
                "_wf2 = WorkflowEngine('t')\n"
                "_sv2 = SchemaValidator()\n"
                "_mesh2 = ServiceMesh()\n"
            ),
        )
    )

    # 29. Add HealthChecker integration to ConnectionPool, RateLimiter, and TieredCache
    #     Spans: ~857 to ~3603 = ~2750 line span
    _giant_health = giant
    _giant_health = _giant_health.replace(
        "class ConnectionPool:\n"
        '    """Manages a pool of reusable connections."""\n'
        "\n"
        "    def __init__(\n"
        "        self,\n"
        "        pool_id: str = \"default\",\n"
        "        max_size: int = 10,\n"
        "        max_idle_seconds: float = 300.0,\n"
        "    ) -> None:",
        "class ConnectionPool:\n"
        '    """Manages a pool of reusable connections."""\n'
        "\n"
        "    def __init__(\n"
        "        self,\n"
        "        pool_id: str = \"default\",\n"
        "        max_size: int = 10,\n"
        "        max_idle_seconds: float = 300.0,\n"
        "        health_checker: HealthChecker | None = None,\n"
        "    ) -> None:",
    )
    _giant_health = _giant_health.replace(
        "        self._total_created = 0\n"
        "        self._total_reused = 0",
        "        self._total_created = 0\n"
        "        self._total_reused = 0\n"
        "        if health_checker is not None:\n"
        "            health_checker.register(\n"
        "                f\"pool:{pool_id}\",\n"
        "                lambda: HealthCheck(\n"
        "                    f\"pool:{pool_id}\",\n"
        "                    HealthStatus.HEALTHY if self.available_count > 0\n"
        "                    else HealthStatus.DEGRADED,\n"
        "                    f\"{self.available_count} available\",\n"
        "                ),\n"
        "            )",
    )
    _giant_health = _giant_health.replace(
        "class RateLimiter:\n"
        '    """Token bucket rate limiter."""\n'
        "\n"
        "    def __init__(\n"
        "        self,\n"
        "        rate: float = 10.0,\n"
        "        burst: int = 20,\n"
        "    ) -> None:",
        "class RateLimiter:\n"
        '    """Token bucket rate limiter."""\n'
        "\n"
        "    def __init__(\n"
        "        self,\n"
        "        rate: float = 10.0,\n"
        "        burst: int = 20,\n"
        "        health_checker: HealthChecker | None = None,\n"
        "    ) -> None:",
    )
    _giant_health = _giant_health.replace(
        "        self._total_allowed = 0\n"
        "        self._total_rejected = 0",
        "        self._total_allowed = 0\n"
        "        self._total_rejected = 0\n"
        "        if health_checker is not None:\n"
        "            health_checker.register(\n"
        "                \"rate_limiter\",\n"
        "                lambda: HealthCheck(\n"
        "                    \"rate_limiter\",\n"
        "                    HealthStatus.HEALTHY if self._tokens > 0\n"
        "                    else HealthStatus.DEGRADED,\n"
        "                    f\"{self._tokens:.0f} tokens available\",\n"
        "                ),\n"
        "            )",
    )
    _giant_health = _giant_health.replace(
        "class TieredCache:\n"
        '    """Two-tier cache with fast L1 and larger L2."""\n'
        "\n"
        "    def __init__(\n"
        "        self,\n"
        "        l1_size: int = 50,\n"
        "        l2_size: int = 500,\n"
        "        l1_ttl: float = 60.0,\n"
        "        l2_ttl: float = 300.0,\n"
        "    ) -> None:",
        "class TieredCache:\n"
        '    """Two-tier cache with fast L1 and larger L2."""\n'
        "\n"
        "    def __init__(\n"
        "        self,\n"
        "        l1_size: int = 50,\n"
        "        l2_size: int = 500,\n"
        "        l1_ttl: float = 60.0,\n"
        "        l2_ttl: float = 300.0,\n"
        "        health_checker: HealthChecker | None = None,\n"
        "    ) -> None:",
    )
    _giant_health = _giant_health.replace(
        "        self._promotions = 0",
        "        self._promotions = 0\n"
        "        if health_checker is not None:\n"
        "            health_checker.register(\n"
        "                \"tiered_cache\",\n"
        "                lambda: HealthCheck(\n"
        "                    \"tiered_cache\",\n"
        "                    HealthStatus.HEALTHY,\n"
        "                    f\"L1={self._l1.size} L2={self._l2.size}\",\n"
        "                ),\n"
        "            )",
    )
    tasks.append(
        EditTask(
            task_id="super_extreme_healthcheck_wide_29",
            description=(
                "Add HealthChecker integration to ConnectionPool, "
                "RateLimiter, and TieredCache. Each class's "
                "constructor should accept an optional "
                "'health_checker' parameter (HealthChecker | None, "
                "default None). When provided, each class should "
                "register a health check during __init__.\n\n"
                "ConnectionPool: register a check named "
                "'pool:{pool_id}' that returns HEALTHY when "
                "available_count > 0, DEGRADED otherwise, with "
                "a message showing the available count.\n\n"
                "RateLimiter: register a check named 'rate_limiter' "
                "that returns HEALTHY when tokens > 0, DEGRADED "
                "otherwise, with a message showing tokens available.\n\n"
                "TieredCache: register a check named 'tiered_cache' "
                "that returns HEALTHY with a message showing L1 and "
                "L2 sizes.\n\n"
                "When no HealthChecker is provided, all classes "
                "should behave exactly as before."
                + _NO_TAMPER_INSTRUCTION
            ),
            original_code=giant,
            expected_code=_giant_health,
            difficulty=EditDifficulty.HARD,
            edit_type=EditType.ADD_FEATURE,
            test_code=(
                "_hc = HealthChecker()\n"
                "\n"
                "# ConnectionPool registers health check\n"
                "_pool = ConnectionPool(pool_id='main', health_checker=_hc)\n"
                "_results = _hc.run_all()\n"
                "assert 'pool:main' in _results\n"
                "assert _results['pool:main'].status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)\n"
                "\n"
                "# RateLimiter registers health check\n"
                "_rl = RateLimiter(health_checker=_hc)\n"
                "_results2 = _hc.run_all()\n"
                "assert 'rate_limiter' in _results2\n"
                "assert _results2['rate_limiter'].status == HealthStatus.HEALTHY\n"
                "\n"
                "# TieredCache registers health check\n"
                "_tc = TieredCache(health_checker=_hc)\n"
                "_results3 = _hc.run_all()\n"
                "assert 'tiered_cache' in _results3\n"
                "assert _results3['tiered_cache'].status == HealthStatus.HEALTHY\n"
                "\n"
                "# Without health_checker, no errors\n"
                "_pool2 = ConnectionPool()\n"
                "_rl2 = RateLimiter()\n"
                "_tc2 = TieredCache()\n"
            ),
        )
    )

    return tasks


TASKS: list[EditTask] = _build_tasks()


def get_tasks() -> list[EditTask]:
    """Return all benchmark tasks."""
    return TASKS


def get_tasks_by_difficulty(
    difficulty: EditDifficulty,
) -> list[EditTask]:
    """Return tasks filtered by difficulty."""
    return [t for t in TASKS if t.difficulty == difficulty]
