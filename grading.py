#!/usr/bin/env python3
"""Standalone FrontierFinance rubric grader for OpenAI-compatible endpoints.

Adapted from Samaya AI's Apache-2.0 licensed FrontierFinance grader:
https://github.com/samaya-ai/frontier-finance

The input is one JSON object per line.  By default the script reads:

* query: ``query`` (also accepts ``question`` / ``prompt``)
* response: ``system_summary`` / ``system_response`` / ``response`` /
  ``answer`` / ``final_answer`` / ``output``
* rubrics: ``rubrics``

Use ``--response-field`` when the response is stored under another key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit(
        "Missing dependency 'openai'. Install it with: python -m pip install 'openai>=1.0'"
    ) from exc


LOGGER = logging.getLogger("frontier_finance_grader")


SYSTEM_PROMPT = """\
You are a senior financial analyst. Your task is to evaluate a financial report against a list of pre-defined rubrics. The report presented to you is generated to answer a specific financial query. For each given rubric, you are expected to produce a binary judgement on whether the rubric is satisfied or not by the financial report.
For each task, you will be given the following:
1. A financial query, which specifies the information the user is seeking.
2. The date the query was made. This is important for assessing the time understanding of the system. Whenever necessary, you should use this date as the temporal anchor for interpreting relative date terms in both the query and the rubrics.
3. A financial report which aims to answer that query.
4. One or more natural language rubrics, each checking a specific aspect of the report.
All of the input will be clearly marked in XML tags. Your task is to judge whether the report adequately satisfies each of the given rubrics. You must evaluate the report objectively and thoroughly.
Pay special attention to the following aspects when making your judgement:
1. **Each rubric should be judged independently**. Even in the case that one rubric seems related to another, you need to give your judgement of whether each rubric is satisfied independently.
2. **Pay attention to numerical units**. The report and the rubric might use different units to represent the same number. Take this into account when making your judgement. For example, "USD 2.1 billion" is equivalent to "USD 2,100 million".
3. **Accept reasonable numerical approximation**. A figure in the report is acceptable if it equals the rubric's figure after rounding the rubric's figure to the (coarser) precision the report uses. A figure stated at the same or finer precision than the rubric's, but with a different value, is NOT acceptable — even if numerically close.
For example, against a rubric value of "3,098 million": "3.1 billion" is acceptable (a correct rounding to two significant figures), but "3,105 million" is not (it asserts a precise, different value). Likewise against "7.14%": "7.1%" is acceptable, but "7.25%" is not.\
"""


USER_PROMPT_TEMPLATE = """\
You will evaluate the report below against the given set of rubrics. The report has been written to answer a specific query.

The query is provided below within the <query> tags.
<query>
{query}
</query>

The date the query was submitted is provided below within the <date> tags. This is important for assessing whether the report correctly understands the time aspect of the query.
<date>
{query_date}
</date>
The financial report is provided below within the <report> tags.
<report>
{response}
</report>

Now that you have read the query and the report, please evaluate whether the report satisfies each of the following rubrics. The list of rubrics is provided below within the <rubrics> tags. Each rubric is annotated with a unique ID, which you should use in your output to refer to that rubric.
<rubrics>
{criteria}
</rubrics>
For each rubric, determine if the report adequately satisfies it. As a reminder, pay attention to the following aspects mentioned before:
- Each rubric should be judged independently.
- Pay attention to numerical units.
- Accept reasonable numerical approximation.
Your output must be ONLY a valid JSON object with the following structure:
```json
{{
  "0": {{
    "reason": "concise 1-sentence reason for your judgement on rubric 0",
    "label": true/false
  }},
  "1": {{
    "reason": "concise 1-sentence reason for your judgement on rubric 1",
    "label": true/false
  }},
  ...
}}
```
The keys must be string representations of the given rubric ID (starting from 0).
The "reason" field should contain your 1-sentence concise reasoning about whether the rubric is satisfied.
The "label" field must be a boolean value (true if the rubric is satisfied, false otherwise).

Now provide your judgements. Recall that the user query is: <query> {query} </query> and the date the query was made is: <date> {query_date} </date>.
Output your evaluation as the JSON object specified above and nothing else.\
"""


JSON_FORMAT_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Make sure you strictly follow the output format and output a valid "
    "JSON object that can be parsed successfully, and nothing else."
)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

QUESTION_FIELDS = ("query", "question", "Question", "prompt", "instruction")
RESPONSE_FIELDS = (
    "system_summary",
    "system_response",
    "response",
    "answer",
    "Answer",
    "final_answer",
    "model_response",
    "output",
    "prediction",
)
ID_FIELDS = ("query_id", "id", "question_id")


@dataclass(slots=True)
class EvalItem:
    source_line: int
    query_id: str
    query: str
    query_date: str
    response: str
    rubrics: list[dict[str, Any]]


def nested_get(record: dict[str, Any], dotted_field: str) -> Any:
    """Read a top-level or dot-separated field from a mapping."""
    value: Any = record
    for part in dotted_field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in record and record[field] is not None:
            return record[field]
    return None


def content_to_text(value: Any) -> str:
    """Convert common answer/message representations to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        # A chat message list: prefer the last assistant message.
        assistant_parts: list[str] = []
        for entry in value:
            if isinstance(entry, dict) and entry.get("role") == "assistant":
                text = content_to_text(entry.get("content"))
                if text:
                    assistant_parts.append(text)
        if assistant_parts:
            return assistant_parts[-1]
        # Multimodal content blocks or an arbitrary list of text fragments.
        parts: list[str] = []
        for entry in value:
            if isinstance(entry, dict) and entry.get("type") in {"text", "output_text"}:
                text = content_to_text(entry.get("text"))
            else:
                text = content_to_text(entry)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    text = content_to_text(message.get("content"))
                    if text:
                        return text
                text = content_to_text(first.get("text"))
                if text:
                    return text
        for field in RESPONSE_FIELDS + ("content", "text", "value"):
            if field in value:
                text = content_to_text(value[field])
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_rubrics(value: Any, *, line_no: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"line {line_no}: 'rubrics' must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for idx, rubric in enumerate(value):
        if isinstance(rubric, str):
            normalized.append(
                {
                    "rubric_id": idx,
                    "rubric_text": rubric.strip(),
                    "must_have": False,
                }
            )
            continue
        if not isinstance(rubric, dict):
            raise ValueError(f"line {line_no}: rubric {idx} must be an object or string")
        text = first_present(rubric, ("rubric_text", "text", "criterion", "description"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"line {line_no}: rubric {idx} has no rubric_text")
        item = dict(rubric)
        item.setdefault("rubric_id", idx)
        item["rubric_text"] = text.strip()
        item["must_have"] = bool(item.get("must_have", False))
        normalized.append(item)
    return normalized


def record_to_item(
    record: dict[str, Any],
    *,
    line_no: int,
    response_field: str | None,
) -> EvalItem:
    query_value = first_present(record, QUESTION_FIELDS)
    if not isinstance(query_value, str) or not query_value.strip():
        raise ValueError(
            f"line {line_no}: query is missing; expected one of {', '.join(QUESTION_FIELDS)}"
        )

    if response_field:
        response_value = nested_get(record, response_field)
    else:
        response_value = first_present(record, RESPONSE_FIELDS)
        if response_value is None and isinstance(record.get("messages"), list):
            response_value = record["messages"]

    query_id_value = first_present(record, ID_FIELDS)
    query_id = str(query_id_value) if query_id_value is not None else f"line-{line_no}"
    query_date = content_to_text(record.get("query_date") or record.get("date"))
    rubrics = normalize_rubrics(record.get("rubrics") or record.get("Rubric"), line_no=line_no)

    return EvalItem(
        source_line=line_no,
        query_id=query_id,
        query=query_value.strip(),
        query_date=query_date,
        response=content_to_text(response_value),
        rubrics=rubrics,
    )


def load_input(path: Path, response_field: str | None) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: each JSONL line must be an object")
            items.append(
                record_to_item(record, line_no=line_no, response_field=response_field)
            )
    if not items:
        raise ValueError(f"input contains no records: {path}")
    return items


def build_user_prompt(item: EvalItem, batch: list[dict[str, Any]]) -> str:
    criteria = "\n\n".join(
        f"{idx}. {rubric['rubric_text']}" for idx, rubric in enumerate(batch)
    )
    return USER_PROMPT_TEMPLATE.format(
        query=item.query,
        query_date=item.query_date,
        response=item.response,
        criteria=criteria,
    )


def parse_judgements(text: str) -> dict[str, dict[str, Any]] | None:
    """Parse a JSON object from a plain or fenced judge response."""
    if not text:
        return None
    for match in JSON_BLOCK_RE.findall(text):
        try:
            parsed = json.loads(match.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return None


def coerce_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if value in (0, 1):
        return bool(value)
    return False


def extract_message_text(message: Any) -> str:
    """Read final or reasoning text from OpenAI/vLLM response messages."""
    candidates = [
        getattr(message, "content", None),
        getattr(message, "reasoning_content", None),
        getattr(message, "reasoning", None),
    ]
    for candidate in candidates:
        text = content_to_text(candidate)
        if text:
            return text
    return ""


class OpenAICompatibleGrader:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model_name: str,
        max_rubrics_per_call: int,
        max_tokens: int,
        temperature: float,
        max_json_parse_retries: int,
        enable_thinking: bool | None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_rubrics_per_call = max_rubrics_per_call
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_json_parse_retries = max_json_parse_retries
        self.enable_thinking = enable_thinking

    async def judge_batch(
        self, item: EvalItem, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
        user_prompt = build_user_prompt(item, batch)
        last_error: str | None = None

        for attempt in range(self.max_json_parse_retries + 1):
            prompt = user_prompt if attempt == 0 else user_prompt + JSON_FORMAT_RETRY_SUFFIX
            request: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.enable_thinking is not None:
                request["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                }
            try:
                completion = await self.client.chat.completions.create(**request)
                raw_text = extract_message_text(completion.choices[0].message)
            except Exception as exc:  # keep one failed query from terminating the run
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "judge request failed query=%s attempt=%d/%d: %s",
                    item.query_id,
                    attempt + 1,
                    self.max_json_parse_retries + 1,
                    last_error,
                )
                continue

            parsed = parse_judgements(raw_text)
            if parsed is not None:
                return parsed, None
            last_error = f"unparseable judge response: {raw_text[:300]!r}"
            LOGGER.warning(
                "unparseable JSON query=%s attempt=%d/%d",
                item.query_id,
                attempt + 1,
                self.max_json_parse_retries + 1,
            )
        return None, last_error

    async def grade(self, item: EvalItem) -> dict[str, Any]:
        base_result: dict[str, Any] = {
            "source_line": item.source_line,
            "query_id": item.query_id,
            "query": item.query,
            "query_date": item.query_date,
            "system_response": item.response,
            "judge_model": self.model_name,
        }

        if not item.response:
            return self._finish_result(
                base_result,
                item.rubrics,
                labels=[False] * len(item.rubrics),
                reasons=["No system response was supplied."] * len(item.rubrics),
                failed=True,
                failure_reason="no_response",
                error=None,
            )

        labels: list[bool] = []
        reasons: list[str] = []
        for start in range(0, len(item.rubrics), self.max_rubrics_per_call):
            batch = item.rubrics[start : start + self.max_rubrics_per_call]
            judgements, error = await self.judge_batch(item, batch)
            if judgements is None:
                return self._finish_result(
                    base_result,
                    item.rubrics,
                    labels=None,
                    reasons=["Judge did not return a valid decision."] * len(item.rubrics),
                    failed=True,
                    failure_reason="judge_error",
                    error=error,
                )
            for idx in range(len(batch)):
                entry = judgements.get(str(idx), {})
                if not isinstance(entry, dict):
                    entry = {}
                labels.append(coerce_label(entry.get("label", False)))
                reasons.append(content_to_text(entry.get("reason")))

        return self._finish_result(
            base_result,
            item.rubrics,
            labels=labels,
            reasons=reasons,
            failed=False,
            failure_reason=None,
            error=None,
        )

    @staticmethod
    def _finish_result(
        base_result: dict[str, Any],
        rubrics: list[dict[str, Any]],
        *,
        labels: list[bool] | None,
        reasons: list[str],
        failed: bool,
        failure_reason: str | None,
        error: str | None,
    ) -> dict[str, Any]:
        rubric_results: list[dict[str, Any]] = []
        result_labels: list[bool | None]
        if labels is None:
            result_labels = [None] * len(rubrics)
        else:
            result_labels = list(labels)
        for rubric, label, reason in zip(rubrics, result_labels, reasons, strict=True):
            result = dict(rubric)
            result["label"] = label
            result["reason"] = reason
            rubric_results.append(result)

        num_rubrics = len(rubric_results)
        num_qualified = sum(bool(r["label"]) for r in rubric_results)
        must_have = [r for r in rubric_results if r.get("must_have", False)]
        num_must_have_qualified = sum(bool(r["label"]) for r in must_have)
        # Upstream FrontierFinance excludes grader-side judge errors from scored
        # metrics, while a missing system response remains a system failure (0).
        if failure_reason == "judge_error":
            qualification_rate = None
            must_have_rate = None
        else:
            qualification_rate = num_qualified / num_rubrics if num_rubrics else 0.0
            must_have_rate = (
                num_must_have_qualified / len(must_have) if must_have else None
            )

        return {
            **base_result,
            "failed": failed,
            "failure_reason": failure_reason,
            "error": error,
            "num_rubrics": num_rubrics,
            "num_qualified": num_qualified,
            "qualification_rate": qualification_rate,
            "score": qualification_rate,
            "num_must_have_rubrics": len(must_have),
            "num_must_have_qualified": num_must_have_qualified,
            "must_have_qualification_rate": must_have_rate,
            "labels": labels or [],
            "rubric_results": rubric_results,
        }


def completed_source_lines(output_path: Path) -> set[int]:
    completed: set[int] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for output_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"cannot resume: output line {output_line} is invalid JSON: {exc}"
                ) from exc
            source_line = record.get("source_line")
            if isinstance(source_line, int):
                completed.add(source_line)
    return completed


def parse_enable_thinking(value: str) -> bool | None:
    lowered = value.lower()
    if lowered == "auto":
        return None
    return lowered == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grade JSONL responses against FrontierFinance rubrics using an "
            "OpenAI-compatible judge endpoint."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base URL, normally ending in /v1.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument("--model-name", required=True, help="Judge model name.")
    parser.add_argument(
        "--response-field",
        default=None,
        help="Response field name, including a dot-separated nested path if needed.",
    )
    parser.add_argument("--max-rubrics-per-call", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--max-json-parse-retries", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        choices=("auto", "true", "false"),
        default="auto",
        help="Pass vLLM chat_template_kwargs.enable_thinking; auto sends no override.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Append while skipping source lines already present in the output.",
    )
    mode.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> str:
    if not args.input.is_file():
        raise ValueError(f"input file does not exist: {args.input}")
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("input and output must be different files")
    except FileNotFoundError:
        pass
    if args.max_rubrics_per_call <= 0:
        raise ValueError("--max-rubrics-per-call must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.request_retries < 0 or args.max_json_parse_retries < 0:
        raise ValueError("retry counts must be non-negative")
    if args.output.exists() and not (args.resume or args.overwrite):
        raise ValueError(
            f"output already exists: {args.output}; use --resume or --overwrite"
        )
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("provide --api-key or set OPENAI_API_KEY")
    return api_key


async def run(args: argparse.Namespace, api_key: str) -> int:
    items = load_input(args.input, args.response_field)
    done = completed_source_lines(args.output) if args.resume else set()
    pending = [item for item in items if item.source_line not in done]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume else "w"
    LOGGER.info(
        "loaded=%d pending=%d skipped=%d output=%s",
        len(items),
        len(pending),
        len(items) - len(pending),
        args.output,
    )
    if not pending:
        LOGGER.info("nothing to grade")
        return 0

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        max_retries=args.request_retries,
    )
    grader = OpenAICompatibleGrader(
        client=client,
        model_name=args.model_name,
        max_rubrics_per_call=args.max_rubrics_per_call,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_json_parse_retries=args.max_json_parse_retries,
        enable_thinking=parse_enable_thinking(args.enable_thinking),
    )

    processed = failed = judge_errors = qualified = rubrics = 0
    try:
        with args.output.open(output_mode, encoding="utf-8") as output_handle:
            # Process bounded groups so the result order is stable and interruption
            # loses at most one small in-flight group.
            for start in range(0, len(pending), args.concurrency):
                group = pending[start : start + args.concurrency]
                results = await asyncio.gather(*(grader.grade(item) for item in group))
                for result in results:
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    processed += 1
                    failed += int(result["failed"])
                    if result["failure_reason"] == "judge_error":
                        judge_errors += 1
                    else:
                        qualified += int(result["num_qualified"])
                        rubrics += int(result["num_rubrics"])
                LOGGER.info(
                    "progress=%d/%d failed=%d judge_errors=%d qualification_rate=%.2f%%",
                    processed,
                    len(pending),
                    failed,
                    judge_errors,
                    100.0 * qualified / rubrics if rubrics else 0.0,
                )
    finally:
        await client.close()

    LOGGER.info(
        "done records=%d failed=%d judge_errors=%d qualified=%d/%d (%.2f%%)",
        processed,
        failed,
        judge_errors,
        qualified,
        rubrics,
        100.0 * qualified / rubrics if rubrics else 0.0,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        api_key = validate_args(args)
        return asyncio.run(run(args, api_key))
    except (ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("interrupted; rerun with --resume to continue")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
