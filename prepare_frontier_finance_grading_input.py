#!/usr/bin/env python3
"""Join finance-agent trace results with FrontierFinance rubric metadata.

The generated JSONL can be passed directly to ``grading.py``.  The script uses
only the Python standard library, including a small read-only XLSX reader, so it
does not require pandas or openpyxl.

Expected trace layout::

    TRACE_ROOT/<run-id>/q001/result.json
    TRACE_ROOT/<run-id>/q002/result.json

Each XLSX metadata row is paired with ``q001``, ``q002`` ... in worksheet order.
When ``--reference-jsonl`` is supplied, authoritative ``query_id``,
``query_date``, rubric, use-case and capability metadata are recovered by an
exact normalized-question match.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


LOGGER = logging.getLogger("prepare_frontier_finance_grading_input")

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

QUESTION_ID_RE = re.compile(r"^q(\d+)$", re.IGNORECASE)
CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$", re.IGNORECASE)


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u00a0", " ")
    return " ".join(value.split()).strip().casefold()


def excel_column_index(cell_ref: str) -> int:
    match = CELL_REF_RE.match(cell_ref)
    if not match:
        raise ValueError(f"invalid XLSX cell reference: {cell_ref}")
    result = 0
    for char in match.group(1).upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def xml_text(element: ET.Element) -> str:
    return "".join(element.itertext())


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(path))
    return [xml_text(item) for item in root.findall(f"{{{MAIN_NS}}}si")]


def resolve_sheet_path(
    archive: zipfile.ZipFile, sheet_name: str | None
) -> tuple[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    relationship_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
        if rel.attrib.get("Id") and rel.attrib.get("Target")
    }

    sheets_node = workbook.find(f"{{{MAIN_NS}}}sheets")
    if sheets_node is None:
        raise ValueError("XLSX workbook has no worksheets")

    sheets = list(sheets_node.findall(f"{{{MAIN_NS}}}sheet"))
    if not sheets:
        raise ValueError("XLSX workbook has no worksheets")

    selected = None
    if sheet_name is None:
        selected = sheets[0]
    else:
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                selected = sheet
                break
        if selected is None:
            available = ", ".join(sheet.attrib.get("name", "") for sheet in sheets)
            raise ValueError(
                f"worksheet {sheet_name!r} not found; available: {available}"
            )

    selected_name = selected.attrib.get("name", "")
    relationship_id = selected.attrib.get(f"{{{REL_NS}}}id")
    target = relationship_targets.get(relationship_id or "")
    if not target:
        raise ValueError(f"cannot resolve worksheet relationship for {selected_name!r}")

    target = target.lstrip("/")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return selected_name, target


def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        return xml_text(inline) if inline is not None else ""

    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    value = value_node.text

    if cell_type == "s":
        index = int(value)
        if index >= len(shared_strings):
            raise ValueError(f"shared string index out of range: {index}")
        return shared_strings[index]
    if cell_type == "b":
        return value == "1"
    if cell_type in {"str", "e"}:
        return value

    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def read_xlsx_records(
    path: Path, sheet_name: str | None
) -> tuple[str, list[dict[str, Any]]]:
    """Read the first/headered worksheet without third-party dependencies."""
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        selected_sheet, sheet_path = resolve_sheet_path(archive, sheet_name)
        root = ET.fromstring(archive.read(sheet_path))

    sheet_data = root.find(f"{{{MAIN_NS}}}sheetData")
    if sheet_data is None:
        raise ValueError(f"worksheet {selected_sheet!r} contains no sheetData")

    rows: list[dict[int, Any]] = []
    for row in sheet_data.findall(f"{{{MAIN_NS}}}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            ref = cell.attrib.get("r")
            if not ref:
                continue
            value = read_cell_value(cell, shared_strings)
            if value is not None:
                values[excel_column_index(ref)] = value
        if values:
            rows.append(values)

    if not rows:
        raise ValueError(f"worksheet {selected_sheet!r} is empty")

    headers = {
        column: str(value).strip()
        for column, value in rows[0].items()
        if value is not None and str(value).strip()
    }
    records: list[dict[str, Any]] = []
    for values in rows[1:]:
        record = {
            header: values.get(column)
            for column, header in headers.items()
            if values.get(column) is not None
        }
        if any(str(value).strip() for value in record.values() if value is not None):
            records.append(record)
    return selected_sheet, records


def case_insensitive_get(
    record: dict[str, Any], candidates: tuple[str, ...]
) -> Any:
    lookup = {key.strip().casefold(): value for key, value in record.items()}
    for candidate in candidates:
        key = candidate.casefold()
        if key in lookup:
            return lookup[key]
    return None


def parse_rubrics(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"rubric cell is not valid JSON: {exc}") from exc
    else:
        raise ValueError("rubric cell is empty")

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("rubrics must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for index, rubric in enumerate(parsed):
        if isinstance(rubric, str):
            rubric = {"rubric_id": index + 1, "rubric_text": rubric}
        if not isinstance(rubric, dict):
            raise ValueError(f"rubric {index} is not an object")
        text = rubric.get("rubric_text") or rubric.get("criteria")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"rubric {index} has no rubric_text or criteria")
        item = dict(rubric)
        item.setdefault("rubric_id", index + 1)
        item["rubric_text"] = text.strip()
        item["must_have"] = bool(item.get("must_have", False))
        normalized.append(item)
    return normalized


def read_csv_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read UTF-8 CSV metadata while preserving every source column."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        records = [dict(row) for row in reader]
    return path.name, records


def load_reference_jsonl(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"reference line {line_no} is invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("query"), str):
                raise ValueError(f"reference line {line_no} has no string query")
            key = normalize_question(record["query"])
            if key in index:
                raise ValueError(
                    f"duplicate normalized question in reference: {record['query']!r}"
                )
            index[key] = record
    return index


def load_questions_file(path: Path | None) -> list[str] | None:
    """Load the exact rollout order for a subset run."""
    if path is None:
        return None
    questions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise ValueError(f"questions file contains no non-empty lines: {path}")
    return questions


def discover_trace_results(trace_root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in trace_root.glob("*/q*/result.json"):
        question_id = path.parent.name.lower()
        if not QUESTION_ID_RE.fullmatch(question_id):
            continue
        found.setdefault(question_id, []).append(path)
    return found


def select_trace(paths: list[Path], duplicate_policy: str) -> Path:
    if len(paths) == 1:
        return paths[0]
    if duplicate_policy == "error":
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"multiple trace results found: {joined}")
    selected = max(
        paths,
        key=lambda path: (path.parent.parent.name, path.stat().st_mtime_ns),
    )
    LOGGER.warning(
        "multiple results for %s; selected latest %s", selected.parent.name, selected
    )
    return selected


def stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [stringify_answer(part) for part in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("final_answer", "answer", "response", "content", "text", "output"):
            if key in value:
                answer = stringify_answer(value[key])
                if answer:
                    return answer
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def extract_final_answer(result: dict[str, Any]) -> str:
    for key in ("final_answer", "answer", "system_response", "response", "output"):
        if key in result:
            answer = stringify_answer(result[key])
            if answer:
                return answer
    nested = result.get("result")
    if nested is not None:
        return stringify_answer(nested)
    return ""


def json_safe_error(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def rejected_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}.rejected{output.suffix}")
    return output.with_name(f"{output.name}.rejected.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join finance-agent final answers with XLSX/CSV rubric metadata."
    )
    parser.add_argument("--traces", required=True, type=Path, help="Trace model root.")
    parser.add_argument(
        "--metadata", required=True, type=Path, help="Metadata XLSX or CSV."
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL.")
    parser.add_argument(
        "--sheet", default=None, help="Worksheet name; default is first sheet."
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional authoritative FrontierFinance JSONL used to recover query_id, "
            "query_date, and rubrics by exact normalized question match."
        ),
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help=(
            "Optional TXT file containing the questions in rollout order. Use this "
            "for subset/rerun traces so q001 maps to line 1 of that TXT instead of "
            "row 1 of the full metadata workbook."
        ),
    )
    parser.add_argument(
        "--duplicate-policy", choices=("latest", "error"), default="latest"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail without writing output if any metadata row or trace is invalid/missing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    if not args.traces.is_dir():
        raise ValueError(f"trace directory does not exist: {args.traces}")
    if not args.metadata.is_file():
        raise ValueError(f"metadata file does not exist: {args.metadata}")
    if args.reference_jsonl is not None and not args.reference_jsonl.is_file():
        raise ValueError(f"reference JSONL does not exist: {args.reference_jsonl}")
    if args.questions_file is not None and not args.questions_file.is_file():
        raise ValueError(f"questions file does not exist: {args.questions_file}")
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"output already exists: {args.output}; use --overwrite")


def convert(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.metadata.suffix.casefold() == ".csv":
        if args.sheet is not None:
            raise ValueError("--sheet cannot be used with CSV metadata")
        sheet_name, metadata_rows = read_csv_records(args.metadata)
    else:
        sheet_name, metadata_rows = read_xlsx_records(args.metadata, args.sheet)
    references = load_reference_jsonl(args.reference_jsonl)
    rollout_questions = load_questions_file(args.questions_file)
    traces = discover_trace_results(args.traces)
    LOGGER.info(
        "metadata_rows=%d sheet=%s trace_ids=%d references=%d",
        len(metadata_rows),
        sheet_name,
        len(traces),
        len(references),
    )

    outputs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    metadata_trace_ids: set[str] = set()
    reference_matches = 0

    metadata_by_question: dict[str, tuple[int, dict[str, Any]]] = {}
    for metadata_index, metadata in enumerate(metadata_rows, 1):
        metadata_question = stringify_answer(
            case_insensitive_get(metadata, ("Question", "query", "prompt"))
        )
        if metadata_question:
            key = normalize_question(metadata_question)
            if key in metadata_by_question:
                raise ValueError(
                    f"duplicate normalized question in metadata: {metadata_question!r}"
                )
            metadata_by_question[key] = (metadata_index, metadata)

    if rollout_questions is None:
        work_items: list[tuple[int, int | None, dict[str, Any], str | None]] = [
            (index, index, metadata, None)
            for index, metadata in enumerate(metadata_rows, 1)
        ]
    else:
        work_items = []
        for trace_index, question in enumerate(rollout_questions, 1):
            matched = metadata_by_question.get(normalize_question(question))
            if matched is None:
                work_items.append((trace_index, None, {"Question": question}, question))
            else:
                metadata_index, metadata = matched
                work_items.append((trace_index, metadata_index, metadata, question))

    for trace_index, metadata_index, metadata, question_override in work_items:
        trace_question_id = f"q{trace_index:03d}"
        metadata_trace_ids.add(trace_question_id)
        question_value = case_insensitive_get(metadata, ("Question", "query", "prompt"))
        question = question_override or stringify_answer(question_value)
        reject_base = {
            "metadata_row": metadata_index + 1 if metadata_index is not None else None,
            "trace_question_id": trace_question_id,
            "question": question,
        }

        if not question:
            rejected.append({**reject_base, "reason": "missing question"})
            continue
        if metadata_index is None:
            rejected.append(
                {**reject_base, "reason": "question not found in metadata workbook"}
            )
            continue

        reference = references.get(normalize_question(question))
        try:
            if reference is not None:
                reference_matches += 1
                rubrics = parse_rubrics(reference.get("rubrics"))
                query_id = str(reference.get("query_id") or trace_question_id)
                query_date = stringify_answer(reference.get("query_date"))
                use_cases = reference.get("use_cases")
                capabilities = reference.get("capabilities")
            else:
                rubrics = parse_rubrics(
                    case_insensitive_get(metadata, ("Rubric", "rubrics", "criteria"))
                )
                query_id_value = case_insensitive_get(
                    metadata, ("query_id", "question_id", "id")
                )
                query_id = stringify_answer(query_id_value) or trace_question_id
                query_date = stringify_answer(
                    case_insensitive_get(metadata, ("query_date", "date"))
                )
                use_cases = None
                capabilities = None
        except ValueError as exc:
            rejected.append({**reject_base, "reason": f"invalid metadata: {exc}"})
            continue

        paths = traces.get(trace_question_id, [])
        if not paths:
            rejected.append({**reject_base, "reason": "trace result.json not found"})
            continue

        try:
            trace_path = select_trace(paths, args.duplicate_policy)
            result = json.loads(trace_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                rejected.append(
                    {**reject_base, "reason": "trace result is not an object"}
                )
                continue
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rejected.append({**reject_base, "reason": f"invalid trace: {exc}"})
            continue

        output = {
            "query_id": query_id,
            "trace_question_id": trace_question_id,
            "query": question,
            "query_date": query_date,
            "final_answer": extract_final_answer(result),
            "rubrics": rubrics,
            "trace_metadata": {
                "source_result": str(trace_path.resolve()),
                "run_id": trace_path.parent.parent.name,
                "success": bool(result.get("success", False)),
                "stop_reason": result.get("stop_reason"),
                "final_error": json_safe_error(result.get("final_error")),
                "total_turns": result.get("total_turns"),
                "tool_calls_count": result.get("tool_calls_count"),
            },
        }
        if use_cases is not None:
            output["use_cases"] = use_cases
        if capabilities is not None:
            output["capabilities"] = capabilities
        outputs.append(output)

    unused_trace_ids = sorted(set(traces) - metadata_trace_ids)
    for trace_id in unused_trace_ids:
        rejected.append(
            {
                "trace_question_id": trace_id,
                "reason": "trace exists but was not joined to a valid metadata row",
                "paths": [str(path) for path in traces[trace_id]],
            }
        )

    if args.strict and rejected:
        examples = "; ".join(
            f"{item.get('trace_question_id', '?')}: {item['reason']}"
            for item in rejected[:5]
        )
        raise ValueError(
            f"strict mode rejected {len(rejected)} records: {examples}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reject_file = rejected_path(args.output)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in outputs:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    with reject_file.open("w", encoding="utf-8") as handle:
        for record in rejected:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    no_response = sum(not record["final_answer"] for record in outputs)
    LOGGER.info(
        "wrote=%d rejected=%d no_response=%d reference_matches=%d "
        "output=%s rejected_file=%s",
        len(outputs),
        len(rejected),
        no_response,
        reference_matches,
        args.output,
        reject_file,
    )
    return len(outputs), len(rejected), no_response


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        validate_paths(args)
        convert(args)
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
