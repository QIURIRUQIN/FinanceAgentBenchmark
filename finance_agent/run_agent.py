import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from model_library.agent import AgentResult
from model_library.base import LLMConfig
from model_library.base.input import TextInput
from tqdm.asyncio import tqdm

from .get_agent import Parameters, get_agent
from .prompt import INSTRUCTIONS_PROMPT
from .tools import VALID_TOOLS


QUESTION_ID_RE = re.compile(r"q\d{3}")


async def run_tests_parallel(
    questions: list[str],
    max_concurrent: int,
    parameters: Parameters,
    question_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run multiple questions in parallel using the agent"""
    if question_ids is None:
        question_ids = [f"q{i:03d}" for i in range(1, len(questions) + 1)]
    if len(question_ids) != len(questions):
        raise ValueError("question_ids and questions must have the same length")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question_ids must be unique")
    invalid_ids = [qid for qid in question_ids if not QUESTION_ID_RE.fullmatch(qid)]
    if invalid_ids:
        raise ValueError(f"invalid question IDs: {invalid_ids}")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question: str, question_id: str):
        async with semaphore:
            agent = get_agent(parameters)
            prompt = INSTRUCTIONS_PROMPT.format(question=question)
            result = await agent.run([TextInput(text=prompt)], question_id=question_id)
            return result

    tasks = [
        process_question(question, question_id)
        for question_id, question in zip(question_ids, questions)
    ]

    results: list[AgentResult] = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for question_id, question, result in zip(question_ids, questions, results):
        if isinstance(result, Exception):
            formatted_results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "success": False,
                    "error": str(result),
                }
            )
            print(f"\nFAIL {question_id} failed: {question}\n   Error: {result}\n")
        else:
            formatted_results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "success": result.success,
                    "result": result.model_dump(mode="json"),
                }
            )
            if not result.success and result.final_error:
                print(
                    f"\nFAIL {question_id} failed: {question}\n   Turns: {result.total_turns}\n   Error: [{result.final_error.type}] {result.final_error.message}\n"
                )
            else:
                print(
                    f"\nOK {question_id} succeeded: {question}\n   Turns: {result.total_turns}\n   Result: {result.final_answer}\n"
                )

    # Write results next to agent logs (use first result's output_dir parent)
    non_error_results = [r for r in results if not isinstance(r, Exception)]
    if non_error_results:
        results_dir = non_error_results[0].output_dir.parent
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(formatted_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return formatted_results


async def main():
    parser = argparse.ArgumentParser(description="Run the harness for the finance agent benchmark")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum number of tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model generation",
    )
    parser.add_argument("--questions", type=str, nargs="+", help="List of questions to process")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="Model to use to generate completions",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help=(
            "Path to a question file. Each non-empty line may be plain question "
            "text, or an explicit-ID row formatted as qNNN<TAB>question. "
            "Explicit IDs preserve the original q*** trace directory names."
        ),
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=VALID_TOOLS,
        choices=VALID_TOOLS,
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns for the agent to take before stopping",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel requests to make to the model",
    )
    args = parser.parse_args()

    # Path 可以将文件地址变为一个可操作的对象，使用 is_exists()、is_file() 等方法
    ENV_FILE = Path(".env")
    load_dotenv(override=True, dotenv_path=ENV_FILE)

    # 观察是多少个问题
    question_ids: list[str] | None = None
    if args.question_file:
        entries: list[tuple[str | None, str]] = []
        with open(args.question_file, encoding="utf-8") as f:
            for file_line_number, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                if "\t" in stripped:
                    question_id, question = stripped.split("\t", 1)
                    question_id = question_id.strip().lower()
                    question = question.strip()
                    if not QUESTION_ID_RE.fullmatch(question_id):
                        raise ValueError(
                            f"Invalid explicit question ID at line {file_line_number}: "
                            f"{question_id!r}"
                        )
                    if not question:
                        raise ValueError(
                            f"Missing question text at line {file_line_number}"
                        )
                    entries.append((question_id, question))
                else:
                    entries.append((None, stripped))

        if not entries:
            raise ValueError(f"No questions found in: {args.question_file}")
        explicit_count = sum(question_id is not None for question_id, _ in entries)
        if explicit_count not in {0, len(entries)}:
            raise ValueError(
                "Question file mixes explicit qNNN<TAB>question rows with plain "
                "questions; use one format consistently."
            )
        questions = [question for _, question in entries]
        if explicit_count:
            question_ids = [question_id for question_id, _ in entries if question_id]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception("No questions provided. One of --question-file or --questions must be used.")

    parameters = Parameters(
        model_name=args.model,
        max_turns=args.max_turns,
        tools=args.tools,
        llm_config=LLMConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    await run_tests_parallel(
        questions=questions,
        max_concurrent=args.parallelism,
        parameters=parameters,
        question_ids=question_ids,
    )

def main_sync():
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()
