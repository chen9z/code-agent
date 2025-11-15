#!/usr/bin/env python3
"""Benchmark runner for Code Agent scenarios defined via JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from code_agent import CodeAgentSession


@dataclass
class ValidatorResult:
    name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    name: str
    success: bool
    duration_sec: float
    tool_calls: int
    tools_used: List[str]
    content: str
    transcript_path: Optional[str]
    validators: List[ValidatorResult]


@contextmanager
def pushd(path: Path) -> Iterable[None]:
    original = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


def load_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in config file {path}: {exc}")


def ensure_directory(path: Optional[Path]) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def save_transcript(transcript: Sequence[str], output_dir: Optional[Path], scenario_name: str) -> Optional[str]:
    if not transcript or output_dir is None:
        return None
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", scenario_name)
    log_path = output_dir / f"{safe_name}_transcript.txt"
    log_path.write_text("\n".join(transcript), encoding="utf-8")
    return str(log_path)


def extract_tool_names(tool_results: Optional[Sequence[Any]]) -> List[str]:
    if not tool_results:
        return []
    names = []
    for entry in tool_results:
        key: Optional[str] = None
        if hasattr(entry, "tool_call"):
            tool_call = getattr(entry, "tool_call")
            key = getattr(tool_call, "name", None)
        elif hasattr(entry, "name"):
            key = getattr(entry, "name")
        elif hasattr(entry, "key"):
            key = getattr(entry, "key")
        elif isinstance(entry, Mapping):
            key = entry.get("name") or entry.get("key")
        else:
            key = None
        if isinstance(key, str) and key:
            names.append(key)
    return names


def run_validators(
    validators: Sequence[Mapping[str, Any]],
    *,
    content: str,
    transcript: Sequence[str],
) -> List[ValidatorResult]:
    results: List[ValidatorResult] = []
    for idx, validator in enumerate(validators or []):
        vtype = validator.get("type")
        name = validator.get("name") or f"validator_{idx}"
        if vtype == "contains":
            values = [str(item) for item in validator.get("values", [])]
            missing = [value for value in values if value not in content]
            status = "pass" if not missing else "fail"
            results.append(
                ValidatorResult(name=name, status=status, details={"missing": missing})
            )
        elif vtype == "regex":
            pattern = re.compile(str(validator.get("pattern")))
            match = bool(pattern.search(content))
            status = "pass" if match else "fail"
            results.append(
                ValidatorResult(name=name, status=status, details={"matched": match})
            )
        elif vtype == "regex_set":
            pattern = re.compile(str(validator.get("pattern")))
            expected = {str(item) for item in validator.get("expected", [])}
            matches = {match if isinstance(match, str) else match[0] for match in pattern.findall(content)}
            ignore = {str(item) for item in validator.get("ignore", [])}
            matches -= ignore
            missing = sorted(expected - matches)
            extra = sorted(matches - expected)
            status = "pass" if not missing and not extra else "fail"
            results.append(
                ValidatorResult(
                    name=name,
                    status=status,
                    details={
                        "missing": missing,
                        "extra": extra,
                        "matches": sorted(matches),
                        "ignored": sorted(ignore) if ignore else [],
                    },
                )
            )
        elif vtype == "transcript_contains":
            values = [str(item) for item in validator.get("values", [])]
            missing = [value for value in values if not any(value in line for line in transcript)]
            status = "pass" if not missing else "fail"
            results.append(
                ValidatorResult(name=name, status=status, details={"missing": missing})
            )
        else:
            results.append(
                ValidatorResult(
                    name=name,
                    status="fail",
                    details={"error": f"unsupported validator type: {vtype}"},
                )
            )
    return results


def run_scenario(
    scenario: Mapping[str, Any],
    *,
    default_max_iterations: int,
    transcript_dir: Optional[Path],
) -> ScenarioResult:
    name = str(scenario.get("name") or scenario.get("prompt") or "scenario")
    workspace = Path(str(scenario["workspace"])).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise SystemExit(f"Workspace for scenario '{name}' is invalid: {workspace}")
    prompt = str(scenario["prompt"])
    max_iterations = int(scenario.get("max_iterations", default_max_iterations))

    session = CodeAgentSession(max_iterations=max_iterations)
    transcript: List[str] = []

    start = time.perf_counter()
    with pushd(workspace):
        result = session.run_turn(prompt, output_callback=transcript.append)
    duration = time.perf_counter() - start

    content = str(result.get("content") or (result.get("tool_plan") or {}).get("content") or "")
    tool_results = result.get("tool_results") or []
    tool_calls = len(tool_results)
    tools_used = extract_tool_names(tool_results)

    validators = run_validators(
        scenario.get("validators", []),
        content=content,
        transcript=transcript,
    )
    success = all(v.status == "pass" for v in validators)
    transcript_path = save_transcript(transcript, transcript_dir, name)

    return ScenarioResult(
        name=name,
        success=success,
        duration_sec=duration,
        tool_calls=tool_calls,
        tools_used=tools_used,
        content=content,
        transcript_path=transcript_path,
        validators=validators,
    )


def summarise(results: Sequence[ScenarioResult]) -> Dict[str, Any]:
    passed = sum(1 for item in results if item.success)
    total = len(results)
    total_duration = sum(item.duration_sec for item in results)
    return {
        "scenarios": [
            {
                "name": item.name,
                "success": item.success,
                "duration_sec": round(item.duration_sec, 3),
                "tool_calls": item.tool_calls,
                "tools_used": item.tools_used,
                "content": item.content,
                "transcript_path": item.transcript_path,
                "validators": [
                    {
                        "name": validator.name,
                        "status": validator.status,
                        "details": validator.details,
                    }
                    for validator in item.validators
                ],
            }
            for item in results
        ],
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": (passed / total) if total else 0.0,
            "total_duration_sec": round(total_duration, 3),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark scenarios against the Code Agent")
    parser.add_argument("--config", required=True, help="Path to JSON config file containing scenarios")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Default max iterations per scenario (overridden by scenario entries)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write benchmark results as JSON",
    )
    parser.add_argument(
        "--transcript-dir",
        help="Directory to store per-scenario transcripts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit("Config file must define a non-empty 'scenarios' list")

    transcript_dir = Path(args.transcript_dir).expanduser() if args.transcript_dir else None
    ensure_directory(transcript_dir)

    results: List[ScenarioResult] = []
    for scenario in scenarios:
        results.append(
            run_scenario(
                scenario,
                default_max_iterations=args.max_iterations,
                transcript_dir=transcript_dir,
            )
        )

    payload = summarise(results)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Benchmark results written to {output_path}")

    summary = payload["summary"]
    print(f"Scenarios passed: {summary['passed']}/{summary['total']} (pass rate {summary['pass_rate']:.0%})")
    print(f"Total duration: {summary['total_duration_sec']}s")

    for item in payload["scenarios"]:
        status = "PASS" if item["success"] else "FAIL"
        print(f"- {item['name']}: {status} | {item['tool_calls']} tool calls | {item['duration_sec']}s")
        for validator in item["validators"]:
            print(f"    • {validator['name']}: {validator['status']} {validator['details']}")

    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
