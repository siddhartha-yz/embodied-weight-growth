#!/usr/bin/env python3
"""Make one structured GPT-6 Astra call and record the proposed numeric edits."""

from __future__ import annotations

import os
import shutil
import json
import math
from pathlib import Path
import sys
import tempfile


MODEL = "gpt-6-astra"
EFFORT = "high"
PASSIVE_ITEMS = {"userMessage", "reasoning", "agentMessage", "plan"}
BASE = (
    "This is a one-shot controller-edit experiment. Use only the supplied evidence "
    "and instructions. Do not use tools, write code, or claim validation. "
    "Return a structured proposal containing direct numeric edits and a short diagnosis."
)


def object_schema(properties):
    return {
        "type": "object", "properties": properties, "required": list(properties),
        "additionalProperties": False,
    }


SCHEMA = object_schema({
    "edits": {
        "type": "array",
        "items": object_schema({
            "unit": {"type": "integer"},
            "coefficient": {"type": "string"},
            "value": {"type": "number"},
        }),
    },
    "diagnosis": {"type": "string"},
})


def parse_answer(raw):
    answer = json.loads(raw)
    if not isinstance(answer, dict) or answer.keys() != SCHEMA["properties"].keys():
        raise ValueError("expected edits and diagnosis")
    if not isinstance(answer["edits"], list) or not isinstance(answer["diagnosis"], str):
        raise ValueError("edits must be a list and diagnosis must be text")
    for edit in answer["edits"]:
        if not isinstance(edit, dict) or edit.keys() != {"unit", "coefficient", "value"}:
            raise ValueError("each edit requires unit, coefficient and value")
        if type(edit["unit"]) is not int:
            raise ValueError("unit must be an integer")
        if not isinstance(edit["coefficient"], str) or not edit["coefficient"].strip():
            raise ValueError("coefficient must name a policy key")
        if type(edit["value"]) not in (int, float) or not math.isfinite(edit["value"]):
            raise ValueError("value must be finite numeric data")
    return answer


def import_sdk():
    if sdk_dir := os.environ.get("CODEX_SDK_DIR"):
        sys.path.insert(0, str(Path(sdk_dir).expanduser().resolve(strict=True)))
    import openai_codex
    return openai_codex


def ask(sdk, prompt, images, audit):
    codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("Set CODEX_BIN to a Codex executable or put codex on PATH.")
    with tempfile.TemporaryDirectory(prefix="neural-surgery-astra-") as work:
        conf = sdk.CodexConfig(
            codex_bin=codex_bin,
            cwd=work,
            config_overrides=(
                "features.shell_tool=false", "features.multi_agent=false",
                "features.apps=false", "features.plugins=false", "features.memories=false",
                'web_search="disabled"', "mcp_servers={}", "project_doc_max_bytes=0",
            ),
        )
        with sdk.Codex(conf) as codex:
            thread = codex.thread_start(
                cwd=work, model=MODEL, sandbox=sdk.Sandbox.read_only,
                approval_mode=sdk.ApprovalMode.deny_all, ephemeral=True,
                base_instructions=BASE,
            )
            audit["sdk_thread_id"] = thread.id
            content = [sdk.TextInput(prompt)]
            for image in images:
                content.extend((
                    sdk.TextInput(f"Supplied image: {image.name}"),
                    sdk.LocalImageInput(str(image)),
                ))
            result = thread.run(content, effort=EFFORT, output_schema=SCHEMA)
            kinds = [getattr(getattr(item, "root", item), "type", None) for item in result.items]
            status = getattr(result.status, "value", str(result.status))
            audit.update({
                "sdk_turn_id": result.id,
                "status": status,
                "item_types": kinds,
                "duration_ms": result.duration_ms,
                "usage": result.usage.model_dump(mode="json") if result.usage else None,
                "raw_response": result.final_response,
                "sdk_error": str(result.error) if result.error else None,
            })
            if status != "completed" or not result.final_response:
                raise RuntimeError(f"Astra produced no completed proposal: {status}")
            if any(kind not in PASSIVE_ITEMS for kind in kinds):
                raise RuntimeError(f"unexpected Codex item types: {kinds}")
            return parse_answer(result.final_response)
