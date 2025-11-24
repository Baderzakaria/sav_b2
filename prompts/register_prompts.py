

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

from freemind_env import load_environment

load_environment()

PROMPT_FILE = Path("prompts/freemind_prompts.json")

def main() -> int:
    if not PROMPT_FILE.exists():
        print(f"Prompt file not found: {PROMPT_FILE}", file=sys.stderr)
        return 1

    with PROMPT_FILE.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    agents = (cfg.get("agents") or {})
    if not agents:
        print("No agents found in prompt file.", file=sys.stderr)
        return 1

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("freemind_prompt_registry")

    try:
        from mlflow.genai import register_prompt, load_prompt
        HAS_PROMPT_REGISTRY = True
    except Exception:
        register_prompt = None
        load_prompt = None
        HAS_PROMPT_REGISTRY = False

    with mlflow.start_run(run_name="register_prompts"):
        for agent_key, agent in agents.items():
            name = agent.get("name") or agent_key
            template = agent.get("prompt_template", "")
            if not template:
                continue

            if HAS_PROMPT_REGISTRY and register_prompt is not None:
                try:

                    existing_version = None
                    try:
                        existing = load_prompt(f"prompts:/{name}@latest")
                        existing_template = existing.template if hasattr(existing, 'template') else str(existing)
                        existing_version = getattr(existing, 'version', None)

                        if existing_template.strip() == template.strip():
                            print(f"Prompt '{name}' unchanged (v{existing_version}), skipping registration")
                            mlflow.log_param(f"prompt_{name}_version", f"v{existing_version} (reused)")
                            continue
                        else:
                            print(f"Prompt '{name}' template changed, creating new version...")
                    except Exception:

                        print(f"Prompt '{name}' not found, creating initial version...")

                    pr = register_prompt(
                        name=name,
                        template=template,
                        commit_message=f"Import from {PROMPT_FILE.name}",
                    )
                    new_version = getattr(pr, 'version', 'unknown')
                    print(f"Registered prompt '{pr.name}' v{new_version}")
                    mlflow.log_param(f"prompt_{name}_version", f"v{new_version}")
                    if existing_version:
                        mlflow.log_param(f"prompt_{name}_previous_version", f"v{existing_version}")
                except Exception as e:

                    print(f"Failed to register '{name}' ({e}); logging as artifact.")
                    mlflow.log_dict({"name": name, "template": template}, artifact_file=f"prompts/{name}.json")
            else:

                print(f"Prompt Registry not available, logging '{name}' as artifact")
                mlflow.log_dict({"name": name, "template": template}, artifact_file=f"prompts/{name}.json")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

