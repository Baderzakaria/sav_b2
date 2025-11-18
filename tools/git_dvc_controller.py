#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional


def run(cmd: List[str], cwd: Optional[str] = None) -> int:
    return subprocess.call(cmd, cwd=cwd)


def read_prompts_version(prompts_path: str) -> Optional[str]:
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:
        pass
    return None


def cmd_git_commit(args: argparse.Namespace) -> int:
    files = args.files or []
    if args.all:
        rc = run(["git", "add", "-A"])
    else:
        rc = 0
        if files:
            rc = run(["git", "add", *files])
    if rc != 0:
        return rc
    return run(["git", "commit", "-m", args.message])


def cmd_git_tag_prompts(args: argparse.Namespace) -> int:
    prompts_path = args.prompts
    version = args.version or read_prompts_version(prompts_path)
    if not version:
        print("ERROR: Prompt version not found; pass --version or ensure JSON has a top-level 'version' field.")
        return 1
    tag_name = args.tag_name or f"prompts/{version}"
    rc = run(["git", "tag", "-a", tag_name, "-m", version])
    if rc != 0:
        return rc
    if args.push:
        return run(["git", "push", "--tags"]) 
    return 0


def cmd_git_push(args: argparse.Namespace) -> int:
    rc = run(["git", "push"])
    if rc != 0:
        return rc
    if args.tags:
        return run(["git", "push", "--tags"]) 
    return 0


def cmd_dvc_init(args: argparse.Namespace) -> int:
    rc = run(["dvc", "init"])
    if rc != 0:
        return rc
    if args.remote:
        return run(["dvc", "remote", "add", "-d", args.remote_name, args.remote])
    return 0


def cmd_dvc_add(args: argparse.Namespace) -> int:
    if not args.paths:
        print("ERROR: Provide one or more paths to add.")
        return 1
    return run(["dvc", "add", *args.paths])


def cmd_dvc_repro(args: argparse.Namespace) -> int:
    if args.stage:
        return run(["dvc", "repro", args.stage])
    return run(["dvc", "repro"])


def cmd_dvc_push(args: argparse.Namespace) -> int:
    return run(["dvc", "push"])


def cmd_dvc_pull(args: argparse.Namespace) -> int:
    return run(["dvc", "pull"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Git & DVC controller")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Git commands
    pg = sub.add_parser("git-commit", help="Stage and commit changes")
    pg.add_argument("-m", "--message", required=True)
    pg.add_argument("--all", action="store_true", help="git add -A before commit")
    pg.add_argument("files", nargs="*")
    pg.set_defaults(func=cmd_git_commit)

    pt = sub.add_parser("git-tag-prompts", help="Create a tag using prompts JSON version")
    pt.add_argument("--prompts", default=os.path.join("prompts", "freemind_prompts.json"))
    pt.add_argument("--version", help="Override version string")
    pt.add_argument("--tag-name", help="Override tag name (default: prompts/<version>)")
    pt.add_argument("--push", action="store_true", help="Push tags after creating")
    pt.set_defaults(func=cmd_git_tag_prompts)

    pp = sub.add_parser("git-push", help="Push commits (and optionally tags)")
    pp.add_argument("--tags", action="store_true")
    pp.set_defaults(func=cmd_git_push)

    # DVC commands
    di = sub.add_parser("dvc-init", help="Initialize DVC and optionally set default remote")
    di.add_argument("--remote", help="Remote URL/path (e.g. /path/to/store or s3://bucket/path)")
    di.add_argument("--remote-name", default="origin")
    di.set_defaults(func=cmd_dvc_init)

    da = sub.add_parser("dvc-add", help="Track files with DVC")
    da.add_argument("paths", nargs="+")
    da.set_defaults(func=cmd_dvc_add)

    dr = sub.add_parser("dvc-repro", help="Reproduce DVC pipeline")
    dr.add_argument("--stage", help="Stage name")
    dr.set_defaults(func=cmd_dvc_repro)

    dp = sub.add_parser("dvc-push", help="Push DVC artifacts to remote")
    dp.set_defaults(func=cmd_dvc_push)

    dl = sub.add_parser("dvc-pull", help="Pull DVC artifacts from remote")
    dl.set_defaults(func=cmd_dvc_pull)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()


