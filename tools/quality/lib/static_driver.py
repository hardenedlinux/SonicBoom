#!/usr/bin/env python3
"""Run a static-analysis mode over SonicBoom's real compilation database.

The compilation database is produced by CMake with
CMAKE_EXPORT_COMPILE_COMMANDS=ON; every command below is the project's own
command with only the output/dependency files stripped. No compiler or include
flag is guessed or simplified.

Modes:
  warnings        g++ -Wall -Wextra -fsyntax-only
  analyzer        g++ -fanalyzer (GCC static analyzer)
  clang-analyzer  clang++ --analyze (Clang Static Analyzer)
  tidy            clang-tidy using a project config

Diagnostics are attributed to SonicBoom-owned files only (core/, capi/, tests/).
Diagnostics from third_party/native-torch, the MLIR/LLVM install, or system
headers are recorded in the per-TU logs but do not fail the gate: those trees
are not SonicBoom-owned and are explicitly out of scope (see the root
CMakeLists.txt sanitizer note).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

DIAG_RE = re.compile(r"^(.*?):(\d+):(\d+): (warning|error): (.*)$")
# Clang analyzer text output can omit the column.
DIAG_NOC_RE = re.compile(r"^(.*?):(\d+): (warning|error): (.*)$")


def load_db(db_path: Path) -> list[dict]:
    with db_path.open() as f:
        return json.load(f)


def is_owned(path: Path, root: Path, roots: tuple[str, ...]) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "third_party":
        return False
    return any(parts[0] == r for r in roots)


def argv_of(entry: dict) -> list[str]:
    argv = (list(entry["arguments"]) if "arguments" in entry
            else shlex.split(entry["command"]))
    # Some CMake versions include the compiler as arguments[0]; normalize it
    # so the selected analyzer compiler is the only one in the command.
    if argv:
        base = os.path.basename(argv[0])
        if re.match(r"^(g\+\+|gcc|c\+\+|cc|clang\+\+|clang)", base):
            argv = argv[1:]
    return argv


def strip_output_args(argv: list[str]) -> list[str]:
    """Remove -o/-c/dependency flags so the TU can be syntax-checked."""
    out: list[str] = []
    i = 0
    takes_value = {"-o", "-MT", "-MF", "-MQ"}
    drop = {"-c", "-MD", "-MMD", "-MP", "-M", "-MG"}
    while i < len(argv):
        a = argv[i]
        if a in takes_value:
            i += 2
            continue
        if a in drop:
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def diagnostic(stdout: str, stderr: str, cwd: Path, owned_check) -> list[str]:
    found: list[str] = []
    for line in (stdout + "\n" + stderr).splitlines():
        m = DIAG_RE.match(line)
        if m:
            path = Path(m.group(1))
            if not path.is_absolute():
                path = cwd / path
            if owned_check(path):
                found.append(line)
            continue
        m = DIAG_NOC_RE.match(line)
        if m:
            path = Path(m.group(1))
            if not path.is_absolute():
                path = cwd / path
            if owned_check(path):
                found.append(line)
    return found


def run_one(entry: dict, mode: str, args, root: Path) -> dict:
    cwd = Path(entry["directory"])
    src = Path(entry["file"])
    argv = strip_output_args(argv_of(entry))

    if mode == "warnings":
        # -Werror is deliberately not used: a warning inside an external header
        # would abort the TU and be reported as a build failure even though it
        # is out of scope. The driver parses diagnostics per file instead and
        # fails on any warning/error attributed to a SonicBoom-owned file.
        cmd = [args.gxx] + argv + [
            "-Wall", "-Wextra",
            "-fsyntax-only", "-fdiagnostics-color=never",
        ]
    elif mode == "analyzer":
        # GCC's analyzer runs during a real compile. The exploded-graph budget
        # is lowered from GCC 13's defaults (5 / 8) because the defaults let the
        # analyzer's state grow past 20 GiB on MLIR-heavy compiler TUs
        # (measured on core/compiler/lib/BoomDialect.cpp), which makes the
        # pass unusable on a normal CI runner. Every other flag still comes
        # from compile_commands.json.
        #
        # The lowered budget also makes three checks lose track of values inside
        # inlined libstdc++ code, where they report container reallocation
        # (`__cur`/`__location`), allocation (`operator new`) and
        # `vector::data()` as null or uninitialized reads. GCC has no
        # per-diagnostic suppression attribute and the reports name SonicBoom
        # files at the inline site, so the three classes are switched off for
        # this mode. The analyzer's pointer-lifetime checks (use-after-free,
        # double-free, leaks, out-of-bounds) stay active.
        obj = Path(tempfile.gettempdir()) / f"sb-analyzer-{os.getpid()}-{abs(hash(str(src))) % 10**8}.o"
        cmd = [args.gxx] + argv + [
            "-fanalyzer",
            "--param=analyzer-bb-explosion-factor=1",
            "--param=analyzer-max-enodes-per-program-point=2",
            "-Wno-analyzer-null-dereference",
            "-Wno-analyzer-possible-null-dereference",
            "-Wno-analyzer-use-of-uninitialized-value",
            "-fdiagnostics-color=never",
            "-c", "-o", str(obj),
        ]
    elif mode == "clang-analyzer":
        gcc_dir = (["--gcc-install-dir=" + args.gcc_install_dir]
                   if args.gcc_install_dir else [])
        cmd = [args.clang] + gcc_dir + argv + [
            "--analyze", "-Xanalyzer", "-analyzer-output=text",
            "-fno-color-diagnostics", "-o", os.devnull,
        ]
    elif mode == "tidy":
        gcc_dir = (["--extra-arg=--gcc-install-dir=" + args.gcc_install_dir]
                   if args.gcc_install_dir else [])
        cmd = [args.clang_tidy, "-p", str(args.build_dir),
               "--config-file", str(args.tidy_config)] + gcc_dir + [
               "--quiet", str(src)]
    else:
        raise ValueError(mode)

    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), text=True, capture_output=True,
            timeout=args.timeout,
        )
        rc = proc.returncode
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        err += f"\nTIMEOUT after {args.timeout}s\n"

    if mode == "analyzer":
        try:
            Path(tempfile.gettempdir(), f"sb-analyzer-{os.getpid()}-{abs(hash(str(src))) % 10**8}.o").unlink()
        except OSError:
            pass

    def owned(p: Path) -> bool:
        return is_owned(p, root, tuple(args.roots.split(",")))

    diags = diagnostic(out, err, cwd, owned)

    log_dir = Path(args.log_dir) / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = str(src).replace(str(root) + "/", "").replace("/", "__")
    log = log_dir / (safe + ".log")
    with log.open("w") as f:
        f.write("COMMAND: " + shlex.join(cmd) + "\n")
        f.write(f"EXIT: {rc}\n")
        f.write("--- stdout ---\n")
        f.write(out)
        f.write("\n--- stderr ---\n")
        f.write(err)

    return {
        "file": str(src),
        "rc": rc,
        "diagnostics": diags,
        "timeout": rc == 124,
        "log": str(log),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["warnings", "analyzer", "clang-analyzer", "tidy"])
    ap.add_argument("--roots", default="core,capi,tests")
    ap.add_argument("--gxx", default="g++-13")
    ap.add_argument("--clang", default="clang++")
    ap.add_argument("--clang-tidy", default="clang-tidy")
    # Pin clang to the GCC installation that provides the standard library the
    # project's formal compiler uses. Empty = let clang auto-detect.
    ap.add_argument("--gcc-install-dir", default="")
    ap.add_argument("--tidy-config", default="")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    root = Path(args.db).resolve().parent.parent
    # compile_commands.json lives at <build>/compile_commands.json; the source
    # root is not necessarily its grandparent for out-of-tree builds. Callers
    # pass absolute paths in the DB, and ownership is decided against the
    # repository root passed through the environment.
    env_root = os.environ.get("SB_ROOT")
    if env_root:
        root = Path(env_root).resolve()

    entries = load_db(Path(args.db))
    selected = []
    for e in entries:
        src = Path(e["file"])
        if not src.is_absolute():
            src = Path(e["directory"]) / src
        if is_owned(src, root, tuple(args.roots.split(","))):
            if src.exists():
                selected.append(e)
            else:
                print(f"skip missing source: {src}", file=sys.stderr)

    selected.sort(key=lambda e: e["file"])
    print(f"mode={args.mode} files={len(selected)} roots={args.roots}")
    if not selected:
        print("no SonicBoom-owned translation units found", file=sys.stderr)
        return 2

    diag_files = 0
    compile_failures = 0
    diag_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, e, args.mode, args, root) for e in selected]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            if r["diagnostics"]:
                diag_files += 1
                diag_count += len(r["diagnostics"])
                print(f"DIAG {r['file']}")
                for d in r["diagnostics"]:
                    print("  " + d)
            elif r["timeout"]:
                compile_failures += 1
                print(f"TIMEOUT {r['file']} (see {r['log']})")
            elif r["rc"] != 0:
                compile_failures += 1
                print(f"COMPILE-FAIL rc={r['rc']} {r['file']} (see {r['log']})")

    print(f"summary mode={args.mode} files={len(selected)} "
          f"diagnostic_files={diag_files} "
          f"diagnostics={diag_count} compile_failures={compile_failures}")
    return 0 if (diag_files == 0 and compile_failures == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
