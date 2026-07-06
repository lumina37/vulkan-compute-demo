from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# logging
_FORMATTER = logging.Formatter(
    "<{asctime}> [{levelname}] {message}",
    "%Y-%m-%d %H:%M:%S",
    style="{",
)
_HANDLER = logging.StreamHandler(sys.stderr)
_HANDLER.setFormatter(_FORMATTER)

logger = logging.getLogger("compile_shaders")
logger.setLevel(logging.INFO)
logger.addHandler(_HANDLER)

# path
_SCRIPT_DIR = Path(__file__).resolve().parent
_GLSL_DIR = _SCRIPT_DIR / "glsl"
_SPIRV_DIR = _SCRIPT_DIR / "spirv"


@dataclass
class ShaderInfo:
    name: str
    macros: list[str] = field(default_factory=list)
    target_env: str = "vulkan1.3"


@dataclass
class CompileTaskInfo:
    src: Path
    dst: Path
    macros: list[str] = field(default_factory=list)
    target_env: str = "vulkan1.3"


_SHADER_INFOS: list[ShaderInfo] = [
    ShaderInfo("gauss_filter/v0"),
    ShaderInfo("gauss_filter/v1"),
    ShaderInfo("grayscale/ro"),
    ShaderInfo("grayscale/rw"),
    ShaderInfo("sgemm/simt/v0"),
    ShaderInfo("sgemm/simt/v1"),
    ShaderInfo("sgemm/simt/v2"),
    ShaderInfo("sgemm/simt/v3"),
    ShaderInfo("sgemm/simt/v4"),
    ShaderInfo("sgemm/simt/v5"),
    ShaderInfo("sgemm/simt/v6"),
    ShaderInfo("sgemm/simt/v7"),
    ShaderInfo("sgemm/tcore/v0"),
    ShaderInfo("sgemm/tcore/v1"),
    ShaderInfo("sgemm/tcore/v2"),
    ShaderInfo("sgemm/tcore/v3"),
    ShaderInfo("sgemm/tcore/v4"),
    ShaderInfo("sgemm/tcore/v5"),
    ShaderInfo("sgemm/dbg/rrr/simon"),
    ShaderInfo("sgemm/dbg/rrr/v0"),
    ShaderInfo("sgemm/dbg/rrr/v1"),
    ShaderInfo("sgemm/dbg/rcc/ggml"),
    ShaderInfo("sgemm/dbg/rcc/v0"),
    ShaderInfo("sgemm/dbg/rcc/v1"),
    ShaderInfo("flash_attention2/v0"),
    ShaderInfo("flash_attention2/v1"),
    ShaderInfo("prefix_sum/v0"),
    ShaderInfo("topk/v0"),
]


def find_compiler() -> Path | None:
    if found := shutil.which("glslangValidator"):
        return Path(found)

    if sdk := os.environ.get("VULKAN_SDK"):
        bin_dir = Path(sdk) / "Bin"
        for candidate in [
            bin_dir / "glslangValidator",
            bin_dir / "glslangValidator.exe",
        ]:
            if candidate.is_file():
                return candidate

    return None


def compile_shader(compiler: Path, task: CompileTaskInfo) -> bool:
    task.dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Compiling: %s", task.src)

    cmd: list[str] = [str(compiler), "-V", str(task.src)]
    cmd.extend(f"-D{m}" for m in task.macros)
    cmd += ["--vn", "code", "--target-env", task.target_env, "-o", str(task.dst)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Compilation failed: %s", task.src)
        if result.stderr:
            logger.error(result.stderr)
        return False
    if result.stderr:
        logger.warning(result.stderr)

    return True


def needs_rebuild(dst: Path, src: Path) -> bool:
    try:
        dst_mtime = os.stat(dst).st_mtime
    except FileNotFoundError:
        return True

    try:
        src_mtime = os.stat(src).st_mtime
    except FileNotFoundError:
        logger.error("Shader src not found: %s", src)
        return False

    return src_mtime > dst_mtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile GLSL compute shaders to SPIR-V C headers")
    parser.add_argument("--compiler", help="Path to glslangValidator executable")
    parser.add_argument("--force", action="store_true", help="Force recompilation of all shaders")
    args = parser.parse_args()

    compiler = Path(args.compiler) if args.compiler else find_compiler()
    if not compiler:
        logger.error("glslangValidator not found")
        sys.exit(1)
    if not compiler.is_file():
        logger.error("glslangValidator is not a file: %s", compiler)
        sys.exit(1)

    tasks: list[CompileTaskInfo] = []
    skipped = 0
    for shader_info in _SHADER_INFOS:
        task = CompileTaskInfo(
            src=_GLSL_DIR / f"{shader_info.name}.comp",
            dst=_SPIRV_DIR / f"{shader_info.name}.h",
            macros=shader_info.macros,
        )
        if not args.force and not needs_rebuild(task.dst, task.src):
            skipped += 1
            continue
        tasks.append(task)

    if not tasks:
        logger.info("Done: 0 compiled, %d skipped, 0 errors", skipped)
        return

    compiled, errors = 0, 0
    with ThreadPoolExecutor(max_workers=os.process_cpu_count() or 1) as executor:
        futures = [executor.submit(compile_shader, compiler, task) for task in tasks]
        for future in as_completed(futures):
            if future.result():
                compiled += 1
            else:
                errors += 1

    logger.info("Done: %d compiled, %d skipped, %d errors", compiled, skipped, errors)
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
