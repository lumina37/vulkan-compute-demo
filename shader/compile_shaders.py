from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


_SCRIPT_DIR = Path(__file__).resolve().parent
_GLSL_DIR = _SCRIPT_DIR / "glsl"
_SPIRV_DIR = _SCRIPT_DIR / "spirv"


_SHADERS = [
    "gauss_filter/v0",
    "gauss_filter/v1",
    "grayscale/ro",
    "grayscale/rw",
    "sgemm/simt/v0",
    "sgemm/simt/v1",
    "sgemm/simt/v2",
    "sgemm/simt/v3",
    "sgemm/simt/v4",
    "sgemm/simt/v5",
    "sgemm/simt/v6",
    "sgemm/simt/v7",
    "sgemm/simt/v8",
    "sgemm/tcore/v0",
    "sgemm/tcore/v1",
    "sgemm/tcore/v2",
    "sgemm/tcore/v3",
    "sgemm/tcore/v4",
    "sgemm/tcore/v5",
    "sgemm/dbg/rrr/simon",
    "sgemm/dbg/rrr/v0",
    "sgemm/dbg/rrr/v1",
    "sgemm/dbg/rcc/ggml",
    "sgemm/dbg/rcc/v0",
    "sgemm/dbg/rcc/v1",
    "flash_attention2/v0",
    "flash_attention2/v1",
    "prefix_sum/v0",
    "topk/v0",
]


def find_glslang() -> Path | None:
    if found := shutil.which("glslangValidator"):
        return Path(found)
    candidates: list[Path] = []
    if sdk := os.environ.get("VULKAN_SDK"):
        bin_dir = Path(sdk) / "Bin"
        candidates += [bin_dir / "glslangValidator", bin_dir / "glslangValidator.exe"]
    candidates += [Path("glslangValidator"), Path("glslangValidator.exe")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def compile_shader(glslang: Path, src: Path, dst: Path, target_env: str = "vulkan1.3") -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(glslang),
            "-V",
            str(src),
            "--vn",
            "code",
            "--target-env",
            target_env,
            "-o",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed: %s", src)
        if result.stderr:
            logger.error(result.stderr)
        return False
    if result.stderr:
        logger.warning(result.stderr)
    return True


def needs_rebuild(dst: Path, src: Path) -> bool:
    if not dst.exists():
        return True
    if src.stat().st_mtime > dst.stat().st_mtime:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile GLSL compute shaders to SPIR-V C headers")
    parser.add_argument("--glslang", help="Path to glslangValidator executable")
    parser.add_argument("--force", action="store_true", help="Force recompilation of all shaders")
    args = parser.parse_args()

    glslang = Path(args.glslang) if args.glslang else find_glslang()
    if not glslang or not glslang.is_file():
        logger.error("glslangValidator not found: %s", glslang or "(auto-detect failed)")
        sys.exit(1)

    compiled, skipped, errors = 0, 0, 0
    for name in _SHADERS:
        src = _GLSL_DIR / f"{name}.comp"
        dst = _SPIRV_DIR / f"{name}.h"
        if not args.force and not needs_rebuild(dst, src):
            skipped += 1
            continue
        logger.info("Compiling: %s", name)
        if compile_shader(glslang, src, dst):
            compiled += 1
        else:
            errors += 1

    logger.info("Done: %d compiled, %d skipped, %d errors", compiled, skipped, errors)
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
