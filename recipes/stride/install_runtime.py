"""Download verified wheels and install the measured runtime in an isolated target."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_url(package: dict) -> str:
    filename = quote(package["filename"], safe="-_.")
    if package["provider"] == "vllm-release":
        return "https://github.com/vllm-project/vllm/releases/download/v0.8.5/" + filename
    if package["provider"] == "pytorch-cu118":
        return "https://download.pytorch.org/whl/cu118/" + filename
    request = Request(f"https://pypi.org/pypi/{package['name']}/{package['version']}/json",
                      headers={"User-Agent": "forward-looking-guidance-runtime"})
    with urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    matches = [item for item in metadata["urls"] if item["filename"] == package["filename"]
               and item["digests"]["sha256"] == package["sha256"]]
    if len(matches) != 1 or not matches[0]["url"].startswith("https://files.pythonhosted.org/"):
        raise ValueError(f"Pinned wheel is unavailable or its published hash differs: {package['filename']}")
    return matches[0]["url"]


def download(package: dict, wheelhouse: Path) -> Path:
    path = wheelhouse / package["filename"]
    if path.exists():
        if path.stat().st_size != package["bytes"] or sha256(path) != package["sha256"]:
            raise ValueError(f"Existing wheel differs from the lock: {path}")
        return path
    request = Request(source_url(package), headers={"User-Agent": "forward-looking-guidance-runtime"})
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
    if temporary.stat().st_size != package["bytes"] or sha256(temporary) != package["sha256"]:
        raise ValueError(f"Downloaded wheel fails the lock: {path.name}")
    temporary.replace(path)
    print("Verified", path.name, flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--target", help="New isolated package directory; never the global site-packages")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8")
    if not args.download_only and not args.target:
        parser.error("Specify --target or --download-only")
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("The locked wheels require Python 3.12 on Linux x86_64")
    target = Path(args.target).resolve() if args.target else None
    if target and target.exists():
        raise ValueError("The runtime target must be a new directory")
    lock_path = ROOT / "runtime-wheels.json"
    lock = json.loads(lock_path.read_text())
    wheelhouse = Path(args.wheelhouse).resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    packages = lock["packages"] + lock.get("compatibility_libraries", [])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        downloaded = list(pool.map(lambda package: download(package, wheelhouse), packages))
    wheels = downloaded[:len(lock["packages"])]
    if args.download_only:
        return
    target.mkdir(parents=True)
    libc_name, libc_version = platform.libc_ver()
    unpack_xformers = libc_name == "glibc" and tuple(map(int, libc_version.split(".")[:2])) < (2, 28)
    pip_wheels = [path for path in wheels if not (unpack_xformers and path.name.startswith("xformers-"))]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-index", "--no-deps",
                    "--no-compile", "--target", str(target), *map(str, pip_wheels)], check=True)
    if unpack_xformers:
        # The measured xFormers wheel has a newer platform tag than its ELF needs.
        # Verify the actual requirement before accepting it on the older host.
        wheel = next(path for path in wheels if path.name.startswith("xformers-"))
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if not (target / name).resolve().is_relative_to(target):
                    raise ValueError("Wheel contains an escaping path")
            archive.extractall(target)
        available = tuple(map(int, libc_version.split(".")[:2]))
        for binary in (target / "xformers").glob("*.so"):
            symbols = subprocess.check_output(["readelf", "--version-info", str(binary)], text=True)
            required = [tuple(map(int, version.split("."))) for version in re.findall(r"GLIBC_(\d+\.\d+)", symbols)]
            if any(version > available for version in required):
                raise ValueError("xFormers requires a newer host glibc")
    compatibility = []
    for package in lock.get("compatibility_libraries", []):
        with zipfile.ZipFile(wheelhouse / package["filename"]) as archive:
            for item in package["extract"]:
                path = target / item["target"]
                if not path.resolve().is_relative_to(target):
                    raise ValueError("Compatibility library escapes the runtime target")
                content = archive.read(item["member"])
                if hashlib.sha256(content).hexdigest() != item["sha256"]:
                    raise ValueError("Compatibility library differs from the lock")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        compatibility.append(dict(name=package["name"], version=package["version"],
                                  sha256=package["sha256"]))
    receipt = dict(lock_sha256=sha256(lock_path), wheels_verified=len(downloaded),
                   python=platform.python_version(), target=str(target),
                   compatibility_libraries=compatibility,
                   isolated_install=True, xformers_original_wheel_unpacked=unpack_xformers)
    (target / "runtime_installation.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
