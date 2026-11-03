"""Fetch the two pinned public model snapshots and record their revisions."""
from pathlib import Path
import argparse
import json
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    settings = json.loads((Path(__file__).parent / "settings.json").read_text())
    for component in ("model", "encoder"):
        spec = settings[component]
        path = Path(args.output_root) / spec["repository"].replace("/", "--")
        receipt = path / "source_revision.json"
        request = path / "download_request.json"
        if receipt.exists() and json.loads(receipt.read_text()) != spec:
            raise ValueError("Existing model revision receipt differs")
        if path.exists() and any(path.iterdir()) and not receipt.exists():
            if not request.exists() or json.loads(request.read_text()) != spec:
                raise ValueError("Existing directory has no matching download request")
        path.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps(spec, indent=2))
        snapshot_download(repo_id=spec["repository"], revision=spec["revision"], local_dir=str(path),
                          allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"])
        receipt.write_text(json.dumps(spec, indent=2))
        request.unlink()
        print(spec["repository"], spec["revision"], flush=True)


if __name__ == "__main__":
    main()
