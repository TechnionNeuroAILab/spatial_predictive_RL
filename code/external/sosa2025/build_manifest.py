"""Build a checksum manifest for downloaded DANDI 001361 assets."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Code lives here; assets + cache live in data/sosa2025/.
ROOT = Path(__file__).resolve().parents[3] / "data" / "sosa2025"
DANDISET = "001361"
VERSION = "0.251124.0550"


def build_manifest(output: Path) -> None:
    """Validate local NWB files against DANDI metadata and write checksums."""
    with open(ROOT / "cache" / "asset_list.json", encoding="utf-8") as fh:
        assets = {a["path"]: a for a in json.load(fh)}
    with open(ROOT / "cache" / "session_identifiers.json", encoding="utf-8") as fh:
        idents = json.load(fh)

    entries = []
    total = 0
    for fp in sorted(ROOT.glob("nwb/**/*.nwb")):
        rel = fp.relative_to(ROOT).as_posix()
        # DANDI asset paths have the form sub-<id>/<filename>.
        asset_path = f"{fp.name.split('_')[0]}/{fp.name}"
        asset = assets[asset_path]
        size = fp.stat().st_size
        assert size == asset["size"], (
            f"size mismatch for {rel}: {size} != {asset['size']}"
        )
        digest = hashlib.sha256()
        with open(fp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
        total += size
        entries.append({
            "local_path": rel,
            "asset_path": asset_path,
            "asset_id": asset["asset_id"],
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "nwb_identifier": idents[asset_path]["identifier"],
            "scene": idents[asset_path]["identifier"].rstrip("/").split("/")[-1],
        })

    sha_code = subprocess.run(
        ["git", "-C", str(ROOT / "code_sosa"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    manifest = {
        "dandiset": DANDISET,
        "dandiset_version": VERSION,
        "dandiset_url": f"https://dandiarchive.org/dandiset/{DANDISET}/{VERSION}",
        "paper_doi": "10.1038/s41593-025-01985-4",
        "code_repo": "https://github.com/GiocomoLab/Sosa_et_al_2024",
        "code_repo_head_sha": sha_code,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(entries),
        "total_bytes": total,
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"wrote {output}: {len(entries)} files, "
        f"{total / 1e9:.3f} GB, code_sosa @ {sha_code}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "MANIFEST.json",
        help="output path (default: data/sosa2025/MANIFEST.json)",
    )
    args = parser.parse_args()
    build_manifest(args.output.resolve())


if __name__ == "__main__":
    main()
