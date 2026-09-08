"""Build MANIFEST.json for downloaded DANDI 001361 assets (run from sosa2025/)."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Code lives here; assets + cache live in external_data/sosa2025/.
ROOT = Path(__file__).resolve().parents[3] / "external_data" / "sosa2025"
DANDISET = "001361"
VERSION = "0.251124.0550"

assets = {a["path"]: a for a in json.load(open(ROOT / "cache/asset_list.json"))}
idents = json.load(open(ROOT / "cache/session_identifiers.json"))

files = sorted(ROOT.glob("nwb/**/*.nwb"))
entries = []
total = 0
for fp in files:
    rel = fp.relative_to(ROOT).as_posix()
    # asset path = sub-<id>/<filename>
    name = fp.name
    sub = name.split("_")[0]
    asset_path = f"{sub}/{name}"
    a = assets[asset_path]
    size = fp.stat().st_size
    assert size == a["size"], f"size mismatch for {rel}: {size} != {a['size']}"
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    total += size
    entries.append({
        "local_path": rel,
        "asset_path": asset_path,
        "asset_id": a["asset_id"],
        "size_bytes": size,
        "sha256": h.hexdigest(),
        "nwb_identifier": idents[asset_path]["identifier"],
        "scene": idents[asset_path]["identifier"].rstrip("/").split("/")[-1],
    })

sha_code = subprocess.run(
    ["git", "-C", str(ROOT / "code_sosa"), "rev-parse", "HEAD"],
    capture_output=True, text=True).stdout.strip()

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
out = ROOT / "MANIFEST.json"
out.write_text(json.dumps(manifest, indent=1) + "\n")
print(f"wrote {out}: {len(entries)} files, {total/1e9:.3f} GB, code_sosa @ {sha_code}")
