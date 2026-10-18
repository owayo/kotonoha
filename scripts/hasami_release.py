#!/usr/bin/env python3
"""公開済み HSD の更新検出、Release 用の版固定と検証付き取得。"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

UPSTREAM = "owayo/hasami"
SNAPSHOT_FILE = "hasami-dictionaries.json"
DICTIONARIES = {"ipadic", "ipadic-neologd", "ipadic-neologd-sudachi"}
MAX_METADATA_BYTES = 1024 * 1024
MODEL_RELEASE_ASSETS = {
    SNAPSHOT_FILE,
    "kotonoha-models.tar.gz",
    "kotonoha-models.tar.gz.sha256",
    "SHA256SUMS",
    "kotonoha-x86_64-unknown-linux-gnu.tar.gz",
    "kotonoha-aarch64-unknown-linux-gnu.tar.gz",
    "kotonoha-aarch64-apple-darwin.tar.gz",
    "kotonoha-x86_64-apple-darwin.tar.gz",
    "kotonoha-x86_64-pc-windows-msvc.zip",
}


def gh_json(endpoint: str, *, missing_ok: bool = False):
    """GitHub API の JSON を取得する。未公開の Release の 404 だけを許容できる。"""
    result = subprocess.run(["gh", "api", endpoint], capture_output=True, check=False)
    if result.returncode:
        if missing_ok and b"HTTP 404" in result.stderr:
            return None
        raise ValueError(f"GitHub API request failed: {endpoint}")
    return json.loads(result.stdout)


def metadata_asset(repository: str, asset: dict) -> bytes:
    """小さい公開添付を取得し、サイズと GitHub の SHA-256 を検証する。"""
    if not 0 < asset["size"] <= MAX_METADATA_BYTES or asset["state"] != "uploaded":
        raise ValueError("Invalid or incomplete metadata asset")
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/releases/assets/{int(asset['id'])}",
            "-H",
            "Accept: application/octet-stream",
        ],
        capture_output=True,
        check=True,
    )
    data = result.stdout
    if len(data) != asset["size"]:
        raise ValueError("Metadata asset size mismatch")
    if (
        asset.get("digest")
        and asset["digest"] != "sha256:" + hashlib.sha256(data).hexdigest()
    ):
        raise ValueError("Metadata asset SHA-256 mismatch")
    return data


def validate_snapshot(snapshot: dict) -> dict:
    """版・ファイル名・ハッシュを検証し、HSD の公開目録を返す。通信は行わない。"""
    tag = snapshot["tag"]
    if (
        snapshot["schema_version"] != 1
        or snapshot["repository"] != UPSTREAM
        or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag)
        or not re.fullmatch(r"[0-9a-f]{64}", snapshot["catalog_sha256"])
    ):
        raise ValueError("Invalid HSD release identity")
    catalog = snapshot["catalog"]
    entries = catalog["dictionaries"]
    if (
        catalog["hasami_version"] != tag[1:]
        or type(catalog["format_version"]) is not int
        or catalog["format_version"] < 1
        or catalog["recommended"] not in DICTIONARIES
        or len(entries) != len(DICTIONARIES)
        or {entry["name"] for entry in entries} != DICTIONARIES
    ):
        raise ValueError("Invalid HSD release catalog")
    for entry in entries:
        if (
            entry["file"] != entry["name"] + ".hsd"
            or type(entry["size"]) is not int
            or not 0 < entry["size"] <= 2**32
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("Invalid HSD filename, size or SHA-256")
    return catalog


def published_snapshot(tag: str = "") -> dict:
    """hasami の安定版 Release と公開済みの HSD 添付を照合して固定する。"""
    if tag and not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Invalid hasami tag")
    endpoint = f"repos/{UPSTREAM}/releases/" + (f"tags/{tag}" if tag else "latest")
    release = gh_json(endpoint)
    if release["draft"] or release["prerelease"]:
        raise ValueError("HSD updates require a published stable release")
    assets = {asset["name"]: asset for asset in release["assets"]}
    data = metadata_asset(UPSTREAM, assets["dictionaries.json"])
    snapshot = dict(
        schema_version=1,
        repository=UPSTREAM,
        tag=release["tag_name"],
        catalog_sha256=hashlib.sha256(data).hexdigest(),
        catalog=json.loads(data),
    )
    catalog = validate_snapshot(snapshot)
    for entry in catalog["dictionaries"]:
        asset = assets[entry["file"]]
        if (
            asset["state"] != "uploaded"
            or asset["size"] != entry["size"]
            or (asset.get("digest") and asset["digest"] != "sha256:" + entry["sha256"])
        ):
            raise ValueError("Published HSD asset differs from the catalog")
    return snapshot


def dictionary_identity(snapshot: dict) -> tuple:
    """公開タグや圧縮方法が変わっても、同じ HSD を重複公開しないための識別値。"""
    catalog = validate_snapshot(snapshot)
    return (
        catalog["format_version"],
        catalog["recommended"],
        tuple(
            sorted((e["name"], e["size"], e["sha256"]) for e in catalog["dictionaries"])
        ),
    )


def rebuild_required(candidate: dict, released: dict | None) -> bool:
    """最後に公開できたモデルとの比較で、失敗したビルドは次回も再試行する。"""
    identity = dictionary_identity(candidate)
    return released is None or identity != dictionary_identity(released)


def last_model_snapshot(repository: str) -> dict | None:
    """Kotonoha の最新公開 Release に添付した HSD の記録を読む。"""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid model repository")
    release = gh_json(f"repos/{repository}/releases/latest", missing_ok=True)
    if release is None:
        return None
    asset = next((a for a in release["assets"] if a["name"] == SNAPSHOT_FILE), None)
    if asset is None:
        return None
    # 添付途中の Release を完了済みとして扱わない。
    if not MODEL_RELEASE_ASSETS.issubset(
        {a["name"] for a in release["assets"] if a["state"] == "uploaded"}
    ):
        return None
    snapshot = json.loads(metadata_asset(repository, asset))
    validate_snapshot(snapshot)
    return snapshot


def verify_dictionary(path: Path, entry: dict) -> bool:
    """辞書の実体を公開目録のサイズ・SHA-256 と照合する。"""
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]


def download_dictionary(snapshot: dict, output: Path) -> Path:
    """推奨 HSD を固定タグから取得して検証する。既存の異なるファイルは保護する。"""
    catalog = validate_snapshot(snapshot)
    entry = next(
        e for e in catalog["dictionaries"] if e["name"] == catalog["recommended"]
    )
    output.mkdir(parents=True, exist_ok=True)
    target = output / entry["file"]
    if target.exists():
        if verify_dictionary(target, entry):
            return target
        raise ValueError("Existing HSD differs from the pinned release")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / entry["file"]
        assert path.resolve().is_relative_to(output.resolve())
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                snapshot["tag"],
                "--repo",
                UPSTREAM,
                "--pattern",
                entry["file"],
                "--output",
                str(path),
            ],
            check=True,
        )
        if not verify_dictionary(path, entry):
            raise ValueError("Downloaded HSD size or SHA-256 mismatch")
        path.replace(target)
    return target


def main():
    """更新確認は読み取りのみ。Release の起動はワークフロー側で行う。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", "owayo/kotonoha")
    )
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--tag", default="")
    snapshot.add_argument("--output", type=Path, required=True)
    download = commands.add_parser("download")
    download.add_argument("--snapshot", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            candidate = published_snapshot()
            changed = rebuild_required(candidate, last_model_snapshot(args.repository))
            print(f"changed={str(changed).lower()}")
            print(f"hasami_tag={candidate['tag']}")
        elif args.command == "snapshot":
            record = published_snapshot(args.tag)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        else:
            record = json.loads(args.snapshot.read_text(encoding="utf-8"))
            path = download_dictionary(record, args.output_dir)
            print(f"dictionary_path={path.resolve().as_posix()}")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        StopIteration,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"HSD release verification failed: {error}\n")


if __name__ == "__main__":
    main()
