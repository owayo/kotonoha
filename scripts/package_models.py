#!/usr/bin/env python3
"""検証済みの KNM モデルと辞書を GitHub Release 用に梱包する（標準ライブラリのみ）。"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import struct
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "kotonoha-models.tar.gz"


def file_digest(path: Path) -> str:
    """ファイル全体の SHA-256 をストリームで計算する。

    Returns:
        SHA-256 の16進文字列。

    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def merge_dictionaries(paths: list[Path]) -> bytes:
    """ランタイムと同じ優先順位で CSV をマージする。

    Returns:
        UTF-8 のアクセント辞書 CSV。

    Raises:
        ValueError: 辞書が空、またはヘッダ・行・アクセント値が不正な場合。

    """
    entries = {}
    for path in paths:
        current = {}
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(
                (line for line in stream if not line.startswith("#")), strict=True
            )
            if next(reader, None) != ["lemma", "reading", "accent_type"]:
                raise ValueError(f"Invalid dictionary header: {path}")
            for row in reader:
                if not row:
                    continue
                if len(row) != 3:
                    raise ValueError(f"Invalid dictionary row in {path}: {row}")
                lemma, reading, accent = (value.strip() for value in row)
                if not lemma or not reading or not re.fullmatch(r"[0-9]+", accent):
                    raise ValueError(f"Invalid dictionary entry in {path}: {row}")
                if not 0 <= int(accent) <= 255:
                    raise ValueError(f"Accent outside u8 range in {path}: {row}")
                # AccentDict::lookup_exact と同じく、同じ CSV 内では先頭を採用する。
                current.setdefault((lemma, reading), int(accent))
        # CSV 間は V66Bundle::from_paths と同じ後勝ち。
        entries.update(current)
    if not entries:
        raise ValueError("The bundled accent dictionary is empty")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["lemma", "reading", "accent_type"])
    writer.writerows(
        (lemma, reading, accent) for (lemma, reading), accent in sorted(entries.items())
    )
    return output.getvalue().encode("utf-8")


def package_models(
    models_dir: Path,
    output_dir: Path,
    lock_path: Path,
    base_dictionary: Path,
    source_revision: str,
    native: bool = True,
) -> Path:
    """固定した SHA-256 を検証してから、モデルと辞書を再現可能な形で梱包する。

    Returns:
        作成したアーカイブのパス。

    Raises:
        ValueError: 入力ファイルの欠損・LFS ポインタ・ハッシュ不一致を検出した場合。

    """
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    sources = {}
    for entry in lock["files"]:
        name = entry["file"]
        if Path(name).name != name or name in sources:
            raise ValueError(f"Invalid or duplicate model filename: {name}")
        path = models_dir / name
        if not path.is_file():
            raise ValueError(f"Missing model bundle input: {path}")
        with path.open("rb") as stream:
            if stream.read(128).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(
                    f"Git LFS pointer found; download the actual model: {path}"
                )
        if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
            raise ValueError(f"Model bundle input differs from {lock_path}: {path}")
        sources[name] = path

    if native:
        converted_manifest = json.loads(
            (models_dir / "knm-manifest.json").read_text(encoding="utf-8")
        )
        converted_files = {
            entry["file"]: entry for entry in converted_manifest["files"]
        }
        for name in list(sources):
            if not name.endswith(".onnx"):
                continue
            sources.pop(name)
            original = next(entry for entry in lock["files"] if entry["file"] == name)
            native_name = original["native_file"]
            if (
                Path(native_name).name != native_name
                or not native_name.endswith(".knm")
                or native_name in sources
            ):
                raise ValueError(f"Invalid or duplicate KNM filename: {native_name}")
            path = models_dir / native_name
            if not path.is_file():
                raise ValueError(
                    f"Missing KNM: {path}; "
                    f"run kotonoha model convert --dir {models_dir}"
                )
            pinned = converted_files.get(path.name)
            if (
                not pinned
                or path.stat().st_size != pinned["size"]
                or file_digest(path) != pinned["sha256"]
            ):
                raise ValueError(f"KNM differs from conversion manifest: {path}")
            with path.open("rb") as stream:
                header = stream.read(32)
                if len(header) != 32:
                    raise ValueError(f"Invalid KNM header: {path}")
                magic, version, meta_len, payload_len, reserved = struct.unpack(
                    "<8sIIQQ", header
                )
                if (
                    magic != b"KOTONNM\0"
                    or version != 1
                    or reserved
                    or meta_len > 65536
                    or path.stat().st_size != 32 + meta_len + payload_len
                ):
                    raise ValueError(f"Invalid KNM header: {path}")
                metadata = json.loads(stream.read(meta_len))
            if metadata.get("source_sha256") != original["sha256"]:
                raise ValueError(f"KNM source checksum differs: {path}")
            sources[path.name] = path

    generated = {
        "accent_dict.csv": merge_dictionaries(
            [base_dictionary, sources["accent_dict_jsut.csv"]]
        ),
    }
    manifest = {
        "schema_version": 1,
        "model_format": "knm-v1" if native else "onnx",
        "kotonoha_revision": source_revision,
        "source": lock,
        "base_dictionary_sha256": file_digest(base_dictionary),
        "files": [
            {"file": name, "size": path.stat().st_size, "sha256": file_digest(path)}
            for name, path in sorted(sources.items())
        ]
        + [
            {
                "file": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in generated.items()
        ],
    }
    generated["models.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / BUNDLE_NAME
    # checkout 時刻・OS によらず同じ入力から同じアーカイブを作る。
    with (
        archive_path.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name in sorted(sources.keys() | generated.keys()):
            info = tarfile.TarInfo(f"kotonoha-models/{name}")
            info.mode = 0o644
            if name in generated:
                data = generated[name]
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.size = sources[name].stat().st_size
                with sources[name].open("rb") as stream:
                    archive.addfile(info, stream)
    checksum_path = output_dir / f"{BUNDLE_NAME}.sha256"
    checksum_path.write_text(
        f"{file_digest(archive_path)}  {BUNDLE_NAME}\n", encoding="utf-8"
    )
    return archive_path


def main():
    """CLI 引数を解析し、配布用アーカイブを作る。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "target/model-release"
    )
    parser.add_argument("--lock", type=Path, default=ROOT / ".github/model-bundle.json")
    parser.add_argument(
        "--base-dictionary", type=Path, default=ROOT / "data/accent_dict.csv"
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--native",
        action="store_true",
        default=True,
        help="Package converted KNM models (default)",
    )
    parser.add_argument(
        "--legacy-onnx",
        action="store_false",
        dest="native",
        help="Package legacy ONNX for comparison",
    )
    args = parser.parse_args()
    try:
        archive = package_models(
            args.models_dir,
            args.output_dir,
            args.lock,
            args.base_dictionary,
            args.source_revision,
            native=args.native,
        )
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Model packaging failed: {error}\n")
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
