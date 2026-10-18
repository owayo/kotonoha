#!/usr/bin/env python3
"""JST の yy.m.counter で採番し、Rust と Python の公開版をそろえる。"""

import argparse
import re
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))


def next_version(current: str, now: datetime) -> str:
    """同じ年月なら連番を進め、年月が変わったら100から始める。

    Returns:
        JST の yy.m.counter 形式の版。

    """
    now = now.astimezone(JST)
    year = now.year % 100
    counter = 100
    match = re.fullmatch(r"([0-9]{2}|[0-9]{4})\.([0-9]{1,2})\.([0-9]+)", current)
    if match:
        previous_year, previous_month, previous_counter = map(int, match.groups())
        if previous_year % 100 == year and previous_month == now.month:
            counter = previous_counter + 1
    return f"{year}.{now.month}.{counter}"


def replace_version(source: str, section: str, version: str) -> str:
    """指定した TOML セクションの版だけを書き換え、依存の版を保つ。

    Returns:
        版を置換した TOML テキスト。

    Raises:
        ValueError: 対象の版が一意に見つからない場合。

    """
    pattern = (
        rf"(?ms)(^\[{re.escape(section)}\][^\S\n]*\n"
        rf'(?:(?!^\[).)*?^version[ \t]*=[ \t]*)"[^"\n]*"'
    )
    updated, count = re.subn(pattern, lambda match: f'{match[1]}"{version}"', source)
    if count != 1:
        raise ValueError(f"Expected one version in [{section}]")
    parsed = tomllib.loads(updated)
    for key in section.split("."):
        parsed = parsed[key]
    if parsed["version"] != version:
        raise ValueError(f"Failed to update [{section}] version")
    return updated


def prepare_release(root: Path, now: datetime) -> str:
    """両方の TOML を検証してから更新する。lockfile は cargo / uv で追従する。

    Returns:
        更新した公開版。

    """
    cargo_path = root / "Cargo.toml"
    python_path = root / "kotonoha-python/pyproject.toml"
    cargo = cargo_path.read_text(encoding="utf-8")
    python = python_path.read_text(encoding="utf-8")
    current = tomllib.loads(cargo)["workspace"]["package"]["version"]
    version = next_version(current, now)
    cargo = replace_version(cargo, "workspace.package", version)
    python = replace_version(python, "project", version)
    cargo_path.write_text(cargo, encoding="utf-8", newline="\n")
    python_path.write_text(python, encoding="utf-8", newline="\n")
    return version


def main():
    """版とタグを GitHub Actions の step output として出力する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        version = prepare_release(args.root, datetime.now(JST))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Release preparation failed: {error}\n")
    print(f"version={version}")
    print(f"tag=v{version}")


if __name__ == "__main__":
    main()
