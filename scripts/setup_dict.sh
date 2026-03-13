#!/usr/bin/env bash
#
# setup_dict.sh - MeCab IPAdic辞書のダウンロード・ビルド自動化スクリプト
#
# MeCab IPAdic辞書をダウンロードし、hasami形態素解析エンジン用の
# .hsd (Hasami Serialized Dictionary) バイナリ辞書を構築する。
#
# 使い方:
#   ./scripts/setup_dict.sh
#
# 出力:
#   data/ipadic.hsd
#

set -euo pipefail

# --- 設定 ---
IPADIC_VERSION="2.7.0-20070801"
IPADIC_URL="https://sourceforge.net/projects/mecab/files/mecab-ipadic/2.7.0-20070801/mecab-ipadic-${IPADIC_VERSION}.tar.gz/download"
IPADIC_SHA256="b62f527d881c504576baed9c6ef6561554658b175ce6ae0096a60307e49e3523"

# プロジェクトルートを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data"
TMPDIR_BASE="${PROJECT_ROOT}/target/dict_build"
IPADIC_DIR="${TMPDIR_BASE}/mecab-ipadic-${IPADIC_VERSION}"
ARCHIVE_PATH="${TMPDIR_BASE}/mecab-ipadic-${IPADIC_VERSION}.tar.gz"
OUTPUT_FILE="${DATA_DIR}/ipadic.hsd"

# --- ユーティリティ関数 ---

log_info() {
    echo "[情報] $*"
}

log_warn() {
    echo "[警告] $*" >&2
}

log_error() {
    echo "[エラー] $*" >&2
}

die() {
    log_error "$@"
    exit 1
}

# --- 前提条件チェック ---

check_prerequisites() {
    log_info "前提条件を確認中..."

    local missing=()

    if ! command -v cargo &>/dev/null; then
        missing+=("cargo (Rustツールチェイン)")
    fi

    if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
        missing+=("curl または wget")
    fi

    if ! command -v tar &>/dev/null; then
        missing+=("tar")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        die "以下のツールがインストールされていません: ${missing[*]}"
    fi

    log_info "前提条件OK"
}

# --- ダウンロード ---

download_ipadic() {
    mkdir -p "${TMPDIR_BASE}"

    if [[ -d "${IPADIC_DIR}" ]]; then
        log_info "IPAdic辞書ソースは既にダウンロード済み: ${IPADIC_DIR}"
        return 0
    fi

    if [[ -f "${ARCHIVE_PATH}" ]]; then
        log_info "アーカイブは既にダウンロード済み: ${ARCHIVE_PATH}"
    else
        log_info "MeCab IPAdic ${IPADIC_VERSION} をダウンロード中..."
        if command -v curl &>/dev/null; then
            curl -fSL --retry 3 --retry-delay 5 -o "${ARCHIVE_PATH}" "${IPADIC_URL}" \
                || die "ダウンロードに失敗しました"
        else
            wget --tries=3 --waitretry=5 -O "${ARCHIVE_PATH}" "${IPADIC_URL}" \
                || die "ダウンロードに失敗しました"
        fi
        log_info "ダウンロード完了"
    fi

    # SHA256チェックサム検証 (利用可能な場合)
    if command -v sha256sum &>/dev/null; then
        log_info "チェックサムを検証中..."
        local actual_hash
        actual_hash="$(sha256sum "${ARCHIVE_PATH}" | cut -d' ' -f1)"
        if [[ "${actual_hash}" != "${IPADIC_SHA256}" ]]; then
            rm -f "${ARCHIVE_PATH}"
            die "チェックサム不一致。期待値: ${IPADIC_SHA256}, 実際: ${actual_hash}"
        fi
        log_info "チェックサム検証OK"
    elif command -v shasum &>/dev/null; then
        log_info "チェックサムを検証中..."
        local actual_hash
        actual_hash="$(shasum -a 256 "${ARCHIVE_PATH}" | cut -d' ' -f1)"
        if [[ "${actual_hash}" != "${IPADIC_SHA256}" ]]; then
            rm -f "${ARCHIVE_PATH}"
            die "チェックサム不一致。期待値: ${IPADIC_SHA256}, 実際: ${actual_hash}"
        fi
        log_info "チェックサム検証OK"
    else
        log_warn "sha256sum/shasumが見つかりません。チェックサム検証をスキップします。"
    fi

    # 展開
    log_info "アーカイブを展開中..."
    tar -xzf "${ARCHIVE_PATH}" -C "${TMPDIR_BASE}" \
        || die "アーカイブの展開に失敗しました"
    log_info "展開完了: ${IPADIC_DIR}"
}

# --- 辞書ビルド ---

build_hsd() {
    log_info "kotonoha-cli をビルド中..."

    cargo build \
        --manifest-path "${PROJECT_ROOT}/Cargo.toml" \
        --release \
        -p kotonoha-cli \
        || die "kotonoha-cli のビルドに失敗しました"

    local cli_bin="${PROJECT_ROOT}/target/release/kotonoha"
    if [[ ! -f "${cli_bin}" ]]; then
        cli_bin="${PROJECT_ROOT}/target/release/kotonoha.exe"
    fi
    if [[ ! -f "${cli_bin}" ]]; then
        die "kotonoha バイナリが見つかりません"
    fi

    mkdir -p "${DATA_DIR}"

    log_info "辞書をビルド中..."
    "${cli_bin}" build-dict \
        --csv-dir "${IPADIC_DIR}" \
        --matrix "${IPADIC_DIR}/matrix.def" \
        --unk "${IPADIC_DIR}/unk.def" \
        --char-def "${IPADIC_DIR}/char.def" \
        --output "${OUTPUT_FILE}" \
        || die "辞書のビルドに失敗しました"
}

# --- 検証 ---

verify_output() {
    if [[ ! -f "${OUTPUT_FILE}" ]]; then
        die "出力ファイルが見つかりません: ${OUTPUT_FILE}"
    fi

    local file_size
    file_size=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || stat -f%z "${OUTPUT_FILE}" 2>/dev/null || echo "0")

    if [[ "${file_size}" -lt 1000 ]]; then
        die "出力ファイルが小さすぎます (${file_size} bytes)。ビルドに問題がある可能性があります。"
    fi

    # マジックバイトを確認
    local magic
    magic="$(head -c 8 "${OUTPUT_FILE}" | tr -d '\0')"
    if [[ "${magic}" != "HSMDICT" ]]; then
        die "出力ファイルのマジックバイトが不正です: ${magic}"
    fi

    log_info "辞書ファイルの検証OK (${file_size} bytes)"
}

# --- クリーンアップ ---

cleanup_temp() {
    if [[ "${KEEP_TEMP:-0}" == "1" ]]; then
        log_info "一時ファイルを保持します: ${TMPDIR_BASE}"
        return 0
    fi

    log_info "一時ファイルを削除中..."
    # IPAdicソースとアーカイブは再ビルド時のために保持
    log_info "クリーンアップ完了 (IPAdicソースは保持)"
}

# --- メイン処理 ---

main() {
    echo "============================================"
    echo "  hasami辞書セットアップスクリプト"
    echo "  IPAdic ${IPADIC_VERSION}"
    echo "============================================"
    echo

    # 既にビルド済みかチェック
    if [[ -f "${OUTPUT_FILE}" ]] && [[ "${FORCE:-0}" != "1" ]]; then
        log_info "辞書ファイルは既に存在します: ${OUTPUT_FILE}"
        log_info "再ビルドするには FORCE=1 を指定してください。"
        exit 0
    fi

    check_prerequisites
    download_ipadic
    build_hsd
    verify_output
    cleanup_temp

    echo
    echo "============================================"
    echo "  セットアップ完了"
    echo "  辞書ファイル: ${OUTPUT_FILE}"
    echo ""
    echo "  使用例:"
    echo "    cargo run -p kotonoha-cli -- tokenize '東京都に住む' --dict ${OUTPUT_FILE}"
    echo "    cargo run -p kotonoha-cli -- analyze '今日は良い天気です' --dict ${OUTPUT_FILE}"
    echo "============================================"
}

main "$@"
