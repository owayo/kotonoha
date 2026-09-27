# kotonoha の開発タスク。make だけで一覧を表示する。
# 版は mise.toml に固定し、IDE からも mise exec 経由で同じツールを使う。
# SYSTEM_TOOLS=1 は PATH 上のツールを使う（版の一致は保証しない）。

.DEFAULT_GOAL := help

BINARY_NAME := kotonoha
INSTALL_PATH ?= /usr/local/bin
CARGO_FLAGS ?= --locked

MISE_CANDIDATES ?= $(HOME)/.local/bin/mise /opt/homebrew/bin/mise /usr/local/bin/mise
NO_MISE_TARGETS := help
ifeq ($(SYSTEM_TOOLS),1)
RUN :=
else
ifndef MISE
MISE := $(firstword $(shell command -v mise 2>/dev/null) $(wildcard $(MISE_CANDIDATES)))
endif
ifeq ($(MISE),)
ifneq ($(filter-out $(NO_MISE_TARGETS),$(or $(MAKECMDGOALS),help)),)
$(error mise が見つかりません。https://mise.jdx.dev から導入するか SYSTEM_TOOLS=1 を指定してください)
endif
endif
RUN := $(if $(MISE),$(MISE) exec --,)
endif

.PHONY: help setup build release run test test-python lint fmt fmt-check check ci install uninstall clean setup-training build-cuda

setup: ## ツールチェーン (mise) と依存を取得する
	@if [ -n "$(MISE)" ]; then "$(MISE)" install; fi
	$(RUN) cargo fetch $(CARGO_FLAGS)
	$(RUN) uv sync --locked --project kotonoha-python

build: ## デバッグ版をビルドする
	$(RUN) cargo build $(CARGO_FLAGS) --workspace

release: ## リリース版をビルドする
	$(RUN) cargo build --release $(CARGO_FLAGS) --workspace

build-cuda: ## ONNX 推論を有効にしてリリース版をビルドする
	$(RUN) cargo build --release $(CARGO_FLAGS) --workspace --all-features

run: ## デバッグ版を実行する (引数は ARGS="...")
	$(RUN) cargo run $(CARGO_FLAGS) -p kotonoha-cli -- $(ARGS)

# PyO3 の extension-module は Python からロードしてテストする。
test: test-python ## テストを実行する
	$(RUN) cargo test $(CARGO_FLAGS) --workspace --exclude kotonoha-python --no-fail-fast
	$(RUN) cargo test $(CARGO_FLAGS) -p kotonoha --all-features --no-fail-fast

test-python: ## Python バインディングの公開 API を検査する
	$(RUN) uv run --locked --project kotonoha-python python -m unittest discover -s kotonoha-python/tests -v

lint: ## clippy を警告ゼロで通す
	$(RUN) cargo clippy $(CARGO_FLAGS) --workspace --all-targets -- -D warnings
	$(RUN) cargo clippy $(CARGO_FLAGS) --workspace --all-targets --all-features -- -D warnings

fmt: ## コードを整形する (書き換える)
	$(RUN) cargo fmt --all

fmt-check: ## 整形済みかを確かめる (書き換えない)
	$(RUN) cargo fmt --all -- --check

check: fmt-check lint ## 整形と静的検査 (書き換えない)

ci: check test ## CI と同じ検査 (書き換えない)

# 学習用の大きな依存は通常の setup と CI には含めない。
setup-training: ## 学習・評価用の Python 依存を取得する
	$(RUN) uv sync --locked --project training

# 同じディレクトリの一時ファイルから置き換え、macOS の署名キャッシュとの衝突を避ける。
install: release ## リリース版を INSTALL_PATH (既定 /usr/local/bin) に入れる
	@mkdir -p "$(INSTALL_PATH)"
	cp "target/release/$(BINARY_NAME)" "$(INSTALL_PATH)/$(BINARY_NAME).new"
	mv -f "$(INSTALL_PATH)/$(BINARY_NAME).new" "$(INSTALL_PATH)/$(BINARY_NAME)"

uninstall: ## INSTALL_PATH から取り除く
	rm -f "$(INSTALL_PATH)/$(BINARY_NAME)"

clean: ## ビルド成果物を消す
	$(RUN) cargo clean

help: ## このヘルプを表示する
	@echo "kotonoha の開発タスク"
	@echo "使い方: make <target>"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'
	@echo "版は mise.toml に固定しています。最初に make setup を実行してください。"
