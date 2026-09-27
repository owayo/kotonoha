fn main() {
    // cargo build でも Python 拡張用のリンク引数を設定する。
    pyo3_build_config::add_extension_module_link_args();
}
