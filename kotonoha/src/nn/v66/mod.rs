//! v66 系アクセント予測ランタイム
//!
//! 12 ONNX を束ねた multi-stage パイプライン (v24 → 9 student → v61/v63 → v66_split1)。
//! 訓練側 `kotonoha/training/train_onnx_v60.py` の特徴抽出と数値完全一致させる。

#[cfg(feature = "cuda")]
pub mod bundle;
pub mod enrich;
pub mod features;
pub mod math;
#[cfg(feature = "cuda")]
pub mod pipeline;
pub mod vocab;

#[cfg(feature = "cuda")]
pub use bundle::{BundleError, V66Bundle, V66Models};
#[cfg(feature = "cuda")]
pub use pipeline::{V66Pipeline, V66PredictError};
