//! kotonoha-models の固定版。ONNX の値は Git LFS ポインタの oid と size。

use super::ModelFile;

/// 配布元 `owayo/kotonoha-models` のコミット。
pub const MODEL_REVISION: &str = "90820175ff715e03631d640e250a68c75d940394";

/// 役割名を固定した 13 KNM と enrich 用アクセント辞書。検証値は変換元に対応する。
pub const MODEL_FILES: [ModelFile; 14] = [
    ModelFile {
        file: "teacher.knm",
        source: "accent_model_v24.onnx",
        size: 18814945,
        sha256: "94ab4099de64a55d1ffb232d892d2e899a5e84e550650eb736a87b2516fe2768",
    },
    ModelFile {
        file: "student-01.knm",
        source: "accent_model_v38.onnx",
        size: 18815137,
        sha256: "a274dd714b0d3629b06b0a4c7c9975748b24332eecfc84e6390cc9f17791606d",
    },
    ModelFile {
        file: "student-02.knm",
        source: "accent_model_v54_split1.onnx",
        size: 18815137,
        sha256: "0161a96d316cede4e5dbaa067aa2c5f52239323c525103c48dca3242b0972e3b",
    },
    ModelFile {
        file: "student-03.knm",
        source: "accent_model_v54_split2.onnx",
        size: 18815137,
        sha256: "b42cba8eec1ab47389c3c6a7bb239a878ad58c166d644d21147f386eb170a13b",
    },
    ModelFile {
        file: "student-04.knm",
        source: "accent_model_v54_split3.onnx",
        size: 18815137,
        sha256: "feac6eab841c9c606397e68ca3f277fa3415463f1bd709925875458da3d689a2",
    },
    ModelFile {
        file: "student-05.knm",
        source: "accent_model_v59_fold0.onnx",
        size: 18815137,
        sha256: "151a3d810759d2bd2404a720c69287aadec2f428c7073d0cdc5e4d53b004d775",
    },
    ModelFile {
        file: "student-06.knm",
        source: "accent_model_v59_fold1.onnx",
        size: 18815137,
        sha256: "f58b73d7e0065fa2b19d4ef6245b562ced339e4dd410111d29cc159a1d8c421e",
    },
    ModelFile {
        file: "student-07.knm",
        source: "accent_model_v59_fold2.onnx",
        size: 18815137,
        sha256: "f9806aee8a91772a3924989893358294f7cb54dbaaa2ee21fe13298491f1a95a",
    },
    ModelFile {
        file: "student-08.knm",
        source: "accent_model_v59_fold3.onnx",
        size: 18815137,
        sha256: "7db92fc17e1ca82dcf9849f629925194a452c5e9047e796711835b1617954ad6",
    },
    ModelFile {
        file: "student-09.knm",
        source: "accent_model_v59_fold4.onnx",
        size: 18815137,
        sha256: "a01986dac2d47501e22906aec6d0744362e814819b8638e8332bddf446cae7a7",
    },
    ModelFile {
        file: "meta-01.knm",
        source: "accent_model_v61.onnx",
        size: 18815777,
        sha256: "d4b80933e8c2826cd61a7dd2777d787594ba46fd85de10e3055c05cf29ff5642",
    },
    ModelFile {
        file: "meta-02.knm",
        source: "accent_model_v63.onnx",
        size: 18815777,
        sha256: "b00d03a87f634afe0fc12ae3bd845f4077f9b450abc58827a6c905df270ca90c",
    },
    ModelFile {
        file: "accent.knm",
        source: "accent_model_v66_split1.onnx",
        size: 18820833,
        sha256: "5b97a0b3b1d73ecdf8c20336fcc121b6dfbb9a1d3eaa20733f0a44f461451859",
    },
    ModelFile {
        file: "accent_dict.csv",
        source: "accent_dict_jsut.csv",
        size: 272398,
        sha256: "005c1c6717baa99a1b8823cdee899a0057ae20bff9523a95d3bbade152d52b02",
    },
];
