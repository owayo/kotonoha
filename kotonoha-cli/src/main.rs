//! kotonoha CLI - 日本語韻律解析コマンドラインツール
//!
//! テキストの形態素解析、ラベル生成、韻律記号抽出などを行う。

use clap::{Parser, Subcommand, ValueEnum};
use kotonoha::njd::InputToken;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Instant;

/// 日本語韻律エンジンのコマンドライン入口
#[derive(Parser)]
#[command(name = "kotonoha", version, about = "Japanese prosody analysis tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// テキストから形態素解析・ラベル生成・韻律抽出まで実行する。
    Analyze {
        /// 解析するテキスト
        text: String,

        /// 形態素解析辞書（.hsd）のパス
        #[arg(long)]
        dict: Option<PathBuf>,

        /// アクセント辞書（CSV）のパス
        #[arg(long)]
        accent_dict: Option<PathBuf>,

        /// 出力形式
        #[arg(long, value_enum, default_value_t = AnalyzeFormat::Labels)]
        format: AnalyzeFormat,

        /// 形態素トークンも表示する。
        #[arg(long)]
        tokens: bool,
    },

    /// 形態素解析だけを実行する。
    Tokenize {
        /// 分割するテキスト
        text: String,

        /// 形態素解析辞書（.hsd）のパス
        #[arg(long)]
        dict: PathBuf,

        /// 出力形式
        #[arg(long, value_enum, default_value_t = TokenizeFormat::Mecab)]
        format: TokenizeFormat,
    },

    /// 形態素解析済み JSON からラベルを生成する。
    Label {
        /// トークンの JSON 配列（省略時は標準入力から読む）
        tokens_json: Option<String>,

        /// 出力形式
        #[arg(long, value_enum, default_value_t = LabelFormat::Labels)]
        format: LabelFormat,
    },

    /// 内部ベンチマークを実行する。
    Bench,

    /// CSV 学習データから CRF アクセント予測器を学習する。
    TrainCrf {
        /// CSV 学習データのパス（surface,pos,reading,accent_type）
        #[arg(long)]
        data: PathBuf,

        /// バイナリ重みファイルの出力先
        #[arg(long, short)]
        output: PathBuf,

        /// 学習率
        #[arg(long, default_value_t = 0.1)]
        lr: f32,

        /// 学習エポック数
        #[arg(long, default_value_t = 30)]
        epochs: usize,

        /// L2 正則化係数
        #[arg(long, default_value_t = 0.01)]
        l2_reg: f32,
    },

    /// MeCab IPAdic のソースから形態素解析辞書を構築する。
    BuildDict {
        /// MeCab CSV ファイルを含むディレクトリ
        #[arg(long)]
        csv_dir: PathBuf,

        /// matrix.def のパス
        #[arg(long)]
        matrix: PathBuf,

        /// unk.def のパス
        #[arg(long)]
        unk: PathBuf,

        /// char.def のパス
        #[arg(long)]
        char_def: PathBuf,

        /// 出力する .hsd ファイルのパス
        #[arg(long, short)]
        output: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
enum AnalyzeFormat {
    Labels,
    PhoneTones,
    Prosody,
    Json,
}

#[derive(Clone, ValueEnum)]
enum TokenizeFormat {
    Mecab,
    Wakachi,
    Json,
}

#[derive(Clone, ValueEnum)]
enum LabelFormat {
    Labels,
    PhoneTones,
    Prosody,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            text,
            dict,
            accent_dict,
            format,
            tokens: show_tokens,
        } => cmd_analyze(
            &text,
            dict.as_deref(),
            accent_dict.as_deref(),
            &format,
            show_tokens,
        ),
        Commands::Tokenize { text, dict, format } => cmd_tokenize(&text, &dict, &format),
        Commands::Label {
            tokens_json,
            format,
        } => cmd_label(tokens_json.as_deref(), &format),
        Commands::Bench => cmd_bench(),
        Commands::TrainCrf {
            data,
            output,
            lr,
            epochs,
            l2_reg,
        } => cmd_train_crf(&data, &output, lr, epochs, l2_reg),
        Commands::BuildDict {
            csv_dir,
            matrix,
            unk,
            char_def,
            output,
        } => cmd_build_dict(&csv_dir, &matrix, &unk, &char_def, &output),
    }
}

fn cmd_analyze(
    text: &str,
    dict_path: Option<&std::path::Path>,
    accent_dict_path: Option<&std::path::Path>,
    format: &AnalyzeFormat,
    show_tokens: bool,
) {
    let mut engine = kotonoha::Engine::with_default_rules();

    if let Some(path) = dict_path
        && let Err(e) = engine.load_dictionary(path)
    {
        eprintln!("Error loading dictionary: {e}");
        std::process::exit(1);
    }

    if let Some(path) = accent_dict_path
        && let Err(e) = engine.load_accent_dict(path)
    {
        eprintln!("Error loading accent dictionary: {e}");
        std::process::exit(1);
    }

    if dict_path.is_some() {
        // Use text-based analysis (includes morphological analysis)
        if show_tokens {
            eprintln!("(token display requires dictionary-based tokenization)");
        }

        match format {
            AnalyzeFormat::Labels => match engine.text_to_labels(text) {
                Ok(labels) => {
                    for label in &labels {
                        println!("{label}");
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            },
            AnalyzeFormat::PhoneTones => match engine.text_to_phone_tones(text) {
                Ok(phone_tones) => {
                    for pt in &phone_tones {
                        println!("{}\t{}", pt.phone, pt.tone);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            },
            AnalyzeFormat::Prosody => match engine.text_to_prosody_symbols(text) {
                Ok(symbols) => {
                    println!("{}", symbols.join(""));
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            },
            AnalyzeFormat::Json => match engine.text_to_labels(text) {
                Ok(labels) => {
                    let json = serde_json::to_string_pretty(&labels).unwrap();
                    println!("{json}");
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            },
        }
    } else {
        eprintln!("Warning: No dictionary specified. Use --dict to specify a .hsd dictionary.");
        eprintln!(
            "Without a dictionary, use the 'label' subcommand with pre-tokenized JSON input."
        );
        std::process::exit(1);
    }
}

fn cmd_tokenize(text: &str, dict_path: &std::path::Path, format: &TokenizeFormat) {
    let mut analyzer = match hasami::Analyzer::load(dict_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error loading dictionary: {e}");
            std::process::exit(1);
        }
    };

    let tokens = analyzer.tokenize(text);

    match format {
        TokenizeFormat::Mecab => {
            for token in &tokens {
                println!(
                    "{}\t{},{},{},{}",
                    token.surface, token.pos, token.base_form, token.reading, token.pronunciation
                );
            }
            println!("EOS");
        }
        TokenizeFormat::Wakachi => {
            let surfaces: Vec<&str> = tokens.iter().map(|t| t.surface.as_ref()).collect();
            println!("{}", surfaces.join(" "));
        }
        TokenizeFormat::Json => {
            let input_tokens: Vec<InputToken> = tokens.into_iter().map(InputToken::from).collect();
            let json = serde_json::to_string_pretty(&input_tokens).unwrap();
            println!("{json}");
        }
    }
}

fn cmd_label(tokens_json: Option<&str>, format: &LabelFormat) {
    let json_str = if let Some(json) = tokens_json {
        json.to_string()
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {e}");
            std::process::exit(1);
        });
        buf
    };

    let tokens: Vec<InputToken> = match serde_json::from_str(&json_str) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error parsing JSON: {e}");
            eprintln!("Expected JSON array of objects with fields: surface, pos, reading, ...");
            std::process::exit(1);
        }
    };

    let engine = kotonoha::Engine::with_default_rules();

    match format {
        LabelFormat::Labels => {
            let labels = engine.tokens_to_labels(&tokens);
            for label in &labels {
                println!("{label}");
            }
        }
        LabelFormat::PhoneTones => {
            let phone_tones = engine.tokens_to_phone_tones(&tokens);
            for pt in &phone_tones {
                println!("{}\t{}", pt.phone, pt.tone);
            }
        }
        LabelFormat::Prosody => {
            let symbols = engine.tokens_to_prosody_symbols(&tokens);
            println!("{}", symbols.join(""));
        }
    }
}

fn cmd_bench() {
    println!("kotonoha internal benchmarks");
    println!("{}", "=".repeat(60));

    let iterations = 1000;

    // Sample tokens for benchmarking
    let sample_tokens = vec![
        InputToken::new("今日", "名詞", "キョウ", "キョー"),
        InputToken::new("は", "助詞", "ワ", "ワ"),
        InputToken::new("良い", "形容詞", "ヨイ", "ヨイ"),
        InputToken::new("天気", "名詞", "テンキ", "テンキ"),
        InputToken::new("です", "助動詞", "デス", "デス"),
    ];

    let engine = kotonoha::Engine::with_default_rules();

    // Benchmark: mora counting
    let words = ["キョウ", "コンニチワ", "トーキョー", "ガッコー", "シャシン"];
    let start = Instant::now();
    for _ in 0..iterations {
        for word in &words {
            let _ = kotonoha::mora::count_mora(word);
        }
    }
    let elapsed = start.elapsed();
    let ops_per_sec = (iterations * words.len()) as f64 / elapsed.as_secs_f64();
    println!(
        "mora count:        {:>10.0} ops/sec  ({:.3} ms for {} iterations)",
        ops_per_sec,
        elapsed.as_secs_f64() * 1000.0,
        iterations * words.len()
    );

    // Benchmark: phoneme conversion
    let kana_words = [
        "カキクケコ",
        "シャシン",
        "コンニチワ",
        "トーキョー",
        "ガッコー",
    ];
    let start = Instant::now();
    for _ in 0..iterations {
        for word in &kana_words {
            let _ = kotonoha::phoneme::katakana_to_phonemes(word);
        }
    }
    let elapsed = start.elapsed();
    let ops_per_sec = (iterations * kana_words.len()) as f64 / elapsed.as_secs_f64();
    println!(
        "phoneme convert:   {:>10.0} ops/sec  ({:.3} ms for {} iterations)",
        ops_per_sec,
        elapsed.as_secs_f64() * 1000.0,
        iterations * kana_words.len()
    );

    // Benchmark: analyze tokens
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = engine.analyze(&sample_tokens);
    }
    let elapsed = start.elapsed();
    let sents_per_sec = iterations as f64 / elapsed.as_secs_f64();
    let tokens_per_sec = (iterations * sample_tokens.len()) as f64 / elapsed.as_secs_f64();
    println!(
        "analyze tokens:    {:>10.0} sents/sec ({:.0} tokens/sec, {:.3} ms)",
        sents_per_sec,
        tokens_per_sec,
        elapsed.as_secs_f64() * 1000.0
    );

    // Benchmark: estimate accent
    let start = Instant::now();
    for _ in 0..iterations {
        let mut nodes = engine.analyze(&sample_tokens);
        let _ = engine.estimate_accent(&mut nodes);
    }
    let elapsed = start.elapsed();
    let sents_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "estimate accent:   {:>10.0} sents/sec  ({:.3} ms for {} iterations)",
        sents_per_sec,
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );

    // Benchmark: generate labels
    let start = Instant::now();
    for _ in 0..iterations {
        let mut nodes = engine.analyze(&sample_tokens);
        let phrases = engine.estimate_accent(&mut nodes);
        let _ = engine.make_label(&nodes, &phrases);
    }
    let elapsed = start.elapsed();
    let sents_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "generate labels:   {:>10.0} sents/sec  ({:.3} ms for {} iterations)",
        sents_per_sec,
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );

    // Benchmark: tokens_to_labels (full pipeline)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = engine.tokens_to_labels(&sample_tokens);
    }
    let elapsed = start.elapsed();
    let sents_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "tokens_to_labels:  {:>10.0} sents/sec  ({:.3} ms for {} iterations)",
        sents_per_sec,
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );

    // Benchmark: prosody extraction
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = engine.tokens_to_prosody_symbols(&sample_tokens);
    }
    let elapsed = start.elapsed();
    let sents_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "prosody extract:   {:>10.0} sents/sec  ({:.3} ms for {} iterations)",
        sents_per_sec,
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );

    println!("{}", "=".repeat(60));
    println!("Done.");
}

fn cmd_build_dict(
    csv_dir: &std::path::Path,
    matrix_path: &std::path::Path,
    unk_path: &std::path::Path,
    char_def_path: &std::path::Path,
    output_path: &std::path::Path,
) {
    let total_start = Instant::now();

    eprintln!("[1/3] CSVファイルを読み込み中...");
    let mut builder = hasami::dict::DictBuilder::new();

    if let Err(e) = builder.add_csv_dir(csv_dir) {
        eprintln!("CSVディレクトリの読み込みに失敗: {e}");
        std::process::exit(1);
    }
    eprintln!("  エントリ数: {}", builder.entry_count());

    eprintln!("[2/3] 接続行列・未知語定義を読み込み中...");
    if let Err(e) = builder.load_matrix(matrix_path) {
        eprintln!("matrix.def の読み込みに失敗: {e}");
        std::process::exit(1);
    }
    if let Err(e) = builder.load_unk(unk_path) {
        eprintln!("unk.def の読み込みに失敗: {e}");
        std::process::exit(1);
    }
    if let Err(e) = builder.load_char_def(char_def_path) {
        eprintln!("char.def の読み込みに失敗: {e}");
        std::process::exit(1);
    }

    eprintln!("[3/3] Double-Array Trieを構築してバイナリ辞書を書き出し中...");
    let trie_start = Instant::now();
    let options = builder.write_options();
    if let Err(e) = builder.write_hsd(output_path, &options, |processed, total| {
        if processed % 50_000 == 0 {
            eprint!("\r  進捗: {processed}/{total} キー");
        }
    }) {
        eprintln!("辞書の構築・書き出しに失敗: {e}");
        std::process::exit(1);
    }
    eprintln!(
        "\r  辞書の構築・書き出し完了 ({:.2}秒)          ",
        trie_start.elapsed().as_secs_f64()
    );

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    let total_elapsed = total_start.elapsed();

    eprintln!();
    eprintln!("=== 辞書ビルド完了 ===");
    eprintln!("  出力ファイル: {}", output_path.display());
    eprintln!("  ファイルサイズ: {:.2} MB", file_size as f64 / 1_048_576.0);
    eprintln!("  所要時間: {:.2}秒", total_elapsed.as_secs_f64());
}

fn cmd_train_crf(
    data_path: &std::path::Path,
    output_path: &std::path::Path,
    learning_rate: f32,
    num_epochs: usize,
    l2_reg: f32,
) {
    use kotonoha::crf::{CrfTrainer, TrainingExample};
    use kotonoha::njd::{NjdNode, Pos};

    let total_start = Instant::now();

    eprintln!("[1/3] 学習データを読み込み中...");
    eprintln!("  ファイル: {}", data_path.display());

    let content = match std::fs::read_to_string(data_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ファイルの読み込みに失敗: {e}");
            std::process::exit(1);
        }
    };

    // CSV は4列の v1 と9列の v2 を受け付け、空行で発話を区切る。
    // v1: surface,pos,reading,accent_type
    // v2: surface,pos,pos_detail1,pos_detail2,conjugation_type,conjugation_form,reading,pronunciation,accent_type
    let mut examples: Vec<TrainingExample> = Vec::new();
    let mut current_nodes: Vec<NjdNode> = Vec::new();
    let mut current_labels: Vec<u8> = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();

        // コメント行を読み飛ばす。
        if line.starts_with('#') {
            continue;
        }

        // 空行で発話を確定する。
        if line.is_empty() {
            if !current_nodes.is_empty() {
                examples.push(TrainingExample {
                    nodes: std::mem::take(&mut current_nodes),
                    labels: std::mem::take(&mut current_labels),
                });
            }
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 4 && fields.len() != 9 {
            eprintln!(
                "{}:{}: CRF学習データは4列または9列が必要です（{}列）",
                data_path.display(),
                line_idx + 1,
                fields.len()
            );
            std::process::exit(1);
        }
        let accent_field = if fields.len() == 9 {
            fields[8]
        } else {
            fields[3]
        };
        let accent_type: u8 = accent_field.parse().unwrap_or_else(|_| {
            eprintln!(
                "{}:{}: 不正なアクセント型: {accent_field}",
                data_path.display(),
                line_idx + 1
            );
            std::process::exit(1);
        });

        let node = if fields.len() == 9 {
            // v2 のノードを構築する。
            let surface = fields[0];
            let pos_str = fields[1];
            let pos_detail1 = fields[2];
            let pos_detail2 = fields[3];
            let ctype = fields[4];
            let cform = fields[5];
            let reading = fields[6];
            let pronunciation = fields[7];
            let pos = Pos::parse(pos_str);
            let mut token = InputToken::new(surface, pos.to_label_str(), reading, pronunciation);
            token.pos_detail1 = pos_detail1.to_string();
            token.pos_detail2 = pos_detail2.to_string();
            token.ctype = ctype.to_string();
            token.cform = cform.to_string();
            let mut node = NjdNode::from_token(&token);
            node.accent_type = accent_type;
            node
        } else {
            // v1 のノードを構築する。
            let surface = fields[0];
            let pos_str = fields[1];
            let reading = fields[2];
            let pos = Pos::parse(pos_str);
            let token = InputToken::new(surface, pos.to_label_str(), reading, reading);
            let mut node = NjdNode::from_token(&token);
            node.accent_type = accent_type;
            node
        };

        let accent_type = node.accent_type;
        current_nodes.push(node);
        current_labels.push(accent_type);
    }

    // 最後の発話も確定する。
    if !current_nodes.is_empty() {
        examples.push(TrainingExample {
            nodes: current_nodes,
            labels: current_labels,
        });
    }

    let total_nodes: usize = examples.iter().map(|e| e.nodes.len()).sum();
    eprintln!("  発話数: {}", examples.len());
    eprintln!("  総ノード数: {total_nodes}");

    eprintln!("[2/3] CRFモデルを学習中...");
    eprintln!("  学習率: {learning_rate}");
    eprintln!("  エポック数: {num_epochs}");
    eprintln!("  L2正則化: {l2_reg}");

    let trainer = CrfTrainer::new(learning_rate, num_epochs).with_l2_reg(l2_reg);
    let predictor = trainer.train(&examples).unwrap_or_else(|e| {
        eprintln!("CRF学習データが不正です: {e}");
        std::process::exit(1);
    });

    eprintln!("[3/3] モデルを保存中...");
    if let Err(e) = CrfTrainer::save_weights(&predictor, output_path) {
        eprintln!("モデルの保存に失敗: {e}");
        std::process::exit(1);
    }

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    let total_elapsed = total_start.elapsed();

    // 学習データで精度を確認する。
    use kotonoha::nn::AccentPredictor;
    let mut correct = 0usize;
    let mut total = 0usize;
    for example in &examples {
        let predicted = predictor.predict(&example.nodes);
        for (pred, gold) in predicted.iter().zip(example.labels.iter()) {
            if pred == gold {
                correct += 1;
            }
            total += 1;
        }
    }
    let train_acc = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    eprintln!();
    eprintln!("=== CRF学習完了 ===");
    eprintln!("  出力ファイル: {}", output_path.display());
    eprintln!("  ファイルサイズ: {:.2} KB", file_size as f64 / 1024.0);
    eprintln!("  学習データ正解率: {:.2}%", train_acc * 100.0);
    eprintln!("  所要時間: {:.2}秒", total_elapsed.as_secs_f64());
}
