# Tacotron2 アーキテクチャ理解ガイド

このドキュメントはTacotron2のコードを読み解き、VITSへの理解の架け橋とするためのガイドです。

## ドキュメント構成

| ドキュメント | 内容 |
|-------------|------|
| **CLAUDE.md** (このファイル) | 全体概要、VITSとの比較 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Encoder/Decoder/Postnetの詳細実装 |
| [docs/ATTENTION.md](docs/ATTENTION.md) | Location Sensitive Attentionの深掘り |
| [docs/WAVEGLOW.md](docs/WAVEGLOW.md) | WaveGlowボコーダの仕組み |
| [docs/TRAINING.md](docs/TRAINING.md) | 学習パイプラインの詳細 |

---

## 全体構造

```
テキスト → [Tacotron2] → メルスペクトログラム → [WaveGlow] → 波形
```

### Tacotron2

```
Tacotron2 (model.py:457-529)
├── Embedding層 (テキスト → ベクトル)
├── Encoder (model.py:149-201)
│   ├── 3層 Conv1d + BatchNorm + ReLU + Dropout
│   └── Bidirectional LSTM
├── Decoder (model.py:204-454)
│   ├── Prenet (89-100行)
│   ├── Attention RNN (LSTMCell)
│   ├── Location Sensitive Attention (29-86行)
│   ├── Decoder RNN (LSTMCell)
│   ├── Linear Projection (mel出力)
│   └── Gate Layer (終了予測)
└── Postnet (model.py:103-146)
    └── 5層 Conv1d + BatchNorm (残差接続)
```

### WaveGlow

```
WaveGlow (waveglow/glow.py)
├── Upsample (ConvTranspose1d)
└── n_flows回のフロー (12回)
    ├── Invertible 1x1 Conv
    └── WN (Affine Coupling)
```

---

## データフロー

```
テキスト入力 (B, T_text)
       │
       ▼  Embedding
(B, 512, T_text)
       │
       ▼  Encoder (3層Conv + BiLSTM)
encoder_outputs (B, T_text, 512)  ← "memory"
       │
       ▼  Decoder + Attention (自己回帰)
mel_outputs (B, 80, T_mel)
       │
       ▼  Postnet (5層Conv、残差接続)
mel_outputs_postnet (B, 80, T_mel)
       │
       ▼  WaveGlow (Flow-based)
audio (B, T_audio)
```

---

## 核心コンセプト

### 1. Location Sensitive Attention

Tacotron2の最重要部分。詳細は [docs/ATTENTION.md](docs/ATTENTION.md) を参照。

**通常のAttention:**
```
energy = v * tanh(W_query * query + W_memory * memory)
```

**Location Sensitive Attention:**
```
energy = v * tanh(W_query * query + W_location * location + W_memory * memory)
                                    ↑ 位置情報を追加
```

- **前回のAttention重み**: 「次は隣を見るべき」を学習
- **累積Attention重み**: 「すでに読んだ場所をスキップ」を学習

これにより**単調な進行（Monotonic Alignment）**を実現。

### 2. Teacher Forcing vs 自己回帰

| 学習時 | 推論時 |
|--------|--------|
| 正解メルを入力 (Teacher Forcing) | 前フレーム出力を入力 |
| 固定長でデコード | Gate予測で終了判定 |
| バッチ処理可能 | 1フレームずつ生成 |

### 3. Loss Function

```python
Total Loss = Mel Loss + Gate Loss

Mel Loss = MSE(decoder_out, target) + MSE(postnet_out, target)
Gate Loss = BCEWithLogits(gate_out, gate_target)
```

- **Mel Loss**: メルスペクトログラムの再構成誤差
- **Gate Loss**: 終了タイミングの予測（自己回帰ループを止めるため必要）

---

## VITSへの架け橋

| 項目 | Tacotron2 | VITS |
|------|-----------|------|
| **アライメント** | Location Sensitive Attention (学習で獲得) | MAS (アルゴリズムで強制) |
| **長さ制御** | Gate Loss で終了を予測 | Duration Predictor で長さを予測 |
| **出力** | メルスペクトログラム | 直接波形 |
| **ボコーダ** | WaveGlow (Flow-based) | HiFi-GAN (GAN-based) |
| **生成方式** | 自己回帰 (遅い) | 非自己回帰 (高速) |
| **学習** | 2段階 (Tacotron2 → WaveGlow) | End-to-End (1段階) |

### 進化の流れ

```
Tacotron2 (Location Sensitive Attention)
    ↓ 「単調進行を学習」という考え方
FastSpeech (Duration Predictor + Attention)
    ↓ 「長さを直接予測」という発想
VITS (MAS + Duration Predictor + Flow + GAN)
    ↓ 「単調性をアルゴリズムで保証 + End-to-End」
```

---

## ファイル構成

### Tacotron2

| ファイル | 内容 |
|---------|------|
| `model.py` | Encoder, Attention, Decoder, Postnet, Tacotron2 |
| `loss_function.py` | Tacotron2Loss (MSE + BCE) |
| `layers.py` | LinearNorm, ConvNorm, TacotronSTFT |
| `hparams.py` | ハイパーパラメータ定義 |
| `data_utils.py` | TextMelLoader, TextMelCollate |
| `train.py` | 学習ループ |
| `text/` | テキスト処理（cleaners, symbols） |

### WaveGlow

| ファイル | 内容 |
|---------|------|
| `waveglow/glow.py` | WaveGlow, WN, Invertible1x1Conv |
| `waveglow/train.py` | 学習スクリプト |
| `waveglow/inference.py` | 推論スクリプト |
| `waveglow/denoiser.py` | デノイザー |
| `waveglow/config.json` | ハイパーパラメータ |

---

## 主要ハイパーパラメータ

### モデル

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `n_symbols` | 148 | シンボル数 |
| `encoder_embedding_dim` | 512 | Encoder出力次元 |
| `n_mel_channels` | 80 | メルチャンネル数 |
| `attention_dim` | 128 | Attention次元 |
| `attention_location_kernel_size` | 31 | 位置情報のカーネルサイズ |
| `decoder_rnn_dim` | 1024 | Decoder RNN次元 |

### 音声処理

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `sampling_rate` | 22050 | サンプリングレート |
| `filter_length` | 1024 | FFTサイズ |
| `hop_length` | 256 | フレーム間隔 |
| `mel_fmin` | 0.0 | 最小周波数 |
| `mel_fmax` | 8000.0 | 最大周波数 |

---

## クイックリファレンス

### 重要な行番号

| コンポーネント | ファイル | 行 |
|---------------|----------|-----|
| LocationLayer | model.py | 10-26 |
| Attention | model.py | 29-86 |
| Prenet | model.py | 89-100 |
| Postnet | model.py | 103-146 |
| Encoder | model.py | 149-201 |
| Decoder | model.py | 204-454 |
| Tacotron2 | model.py | 457-529 |
| Tacotron2Loss | loss_function.py | 4-19 |
| WaveGlow | waveglow/glow.py | 178-293 |
