# Tacotron2 アーキテクチャ詳細

このドキュメントでは、Tacotron2の各コンポーネントの実装詳細とテンソル形状の変換を解説します。

## 全体構造

```
Tacotron2 (model.py:457-529)
├── Embedding層 (464-468行)
├── Encoder (149-201行)
│   ├── 3層 Conv1d + BatchNorm + ReLU + Dropout
│   └── Bidirectional LSTM
├── Decoder (204-454行)
│   ├── Prenet (89-100行)
│   ├── Attention RNN (LSTMCell)
│   ├── Location Sensitive Attention (29-86行)
│   ├── Decoder RNN (LSTMCell)
│   ├── Linear Projection (mel出力)
│   └── Gate Layer (終了予測)
└── Postnet (103-146行)
    └── 5層 Conv1d + BatchNorm
```

---

## 1. Embedding層

**ファイル**: `model.py:464-468`

```python
self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
# n_symbols = 148, symbols_embedding_dim = 512
```

### テンソル形状

```
入力:  (B, T_text)        # テキストのシンボルID列
       ↓ Embedding
出力:  (B, T_text, 512)
       ↓ transpose(1, 2)
最終:  (B, 512, T_text)   # Encoderへの入力形式
```

### 初期化

Xavier均一分布で初期化:
```python
std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
val = sqrt(3.0) * std
embedding.weight.data.uniform_(-val, val)
```

---

## 2. Encoder

**ファイル**: `model.py:149-201`

### 構造

```
入力: (B, 512, T_text)
      ↓
[Conv1d(512, 512, k=5) + BatchNorm + ReLU + Dropout(0.5)] × 3
      ↓
(B, 512, T_text)
      ↓ transpose(1, 2)
(B, T_text, 512)
      ↓
Bidirectional LSTM (256 × 2 = 512)
      ↓
出力: (B, T_text, 512)  ← "memory" または "encoder_outputs"
```

### Conv層の詳細

```python
ConvNorm(encoder_embedding_dim,      # 512
         encoder_embedding_dim,      # 512
         kernel_size=5,
         padding=2,                  # same padding
         dilation=1)
```

- 3層の1D畳み込み
- 各層: Conv → BatchNorm → ReLU → Dropout(0.5)
- padding=2 なので出力サイズは入力と同じ

### BiLSTM の詳細

```python
self.lstm = nn.LSTM(
    encoder_embedding_dim,           # 512
    encoder_embedding_dim // 2,      # 256 (各方向)
    num_layers=1,
    batch_first=True,
    bidirectional=True               # 両方向
)
# 出力: 256 × 2 = 512次元
```

### 学習時 vs 推論時

| 処理 | 学習時 (forward) | 推論時 (inference) |
|-----|-----------------|-------------------|
| パディング処理 | `pack_padded_sequence` 使用 | 使用しない |
| 入力長 | 可変長をバッチ処理 | 単一サンプル |

学習時は `pack_padded_sequence` でパディングを効率的に処理:
```python
x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
outputs, _ = self.lstm(x)
outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
```

---

## 3. Decoder

**ファイル**: `model.py:204-454`

Decoderは複数のサブコンポーネントで構成されます。

### 3.1 Prenet

**ファイル**: `model.py:89-100`

```python
Prenet(n_mel_channels * n_frames_per_step,  # 80
       [prenet_dim, prenet_dim])            # [256, 256]
```

```
入力:  (*, 80)   # 前フレームのメルスペクトログラム
       ↓ Linear(80, 256) + ReLU + Dropout(0.5)
       ↓ Linear(256, 256) + ReLU + Dropout(0.5)
出力:  (*, 256)
```

**重要**: Dropout は `training=True` に固定されており、推論時でも適用されます:
```python
x = F.dropout(F.relu(linear(x)), p=0.5, training=True)  # 常にTrue
```

これは「テストタイムドロップアウト」として、推論時の多様性と安定性に寄与します。

### 3.2 Attention RNN

```python
self.attention_rnn = nn.LSTMCell(
    prenet_dim + encoder_embedding_dim,  # 256 + 512 = 768
    attention_rnn_dim                    # 1024
)
```

```
入力:  concat(prenet_out, attention_context) = (B, 768)
       ↓ LSTMCell
出力:  hidden_state (B, 1024), cell_state (B, 1024)
       ↓ Dropout(0.1)
最終:  attention_hidden (B, 1024)
```

### 3.3 Decoder RNN

```python
self.decoder_rnn = nn.LSTMCell(
    attention_rnn_dim + encoder_embedding_dim,  # 1024 + 512 = 1536
    decoder_rnn_dim                            # 1024
)
```

```
入力:  concat(attention_hidden, attention_context) = (B, 1536)
       ↓ LSTMCell
出力:  hidden_state (B, 1024), cell_state (B, 1024)
       ↓ Dropout(0.1)
最終:  decoder_hidden (B, 1024)
```

### 3.4 出力層

```python
# メルスペクトログラム出力
self.linear_projection = LinearNorm(
    decoder_rnn_dim + encoder_embedding_dim,  # 1536
    n_mel_channels * n_frames_per_step        # 80
)

# 終了予測 (Gate)
self.gate_layer = LinearNorm(
    decoder_rnn_dim + encoder_embedding_dim,  # 1536
    1,
    w_init_gain='sigmoid'
)
```

```
入力:  concat(decoder_hidden, attention_context) = (B, 1536)
       ↓
mel_output:      Linear(1536, 80) → (B, 80)
gate_prediction: Linear(1536, 1)  → (B, 1)
```

### 3.5 Decode処理の1ステップ

**ファイル**: `model.py:340-379`

```python
def decode(self, decoder_input):
    # 1. Attention RNN
    cell_input = concat(decoder_input, attention_context)  # (B, 768)
    attention_hidden = attention_rnn(cell_input)           # (B, 1024)

    # 2. Attention計算 (詳細は ATTENTION.md)
    attention_context, attention_weights = attention_layer(...)

    # 3. 累積Attention重みを更新
    attention_weights_cum += attention_weights

    # 4. Decoder RNN
    decoder_input = concat(attention_hidden, attention_context)  # (B, 1536)
    decoder_hidden = decoder_rnn(decoder_input)                  # (B, 1024)

    # 5. 出力計算
    output_input = concat(decoder_hidden, attention_context)     # (B, 1536)
    mel_output = linear_projection(output_input)                 # (B, 80)
    gate_prediction = gate_layer(output_input)                   # (B, 1)

    return mel_output, gate_prediction, attention_weights
```

---

## 4. 学習時 vs 推論時の違い

### 学習時 (forward)

**ファイル**: `model.py:381-416`

```python
def forward(self, memory, decoder_inputs, memory_lengths):
    # 1. Go frame (全ゼロ) を準備
    decoder_input = get_go_frame(memory)  # (B, 80)

    # 2. Teacher forcing: 正解メルを入力として使用
    decoder_inputs = parse_decoder_inputs(decoder_inputs)
    decoder_inputs = concat(go_frame, decoder_inputs)
    decoder_inputs = prenet(decoder_inputs)  # 全フレーム一括処理

    # 3. 固定長でループ (正解の長さ分)
    while len(mel_outputs) < T_out:
        mel_output, gate_output, alignment = decode(decoder_inputs[t])
        ...
```

特徴:
- **Teacher Forcing**: 正解メルスペクトログラムを入力として使用
- **固定長**: 正解の長さ分だけデコード
- **バッチ処理**: Prenetは全フレームを一括処理

### 推論時 (inference)

**ファイル**: `model.py:418-454`

```python
def inference(self, memory):
    # 1. Go frame からスタート
    decoder_input = get_go_frame(memory)  # (B, 80)

    # 2. 自己回帰ループ
    while True:
        decoder_input = prenet(decoder_input)  # 毎ステップ処理
        mel_output, gate_output, alignment = decode(decoder_input)

        # 3. 終了判定
        if sigmoid(gate_output) > 0.5:
            break
        if len(mel_outputs) >= max_decoder_steps:  # 1000
            break

        # 4. 前フレーム出力を次の入力に
        decoder_input = mel_output
```

特徴:
- **自己回帰**: 前フレームの出力を次の入力として使用
- **動的長**: Gate予測で終了を判定
- **逐次処理**: Prenetは1フレームずつ処理

---

## 5. Postnet

**ファイル**: `model.py:103-146`

Decoderの出力を洗練させる残差ネットワーク。

### 構造

```
入力:  (B, 80, T_mel)
       ↓
Conv1d(80, 512, k=5) + BatchNorm + Tanh + Dropout(0.5)
       ↓
[Conv1d(512, 512, k=5) + BatchNorm + Tanh + Dropout(0.5)] × 3
       ↓
Conv1d(512, 80, k=5) + BatchNorm + Dropout(0.5)  # 最終層はTanhなし
       ↓
出力:  (B, 80, T_mel)  ← 残差として加算
```

### 残差接続

```python
mel_outputs_postnet = self.postnet(mel_outputs)
mel_outputs_postnet = mel_outputs + mel_outputs_postnet  # 残差加算
```

Postnetは「補正値」を学習し、Decoderの出力に加算することで品質を向上させます。

---

## 6. テンソル形状の完全なフロー

### 学習時

```
テキスト入力: (B, T_text)
    ↓ Embedding + transpose
(B, 512, T_text)
    ↓ Encoder
encoder_outputs: (B, T_text, 512)  ← "memory"

メル入力 (正解): (B, 80, T_mel)
    ↓ parse_decoder_inputs
(T_mel, B, 80)
    ↓ Prenet
(T_mel, B, 256)

[各タイムステップ t = 0, 1, ..., T_mel-1]
    prenet_out: (B, 256)
    attention_context: (B, 512)
        ↓ Attention RNN
    attention_hidden: (B, 1024)
        ↓ Attention Layer
    attention_context: (B, 512)
    attention_weights: (B, T_text)
        ↓ Decoder RNN
    decoder_hidden: (B, 1024)
        ↓ Output Layers
    mel_output: (B, 80)
    gate_output: (B, 1)

mel_outputs: (B, 80, T_mel)
    ↓ Postnet
mel_outputs_postnet: (B, 80, T_mel)

最終出力:
- mel_outputs: (B, 80, T_mel)
- mel_outputs_postnet: (B, 80, T_mel)
- gate_outputs: (B, T_mel)
- alignments: (B, T_mel, T_text)
```

---

## 7. 主要なハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `n_symbols` | 148 | シンボル数 |
| `symbols_embedding_dim` | 512 | 埋め込み次元 |
| `encoder_n_convolutions` | 3 | Encoder Conv層数 |
| `encoder_kernel_size` | 5 | Encoder Convカーネルサイズ |
| `encoder_embedding_dim` | 512 | Encoder出力次元 |
| `n_mel_channels` | 80 | メルスペクトログラムチャネル数 |
| `prenet_dim` | 256 | Prenet隠れ層次元 |
| `attention_rnn_dim` | 1024 | Attention RNN次元 |
| `decoder_rnn_dim` | 1024 | Decoder RNN次元 |
| `postnet_n_convolutions` | 5 | Postnet Conv層数 |
| `postnet_embedding_dim` | 512 | Postnet隠れチャネル数 |
| `max_decoder_steps` | 1000 | 最大デコード長 |
| `gate_threshold` | 0.5 | 終了判定閾値 |
| `p_attention_dropout` | 0.1 | Attention Dropout率 |
| `p_decoder_dropout` | 0.1 | Decoder Dropout率 |
