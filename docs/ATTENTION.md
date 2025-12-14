# Location Sensitive Attention 詳細解説

このドキュメントでは、Tacotron2の核心であるLocation Sensitive Attentionの仕組みを詳しく解説します。

## なぜ位置情報が必要なのか？

### 音声合成における問題

通常のAttention（Content-based Attention）は、**内容の類似度だけ**でどこを見るか決めます：

```
energy = v^T * tanh(W_query * query + W_memory * memory)
```

しかし、音声合成では以下の問題が発生します：

1. **繰り返し問題**: 同じ単語が複数回出てくると、同じ場所を何度も見てしまう
2. **スキップ問題**: テキストの一部を飛ばしてしまう
3. **逆行問題**: 一度読んだ場所に戻ってしまう

### 解決策: Location Sensitive Attention

音声合成では「**テキストは左から右に順番に読む**」という制約があります。

Location Sensitive Attentionは、**前回どこを見たか**という位置情報を追加することで、単調な進行を学習します：

```
energy = v^T * tanh(W_query * query + W_location * location + W_memory * memory)
                                      ↑ 位置情報を追加
```

---

## 実装の詳細

### 1. LocationLayer

**ファイル**: `model.py:10-26`

```python
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        # attention_n_filters = 32
        # attention_kernel_size = 31
        # attention_dim = 128

        # 1D畳み込み: 位置情報を処理
        self.location_conv = ConvNorm(
            2,                        # 入力: 2チャンネル
            attention_n_filters,      # 出力: 32チャンネル
            kernel_size=31,           # 31フレームの範囲を見る
            padding=15                # same padding
        )

        # 線形層: Attention次元に投影
        self.location_dense = LinearNorm(
            attention_n_filters,      # 32
            attention_dim             # 128
        )
```

### 入力の2チャンネルとは？

```python
attention_weights_cat = torch.cat(
    (self.attention_weights.unsqueeze(1),      # 前回のAttention重み
     self.attention_weights_cum.unsqueeze(1)), # 累積Attention重み
    dim=1)
# 形状: (B, 2, T_text)
```

| チャンネル | 意味 | 役割 |
|-----------|------|------|
| チャンネル0 | 前回のAttention重み | 「直前にどこを見たか」 |
| チャンネル1 | 累積Attention重み | 「今までどこを見たか」 |

### kernel_size = 31 の意味

31フレームの範囲（前後15フレーム）で位置情報を畳み込み処理します。

これにより：
- 「前回のピーク位置のすぐ右側」を見つけやすくなる
- 局所的な位置パターンを学習できる

### LocationLayerの処理フロー

```
入力:  attention_weights_cat (B, 2, T_text)
       ↓ Conv1d(2, 32, k=31)
(B, 32, T_text)
       ↓ transpose(1, 2)
(B, T_text, 32)
       ↓ Linear(32, 128)
出力:  processed_attention (B, T_text, 128)
```

---

## 2. Attentionクラス

**ファイル**: `model.py:29-86`

### 初期化

```python
class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, ...):
        # Query処理: Decoder状態から
        self.query_layer = LinearNorm(
            attention_rnn_dim,    # 1024
            attention_dim         # 128
        )

        # Memory処理: Encoder出力から（事前計算可能）
        self.memory_layer = LinearNorm(
            embedding_dim,        # 512
            attention_dim         # 128
        )

        # エネルギー計算
        self.v = LinearNorm(
            attention_dim,        # 128
            1                     # スカラー出力
        )

        # 位置情報処理
        self.location_layer = LocationLayer(...)
```

### get_alignment_energies()

**ファイル**: `model.py:43-63`

```python
def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
    """
    query:                 (B, 1024)  Attention RNNの出力
    processed_memory:      (B, T_text, 128)  事前計算済み
    attention_weights_cat: (B, 2, T_text)  位置情報
    """

    # 1. Query処理: (B, 1024) → (B, 1, 128)
    processed_query = self.query_layer(query.unsqueeze(1))

    # 2. Location処理: (B, 2, T_text) → (B, T_text, 128)
    processed_attention_weights = self.location_layer(attention_weights_cat)

    # 3. エネルギー計算（Additive Attention）
    energies = self.v(torch.tanh(
        processed_query +              # (B, 1, 128)     → broadcast
        processed_attention_weights +  # (B, T_text, 128)
        processed_memory               # (B, T_text, 128)
    ))
    # 形状: (B, T_text, 1) → squeeze → (B, T_text)

    return energies
```

### forward()

**ファイル**: `model.py:65-86`

```python
def forward(self, attention_hidden_state, memory, processed_memory,
            attention_weights_cat, mask):
    """
    attention_hidden_state: (B, 1024)  Attention RNNの隠れ状態
    memory:                 (B, T_text, 512)  Encoder出力
    processed_memory:       (B, T_text, 128)  事前計算済み
    attention_weights_cat:  (B, 2, T_text)  位置情報
    mask:                   (B, T_text)  パディングマスク
    """

    # 1. エネルギー計算
    alignment = self.get_alignment_energies(
        attention_hidden_state, processed_memory, attention_weights_cat
    )
    # 形状: (B, T_text)

    # 2. マスク適用（パディング位置を-infに）
    if mask is not None:
        alignment.data.masked_fill_(mask, -float("inf"))

    # 3. Softmaxで正規化 → Attention重み
    attention_weights = F.softmax(alignment, dim=1)
    # 形状: (B, T_text)

    # 4. Context vector計算（重み付き平均）
    attention_context = torch.bmm(
        attention_weights.unsqueeze(1),  # (B, 1, T_text)
        memory                           # (B, T_text, 512)
    )
    # 形状: (B, 1, 512) → squeeze → (B, 512)

    return attention_context, attention_weights
```

---

## 3. 累積Attention重みの更新

**ファイル**: `model.py:365`

Decoder内で毎ステップ更新されます：

```python
self.attention_weights_cum += self.attention_weights
```

### なぜ累積するのか？

| 情報 | 役割 |
|------|------|
| 前回の重み | 「次は隣を見るべき」を学習 |
| 累積重み | 「すでに読んだ場所をスキップ」を学習 |

累積重みにより、モデルは「すでにAttentionを当てた場所」を記憶し、同じ場所を繰り返し見ることを避けられます。

---

## 4. Attention処理の完全なフロー

```
[Decoderの各ステップ]

1. 前回の状態を取得
   attention_weights:     (B, T_text)  # 前回のAttention重み
   attention_weights_cum: (B, T_text)  # 累積Attention重み

2. 位置情報を結合
   attention_weights_cat = concat(
       attention_weights.unsqueeze(1),      # (B, 1, T_text)
       attention_weights_cum.unsqueeze(1)   # (B, 1, T_text)
   )  # → (B, 2, T_text)

3. LocationLayerで処理
   processed_attention = location_layer(attention_weights_cat)
   # (B, T_text, 128)

4. エネルギー計算
   energies = v(tanh(
       query_layer(attention_hidden) +  # (B, 1, 128)
       processed_attention +             # (B, T_text, 128)
       processed_memory                  # (B, T_text, 128)
   ))
   # (B, T_text)

5. Softmaxで正規化
   attention_weights = softmax(energies)  # (B, T_text)

6. Context vector計算
   attention_context = bmm(attention_weights, memory)  # (B, 512)

7. 累積重みを更新
   attention_weights_cum += attention_weights
```

---

## 5. 可視化: Alignmentの例

正常に学習されたモデルのAlignment（Attention重み）は、対角線に近い形になります：

```
     T_text (テキスト位置) →
   ┌────────────────────────┐
 T │ ■                      │
 _ │  ■■                    │
 m │    ■■                  │
 e │      ■■■               │
 l │         ■■             │
   │           ■■■          │
 ↓ │              ■■        │
   │                ■■■     │
   │                   ■■■  │
   └────────────────────────┘
```

- 左上から右下への対角線パターン
- 各メルフレームは、対応するテキスト位置付近にAttentionを当てる
- 単調に右へ進行（逆行しない）

---

## 6. VITSのMASとの関連

### Tacotron2: Location Sensitive Attention

- **学習で獲得**: 位置情報を入力として、単調な進行を学習
- **ソフト**: Softmaxで重み付け（複数位置に分散可能）
- **自己回帰**: 1フレームずつ順番に生成

### VITS: Monotonic Alignment Search (MAS)

- **アルゴリズムで強制**: 動的計画法で最適な単調アライメントを探索
- **ハード**: 各メルフレームは1つのテキスト位置にのみ対応
- **非自己回帰**: Duration Predictorで長さを予測し、並列生成

### 比較表

| 項目 | Tacotron2 | VITS |
|------|-----------|------|
| 単調性の保証 | 学習で近似 | アルゴリズムで保証 |
| アライメント | ソフト (確率分布) | ハード (1対1) |
| 生成速度 | 遅い (自己回帰) | 速い (並列) |
| Attention失敗 | あり得る | 構造的に防止 |

### 進化の流れ

```
Tacotron2 (Location Sensitive Attention)
    ↓ 「単調進行を学習」という考え方
    ↓
FastSpeech (Duration Predictor + Attention)
    ↓ 「長さを直接予測」という発想
    ↓
VITS (MAS + Duration Predictor)
    ↓ 「単調性をアルゴリズムで保証」
```

Tacotron2の「位置情報を考慮したAttention」は、後のEnd-to-End TTSモデルに大きな影響を与えました。

---

## 7. 実装のポイント

### processed_memoryの事前計算

`model.py:288`で、Decoderの初期化時にEncoder出力を事前処理します：

```python
self.processed_memory = self.attention_layer.memory_layer(memory)
# memory: (B, T_text, 512) → processed_memory: (B, T_text, 128)
```

これにより、毎ステップの計算量を削減できます。

### マスク処理

パディング位置にAttentionが当たらないよう、エネルギーを-infに設定：

```python
if mask is not None:
    alignment.data.masked_fill_(mask, self.score_mask_value)  # -inf
```

Softmax後、-infの位置は0になります。

### 数値安定性

`score_mask_value = -float("inf")` を使用することで、Softmax後に確実に0になります。
