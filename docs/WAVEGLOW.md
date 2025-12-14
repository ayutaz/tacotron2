# WaveGlow ボコーダ詳細

このドキュメントでは、WaveGlowの仕組みとTacotron2との連携について解説します。

## WaveGlowとは

WaveGlowは**Flow-based**の音声生成モデルで、メルスペクトログラムから波形を生成します。

### Tacotron2 + WaveGlow パイプライン

```
テキスト → [Tacotron2] → メルスペクトログラム → [WaveGlow] → 波形 → WAVファイル
                          (B, 80, T_mel)          (B, T_audio)
```

---

## 1. Flow-basedモデルの基本概念

### 従来のモデルとの違い

| モデル | 生成方法 | 学習 | 特徴 |
|--------|----------|------|------|
| VAE | サンプリング + デコーダ | 近似的 | 速いが品質に限界 |
| GAN | ノイズ → 生成器 | 敵対的 | 高品質だが学習不安定 |
| **Flow** | 可逆変換 | 厳密な尤度 | 安定した学習、並列生成 |

### Flow-basedモデルの原理

```
学習時:  データ x → [可逆変換 f] → 潜在変数 z ~ N(0, I)
推論時:  z ~ N(0, I) → [逆変換 f^-1] → 生成データ x
```

**ポイント**:
- 変換 `f` は可逆（逆関数が存在）
- 変換のヤコビアン（行列式）が計算可能
- 厳密な対数尤度を最大化できる

### 変数変換の公式

```
log p(x) = log p(z) + log |det(∂z/∂x)|
         = log p(z) + Σ log |det(∂f_i/∂h_i)|
```

- `p(z)`: 潜在変数の分布（標準正規分布）
- `det(∂z/∂x)`: ヤコビアン行列式（変換による体積変化）

---

## 2. WaveGlowの構造

**ファイル**: `waveglow/glow.py`

### 全体構成

```
WaveGlow
├── Upsample (ConvTranspose1d)  # メルのアップサンプリング
└── n_flows回のフロー (12回)
    ├── Invertible1x1Conv        # 可逆1×1畳み込み
    └── WN (Affine Coupling)     # アフィンカップリング層
```

### ハイパーパラメータ (config.json)

```json
{
  "n_mel_channels": 80,
  "n_flows": 12,
  "n_group": 8,
  "n_early_every": 4,
  "n_early_size": 2,
  "WN_config": {
    "n_layers": 8,
    "n_channels": 256,
    "kernel_size": 3
  }
}
```

---

## 3. Invertible 1x1 Conv

**ファイル**: `waveglow/glow.py:62-102`

### 目的

チャンネル間の依存関係を学習する可逆な変換。

### 実装

```python
class Invertible1x1Conv(torch.nn.Module):
    def __init__(self, c):
        # QR分解で直交行列を初期化
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]  # det = 1 に調整
        self.W = nn.Parameter(W)

    def forward(self, z, reverse=False):
        if not reverse:
            # 順方向: z' = W @ z
            W = self.W.unsqueeze(2)  # (c, c, 1) for conv1d
            z = F.conv1d(z, W)
            log_det_W = torch.logdet(self.W) * (batch * n_of_groups)
            return z, log_det_W
        else:
            # 逆方向: z = W^-1 @ z'
            W_inverse = self.W.inverse()
            z = F.conv1d(z, W_inverse.unsqueeze(2))
            return z
```

### テンソル形状

```
入力:  (B, c, T)
       ↓ conv1d with W (c, c, 1)
出力:  (B, c, T)

log_det_W = log|det(W)| × batch_size × n_groups
```

---

## 4. Affine Coupling Layer (WN)

**ファイル**: `waveglow/glow.py:105-175`

### アフィンカップリングの原理

入力を半分に分割し、片方を使ってもう片方を変換：

```
入力:  x = [x_a, x_b]  (半分ずつ)

変換:  [s, t] = NN(x_a, condition)  # NNでスケールとシフトを予測
       y_a = x_a                    # 片方はそのまま
       y_b = exp(s) * x_b + t       # もう片方をアフィン変換

出力:  y = [y_a, y_b]
```

### なぜ可逆なのか？

逆変換が簡単に計算できる：

```
x_a = y_a
x_b = (y_b - t) / exp(s) = (y_b - t) * exp(-s)
```

### WN (WaveNet-style Network)

メルスペクトログラムを条件として、スケール `s` とシフト `t` を予測：

```python
class WN(torch.nn.Module):
    def __init__(self, n_in_channels, n_mel_channels, ...):
        # 開始層
        self.start = Conv1d(n_in_channels, n_channels)

        # メル条件層
        self.cond_layer = Conv1d(n_mel_channels, 2 * n_channels * n_layers)

        # Dilated Conv層 (n_layers=8)
        self.in_layers = ModuleList()
        self.res_skip_layers = ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128
            self.in_layers.append(Conv1d(n_channels, 2*n_channels,
                                         kernel_size=3, dilation=dilation))

        # 出力層: [log_s, t] を出力
        self.end = Conv1d(n_channels, 2 * n_in_channels)
```

### Gated Activation

WaveNetスタイルのゲート付き活性化：

```python
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act  # 要素ごとの積
```

---

## 5. 学習時の処理 (forward)

**ファイル**: `waveglow/glow.py:207-249`

```python
def forward(self, forward_input):
    spect, audio = forward_input
    # spect: (B, 80, T_mel)  メルスペクトログラム
    # audio: (B, T_audio)    生音声

    # 1. メルのアップサンプリング
    spect = self.upsample(spect)
    # ConvTranspose1d(80, 80, kernel=1024, stride=256)
    # (B, 80, T_mel) → (B, 80, T_audio)

    # 2. グループ化 (n_group=8)
    # audio: (B, T_audio) → (B, 8, T_audio/8)
    # spect: (B, 80, T_audio) → (B, 640, T_audio/8)  # 80*8=640

    # 3. n_flows回のフロー処理 (12回)
    for k in range(self.n_flows):
        # 早期出力 (4フローごと)
        if k % 4 == 0 and k > 0:
            output_audio.append(audio[:, :2, :])  # 2チャンネル出力
            audio = audio[:, 2:, :]               # 残りを継続

        # Invertible 1x1 Conv
        audio, log_det_W = self.convinv[k](audio)
        log_det_W_list.append(log_det_W)

        # Affine Coupling
        n_half = audio.size(1) // 2
        audio_0 = audio[:, :n_half, :]  # 前半
        audio_1 = audio[:, n_half:, :]  # 後半

        output = self.WN[k]((audio_0, spect))  # WNで[log_s, t]を予測
        log_s = output[:, n_half:, :]
        b = output[:, :n_half, :]

        audio_1 = torch.exp(log_s) * audio_1 + b  # アフィン変換
        log_s_list.append(log_s)

        audio = torch.cat([audio_0, audio_1], 1)

    output_audio.append(audio)
    return torch.cat(output_audio, 1), log_s_list, log_det_W_list
```

### データフロー図

```
audio (B, T_audio)
    ↓ グループ化
(B, 8, T_audio/8)
    ↓ Flow 0-3
(B, 8, T_audio/8), 早期出力: 2ch
    ↓ Flow 4-7
(B, 6, T_audio/8), 早期出力: 2ch
    ↓ Flow 8-11
(B, 4, T_audio/8), 早期出力: 2ch
    ↓
z: (B, 8, T_audio/8)  最終的な潜在変数
```

---

## 6. 推論時の処理 (infer)

**ファイル**: `waveglow/glow.py:251-293`

```python
def infer(self, spect, sigma=1.0):
    # 1. メルのアップサンプリング
    spect = self.upsample(spect)
    # アーティファクト除去
    time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
    spect = spect[:, :, :-time_cutoff]

    # 2. グループ化
    # spect: (B, 640, T_audio/8)

    # 3. ガウスノイズからスタート
    audio = torch.randn(B, n_remaining_channels, T).cuda()
    audio = sigma * audio  # sigmaでスケーリング

    # 4. 逆フロー処理 (逆順: 11, 10, ..., 0)
    for k in reversed(range(self.n_flows)):
        n_half = audio.size(1) // 2
        audio_0 = audio[:, :n_half, :]
        audio_1 = audio[:, n_half:, :]

        # WNで[log_s, t]を予測
        output = self.WN[k]((audio_0, spect))
        s = output[:, n_half:, :]
        b = output[:, :n_half, :]

        # アフィン変換の逆
        audio_1 = (audio_1 - b) / torch.exp(s)
        audio = torch.cat([audio_0, audio_1], 1)

        # Invertible 1x1 Convの逆
        audio = self.convinv[k](audio, reverse=True)

        # 早期出力の復元
        if k % 4 == 0 and k > 0:
            z = sigma * torch.randn(B, 2, T).cuda()
            audio = torch.cat((z, audio), 1)

    # 5. グループ解除 → 波形
    audio = audio.permute(0, 2, 1).view(B, -1)
    return audio
```

### sigma パラメータ

`sigma` は生成の多様性を制御：
- `sigma = 1.0`: 標準的な生成
- `sigma < 1.0`: より保守的（平均に近い）
- `sigma > 1.0`: より多様（ノイズが多い）

---

## 7. Loss関数

**ファイル**: `waveglow/glow.py:43-59`

```python
class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output

        # 1. 潜在変数のノルム (負の対数尤度)
        loss = torch.sum(z * z) / (2 * self.sigma ** 2)

        # 2. アフィンカップリングのlog_s
        for i, log_s in enumerate(log_s_list):
            loss = loss - torch.sum(log_s)

        # 3. Invertible 1x1 Convのlog_det
        for i, log_det_W in enumerate(log_det_W_list):
            loss = loss - log_det_W

        # 正規化
        return loss / (z.size(0) * z.size(1) * z.size(2))
```

### Loss の解釈

```
Loss = ||z||² / (2σ²) - Σ log|s| - Σ log|det(W)|
       \_________/     \_______/   \___________/
         NLL項          スケール項    行列式項
```

- **NLL項**: 潜在変数が標準正規分布に従うよう促す
- **スケール項**: アフィン変換のヤコビアン（体積変化）
- **行列式項**: 1×1畳み込みのヤコビアン

---

## 8. Tacotron2との連携

### メルスペクトログラムの形式

両モデルで同じ設定を使用：

```python
# hparams.py
sampling_rate = 22050      # サンプリングレート
filter_length = 1024       # FFTサイズ
hop_length = 256           # ホップ長
n_mel_channels = 80        # メルチャンネル数
mel_fmin = 0.0             # 最小周波数
mel_fmax = 8000.0          # 最大周波数
```

### 推論パイプライン

```python
# 1. Tacotron2でメル生成
tacotron2 = Tacotron2(hparams).cuda().eval()
mel_outputs, mel_outputs_postnet, _, _ = tacotron2.inference(text_tensor)
# mel_outputs_postnet: (1, 80, T_mel)

# 2. WaveGlowで波形生成
waveglow = WaveGlow(config).cuda().eval()
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
# audio: (1, T_audio)

# 3. WAVファイル保存
audio = audio.squeeze().cpu().numpy()
wavfile.write('output.wav', 22050, audio)
```

### デノイザー（オプション）

WaveGlowはシステマティックなバイアスを生成することがあります。デノイザーで除去：

```python
# waveglow/denoiser.py
class Denoiser(torch.nn.Module):
    def __init__(self, waveglow):
        # ゼロ入力でバイアスを記録
        mel_input = torch.zeros(1, 80, 88).cuda()
        bias_audio = waveglow.infer(mel_input, sigma=0.0)
        bias_spec, _ = self.stft.transform(bias_audio)
        self.bias_spec = bias_spec[:, :, 0:1]

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio)
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        return self.stft.inverse(audio_spec_denoised, audio_angles)
```

---

## 9. VITSとの比較

| 項目 | WaveGlow | VITS (HiFi-GAN) |
|------|----------|-----------------|
| 種類 | Flow-based | GAN-based |
| 学習 | 尤度最大化 | 敵対的学習 |
| 速度 | 中程度 | 非常に高速 |
| 品質 | 高品質 | 非常に高品質 |
| メモリ | 大きい | 小さい |
| End-to-End | No (2段階) | Yes (1段階) |

VITSは、Tacotron2 + WaveGlowの2段階パイプラインを1つのモデルに統合し、さらにGANベースのボコーダで高速化しています。

---

## 10. ファイル構成

```
waveglow/
├── glow.py           # メインモデル (WaveGlow, WN, Invertible1x1Conv)
├── train.py          # 学習スクリプト
├── inference.py      # 推論スクリプト
├── mel2samp.py       # データローダー
├── denoiser.py       # デノイザー
├── config.json       # ハイパーパラメータ
└── tacotron2/        # サブモジュール (STFT等のユーティリティ)
    ├── layers.py
    ├── stft.py
    └── audio_processing.py
```
