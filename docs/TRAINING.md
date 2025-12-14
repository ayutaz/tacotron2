# 学習パイプライン詳細

このドキュメントでは、Tacotron2の学習パイプライン全体を解説します。

## 全体の流れ

```
ファイルシステム
    ↓ filelists/*.txt
TextMelLoader (データ読み込み)
    ↓ get_text(), get_mel()
TextMelCollate (バッチ作成)
    ↓ パディング・ソート
DataLoader
    ↓ バッチ
model.parse_batch()
    ↓ GPU転送
model.forward()
    ↓ 予測
Tacotron2Loss
    ↓ Loss計算
backward() + optimizer.step()
    ↓ 重み更新
```

---

## 1. データファイルの形式

### filelists/*.txt

```
# 形式: audiopath|text
LJSpeech-1.1/wavs/LJ001-0001.wav|Printing, in the only sense...
LJSpeech-1.1/wavs/LJ001-0002.wav|For more than a century...
```

**ファイル**: `utils.py:18-21`

```python
def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text
```

---

## 2. テキスト処理

### 2.1 Text Cleaners

**ファイル**: `text/cleaners.py`

```python
def english_cleaners(text):
    # 1. ASCII変換: 'é' → 'e'
    text = convert_to_ascii(text)

    # 2. 小文字化: 'HELLO' → 'hello'
    text = lowercase(text)

    # 3. 数字展開: '123' → 'one hundred twenty three'
    text = expand_numbers(text)

    # 4. 略語展開: 'Mr.' → 'mister'
    text = expand_abbreviations(text)

    # 5. 空白正規化: '  ' → ' '
    text = collapse_whitespace(text)

    return text
```

### 数字展開の詳細

**ファイル**: `text/numbers.py`

```python
_inflect = inflect.engine()

def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        # 年として読む: 2019 → twenty nineteen
        ...
    else:
        # 通常の数字: 123 → one hundred twenty three
        return _inflect.number_to_words(num)
```

### 2.2 シンボル変換

**ファイル**: `text/__init__.py`

```python
def text_to_sequence(text, cleaner_names):
    sequence = []

    # クリーニング
    clean_text = _clean_text(text, cleaner_names)

    # シンボルID列に変換
    sequence = _symbols_to_sequence(clean_text)
    # 例: 'hello' → [33, 30, 37, 37, 40]

    return sequence
```

### 2.3 シンボル定義

**ファイル**: `text/symbols.py`

```python
_pad = '_'              # パディング (ID: 0)
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_arpabet = [...]        # CMU辞書のARPAbet記号

symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
# 合計: 148シンボル
```

---

## 3. 音声処理

### 3.1 WAVファイル読み込み

**ファイル**: `utils.py:13-15`

```python
def load_wav_to_torch(full_path):
    sampling_rate, data = wavfile.read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate
```

### 3.2 STFT (Short-Time Fourier Transform)

**ファイル**: `stft.py:42-105`

```python
class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024):
        # フーリエ基底の準備
        fourier_basis = np.fft.fft(np.eye(filter_length))
        # 実部と虚部に分離
        self.forward_basis = torch.FloatTensor(
            np.concatenate([np.real(cutoff), np.imag(cutoff)], axis=0)
        )

    def transform(self, input_data):
        # Reflect padding
        input_data = F.pad(input_data, (filter_length//2, filter_length//2),
                          mode='reflect')

        # Conv1dでSTFT計算
        forward_transform = F.conv1d(input_data, self.forward_basis,
                                     stride=self.hop_length)

        # 振幅と位相を計算
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase
```

### 3.3 メルスペクトログラム変換

**ファイル**: `layers.py:42-80`

```python
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length,
                 n_mel_channels, sampling_rate, mel_fmin, mel_fmax):
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        # メルフィルタバンク
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,  # 80
            fmin=mel_fmin,          # 0.0
            fmax=mel_fmax           # 8000.0
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    def mel_spectrogram(self, y):
        # y: (B, T) in [-1, 1]

        # STFT
        magnitudes, phases = self.stft_fn.transform(y)
        # magnitudes: (B, 513, T_frames)

        # メルフィルタバンク適用
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        # (80, 513) @ (B, 513, T) → (B, 80, T)

        # 動的レンジ圧縮
        mel_output = self.spectral_normalize(mel_output)
        # log(clamp(x, min=1e-5))

        return mel_output  # (B, 80, T_frames)
```

### STFTパラメータの意味

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `filter_length` | 1024 | FFTサイズ（周波数分解能） |
| `hop_length` | 256 | フレーム間隔（時間分解能） |
| `win_length` | 1024 | 窓関数長 |
| `n_mel_channels` | 80 | メル周波数帯の数 |
| `mel_fmin` | 0.0 | 最小周波数 |
| `mel_fmax` | 8000.0 | 最大周波数 |

---

## 4. データローダー

### 4.1 TextMelLoader

**ファイル**: `data_utils.py:11-64`

```python
class TextMelLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        # ファイルリスト読み込み
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)

        # STFT設定
        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate,
            hparams.mel_fmin, hparams.mel_fmax
        )

        # シャッフル
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        audio_norm = audio / self.max_wav_value  # [-1, 1]に正規化
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        return melspec.squeeze(0)  # (80, T_frames)
```

### 4.2 TextMelCollate

**ファイル**: `data_utils.py:67-111`

バッチ作成時にパディングとソートを行います：

```python
class TextMelCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step  # 1

    def __call__(self, batch):
        # 1. テキスト長でソート（降順）
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            descending=True
        )

        # 2. テキストパディング
        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len).zero_()
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx][0]
            text_padded[i, :text.size(0)] = text

        # 3. メルスペクトログラムパディング
        max_target_len = max([x[1].size(1) for x in batch])
        # n_frames_per_stepの倍数に調整
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len).zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len).zero_()
        output_lengths = torch.LongTensor(len(batch))

        for i, idx in enumerate(ids_sorted_decreasing):
            mel = batch[idx][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1  # 終了位置以降は1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
```

### Gate Paddingの意味

```
実際のメル:  [frame0, frame1, frame2, frame3, PAD, PAD, PAD]
Gate target: [  0   ,   0   ,   0   ,   1  ,  1 ,  1 ,  1 ]
                                      ↑ 終了位置
```

最後のフレーム以降を`1`（終了）としてマーク。

---

## 5. 学習ループ

**ファイル**: `train.py`

### 5.1 初期化

```python
def prepare_dataloaders(hparams):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        trainset,
        batch_size=hparams.batch_size,  # 64
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_loader, valset, collate_fn
```

### 5.2 メインループ

```python
def train(output_directory, log_directory, checkpoint_path, warm_start, hparams):
    # モデル初期化
    model = Tacotron2(hparams).cuda()
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

    # データローダー
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # 学習ループ
    for epoch in range(hparams.epochs):
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            # 1. バッチ解析
            x, y = model.parse_batch(batch)

            # 2. フォワードパス
            y_pred = model(x)

            # 3. Loss計算
            loss = criterion(y_pred, y)

            # 4. バックプロパゲーション
            loss.backward()

            # 5. 勾配クリッピング
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh  # 1.0
            )

            # 6. 重み更新
            optimizer.step()

            # 7. ログ出力
            if iteration % hparams.log_interval == 0:
                logger.log_training(loss, grad_norm, iteration)

            # 8. チェックポイント保存
            if iteration % hparams.iters_per_checkpoint == 0:
                validate(model, criterion, valset, iteration, collate_fn)
                save_checkpoint(model, optimizer, iteration, checkpoint_path)
```

### 5.3 parse_batch()

**ファイル**: `model.py:473-485`

```python
def parse_batch(self, batch):
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch

    # GPU転送
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    max_len = torch.max(input_lengths.data).item()

    # x: モデル入力, y: ターゲット
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    y = (mel_padded, gate_padded)

    return x, y
```

---

## 6. Loss関数

**ファイル**: `loss_function.py`

```python
class Tacotron2Loss(nn.Module):
    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_out, mel_out_postnet, gate_out, _ = model_output

        # 勾配計算しない
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        # Gate形状変換: (B, T) → (B*T, 1)
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)

        # Mel Loss (MSE)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                   nn.MSELoss()(mel_out_postnet, mel_target)

        # Gate Loss (BCE)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        return mel_loss + gate_loss
```

### Loss構成

```
Total Loss = Mel Loss + Gate Loss

Mel Loss = MSE(decoder_out, target) + MSE(postnet_out, target)
         = 2つのMSEの合計

Gate Loss = BCEWithLogits(gate_out, gate_target)
          = 終了予測の二値分類Loss
```

### なぜ2つのMel Lossがあるのか？

1. **Decoder出力のMSE**: Decoderが直接メルを学習
2. **Postnet出力のMSE**: Postnetが補正を学習

両方にLossをかけることで、両方のモジュールが学習できます。

---

## 7. 検証とチェックポイント

### 7.1 検証 (validate)

```python
def validate(model, criterion, valset, iteration, collate_fn):
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, batch_size=hparams.batch_size, collate_fn=collate_fn)
        val_loss = 0.0
        for batch in val_loader:
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    model.train()
    logger.log_validation(val_loss, iteration)
```

### 7.2 チェックポイント保存

```python
def save_checkpoint(model, optimizer, iteration, filepath):
    torch.save({
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filepath)
```

---

## 8. 完全なデータフロー図

```
WAVファイル: LJ001-0001.wav (22050 Hz, 48000 samples)
テキスト: "Printing, in the only sense..."
    ↓
[TextMelLoader.__getitem__]
    ├─ get_text():
    │   └─ english_cleaners() → text_to_sequence()
    │      "Printing..." → [34, 52, 34, 25, ...] (IntTensor[78])
    │
    └─ get_mel():
        └─ load_wav → normalize → STFT → mel_spectrogram
           48000 samples → (FloatTensor[80, 188])
    ↓
[TextMelCollate.__call__] (batch_size=64)
    ├─ テキストパディング: [64, max_text_len]
    ├─ メルパディング: [64, 80, max_mel_len]
    └─ Gateパディング: [64, max_mel_len]
    ↓
[model.parse_batch]
    └─ GPU転送, 型変換
    ↓
[model.forward]
    ├─ Embedding: [64, T_text] → [64, T_text, 512]
    ├─ Encoder: [64, T_text, 512] → encoder_outputs
    ├─ Decoder: Teacher Forcing → mel_outputs, gate_outputs
    └─ Postnet: mel_outputs → mel_outputs_postnet
    ↓
[Tacotron2Loss.forward]
    ├─ mel_loss = MSE(mel_out, target) + MSE(postnet_out, target)
    └─ gate_loss = BCEWithLogits(gate_out, target)
    ↓
[backward + optimizer.step]
    └─ 重み更新
```

---

## 9. ハイパーパラメータ一覧

**ファイル**: `hparams.py`

### 音声処理

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `sampling_rate` | 22050 | サンプリングレート |
| `filter_length` | 1024 | FFTサイズ |
| `hop_length` | 256 | フレーム間隔 |
| `win_length` | 1024 | 窓関数長 |
| `n_mel_channels` | 80 | メルチャンネル数 |
| `mel_fmin` | 0.0 | 最小周波数 |
| `mel_fmax` | 8000.0 | 最大周波数 |

### 学習設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `batch_size` | 64 | バッチサイズ |
| `learning_rate` | 1e-3 | 学習率 |
| `weight_decay` | 1e-6 | 重み減衰 |
| `grad_clip_thresh` | 1.0 | 勾配クリッピング閾値 |
| `epochs` | 500 | エポック数 |
| `iters_per_checkpoint` | 1000 | チェックポイント間隔 |

### テキスト処理

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `text_cleaners` | ['english_cleaners'] | クリーナー |
| `n_symbols` | 148 | シンボル数 |
