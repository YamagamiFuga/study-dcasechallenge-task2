# ====================================
# 必要なライブラリのインポート
# ====================================
import librosa  # 音声処理ライブラリ
import librosa.display  # スペクトログラム表示用
import matplotlib.pyplot as plt  # グラフ描画用
import numpy as np  # 数値計算用

# ====================================
# 音声ファイルの読み込み
# ====================================
# 分析対象の音声ファイルパスを指定
file_path = "dev_fan/fan/train/section_00_source_train_normal_0000_n_B.wav"

# librosa.load()で音声ファイルを読み込む
# y: 音声時系列データ（サンプル値の配列）
# sr: サンプリングレート（1秒間のサンプル数）
# sr=Noneで元のサンプリングレートを保持
y, sr = librosa.load(file_path, sr=None)

# 読み込んだ音声の情報を表示
print("長さ:", len(y))  # 総サンプル数
print("サンプリングレート:", sr)  # Hz単位

# ====================================
# 複数のスペクトログラムを並べて表示するための準備
# ====================================
# 3行1列のサブプロットを作成し、高さを調整
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# ====================================
# 1. スペクトログラム（通常のスペクトログラム）
# ====================================
# 短時間フーリエ変換（STFT）を適用
# 音声を短い時間フレームに分割し、各フレームでフーリエ変換を実施
# これにより時間と周波数の両方の情報を得られる
D = librosa.stft(y)

# STFTの結果から振幅スペクトログラムをdB（デシベル）スケールに変換
# |D|^2 は パワースペクトル（電力）を表す
# librosa.power_to_db() で dB スケール（対数スケール）に変換
# ref=np.max で最大値を0dBとして正規化
S_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)

# スペクトログラムを画像として表示
# sr=sr でサンプリングレートを指定（時間軸のスケールを正しく設定）
# x_axis='time' で横軸を時間、y_axis='hz' で縦軸を周波数に設定
img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('スペクトログラム')
# カラーバーを追加して、dB値の範囲を表示
fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

# ====================================
# 2. メルスペクトログラム（周波数スケールを人間の聴覚に合わせたバージョン）
# ====================================
# メル尺度上のスペクトログラムを計算
# 人間の耳は周波数を線形（リニア）ではなく対数的に認識する
# メル尺度はこの聴覚特性を模拣した周波数スケール
# 低周波数では周波数の微小な変化を感じやすく、高周波数では粗い分解能
S = librosa.feature.melspectrogram(y=y, sr=sr)

# メルスペクトログラムをdBスケールに変換
# 元のメルスペクトログラムはパワー（エネルギー）で表現されている
# dB変換することで、人間が知覚しやすい対数スケールに
S_db_mel = librosa.power_to_db(S, ref=np.max)

# メルスペクトログラムを画像として表示
# y_axis='mel' で縦軸をメル周波数に設定（線形周波数ではない）
img2 = librosa.display.specshow(S_db_mel, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
axes[1].set_title('メルスペクトログラム')
# カラーバーを追加
fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

# ====================================
# 3. 対数メルエネルギー（MFCCの前処理や音声処理で一般的）
# ====================================
# メルスペクトログラムを計算（dB変換前）
S_mel = librosa.feature.melspectrogram(y=y, sr=sr)

# 自然対数を適用してスペクトラムを圧縮
# 1e-9 を加算して、ゼロ以下の値の対数を避ける（log(0)は未定義）
# これにより、小さなエネルギー値も視覚的に区別できるようになる
S_mel_log = np.log(S_mel + 1e-9)

# 対数メルエネルギーを画像として表示
# dB変換ではなく自然対数なので、colorbar時の単位が異なる
img3 = librosa.display.specshow(S_mel_log, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
axes[2].set_title('対数メルエネルギー')
# カラーバーを追加（dB形式ではなく対数値）
fig.colorbar(img3, ax=axes[2])

# ====================================
# 画像の保存と表示
# ====================================
# サブプロット間のスペースを自動調整して見やすくする
plt.tight_layout()

# 作成したスペクトログラムを画像ファイルとして保存
# dpi=100 で解像度を指定（100ドット/インチ）
# bbox_inches='tight' で余白を最小化
plt.savefig('spectrograms.png', dpi=100, bbox_inches='tight')
print("スペクトログラムを spectrograms.png に保存しました")

# 画面にプロットを表示（対話型環境での確認用）
plt.show()