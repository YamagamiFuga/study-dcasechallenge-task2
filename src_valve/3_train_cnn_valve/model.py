# ファイルパスを安全に扱うために使う
import os

from pathlib import Path

# 表形式データ（CSVなど）を扱うために使う
import pandas as pd

# ループの進捗バーを表示するために使う
from tqdm import tqdm

# 音声ファイルの読み込みや特徴量抽出に使う
import librosa
import librosa.display

# グラフ表示に使う
import matplotlib.pyplot as plt

# 数値計算に使う
import numpy as np

# 学習データとテストデータに分けるために使う
from sklearn.model_selection import train_test_split

# PyTorch本体
import torch

# Datasetは独自データセット作成用、DataLoaderはバッチ取得用
from torch.utils.data import Dataset, DataLoader

# ニューラルネットワークの部品
import torch.nn as nn

# 最適化アルゴリズム（重み更新）に使う
import torch.optim as optim

import torch.nn as nn

from torchvision.models import resnet18


from collections import Counter

from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans





base_dir = os.path.join("dev_valve")

# 学習用の音声ファイルが入っているディレクトリのパス
train_dir = os.path.join(base_dir, "valve/train")

# テスト用の音声ファイルが入っているディレクトリのパス
test_dir = os.path.join(base_dir, "valve/test")



class MelSpectrogramProcessor:
    """音声ファイルからメルスペクトログラムを抽出するクラス"""
    
    def __init__(self, sr=16000, n_mels=64, n_fft=2048, hop_length=512,
                power=2.0, fmin=200, fmax=8000):
        """
        メルスペクトログラム抽出のパラメータを初期化

        Args:
            sr (int): サンプリングレート
                音声を1秒当たり何個の点に分割するか
                意味：音声の時間分解能を決定する
                例：sr=22050なら1秒間に22050個の点にする
                
            n_mels (int): メル周波数帯の数
                STFTで得られたスペクトログラムを、メル尺度に基づいてn_mels個の周波数帯に分割する
                意味：人間の聴覚特性に合わせた周波数分解能を提供する
                例：n_mels=128なら128個のメル周波数帯に分割する
                
            n_fft (int): FFTサイズ
                音声を短いフレームに分割してフーリエ変換を行う際のフレームサイズ
                意味：周波数分解能と時間分解能のトレードオフを決定する
                例：n_fft=2048なら2048サンプルごとにFFTを計算する
                
            hop_length (int): フレームのホップ長
                音声を短いフレームに分割してフーリエ変換を行う際の移動距離
                意味：フレームの重なり具合を決定し、ぶつ切りのないスペクトログラムを生成する
                例：hop_length=512なら512サンプルごとに次のフレームを計算する
                今回はn_fft=2048でhop_length=512なので、75%の重なりがあるフレーム分割になる
                512 / 16000 = 0.032秒ごとにスペクトログラムが計算されることになる
                
            power (float): パワースペクトラムの指数
                メルスペクトログラムを計算する際のスペクトルのパワーを指定する
                意味：スペクトルのエネルギー表現を決定する
                例：power=2.0なら振幅スペクトルの二乗（エネルギー）を使用する
            
            fmin (int): 最小周波数
                メルスペクトログラムに含める最小周波数を指定する
                意味：分析対象の音声の周波数範囲を制限する
                例：fmin=0なら0Hzから分析を開始する
            
            fmax (int): 最大周波数
                メルスペクトログラムに含める最大周波数を指定する
                意味：分析対象の音声の周波数範囲を制限する
                fmax=None（デフォルト）ならサンプリングレートの半分（ナイキスト周波数）まで分析する
                    sr=16000の場合、fmaxは8000（ナイキスト周波数）
                「機械の異常は何kHzあたりに出る？」みたいな当たりがあるなら、fmax をその少し上に置くのが定石
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.fmin = fmin
        self.fmax = fmax

    def extract_mel_spectrogram(self, audio_path, max_len=128):
        """
        音声ファイルからメルスペクトログラムを抽出

        Args:
            audio_path (str): 音声ファイルのパス
            max_len (int): 抽出するメルスペクトログラムの最大時間ステップ数

        Returns:
            mel_spec (np.ndarray): メルスペクトログラム (n_mels, time_steps)
        """
        # 音声ファイルを読み込み
        y, sr = librosa.load(audio_path, sr=self.sr)

        # メルスペクトログラムを計算
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # デシベルスケールに変換（より人間の聴覚に合わせた表現）
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # メルスペクトログラムの時間ステップ数をmax_lenに合わせる
        if mel_spec_db.shape[1] < max_len:
            # 短い → 0で埋める
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=max_len, axis=1)
        else:
            # 長い → 切る
            mel_spec_db = mel_spec_db[:, :max_len]

        
        return mel_spec_db   

# CNNの学習に使うデータセットクラスを定義する
class TrainDataset(Dataset):

    def __init__(self, path_label, processor):
        self.path_label = path_label
        self.processor = processor

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]

        mel = self.processor.extract_mel_spectrogram(str(path))
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)
        
        return mel, label
    

class ResNetFrontend(nn.Module):
    
    def __init__(self, n_classes, emb_dim, pretrained):
        super(ResNetFrontend, self).__init__()
        
        # torchvisionのresnet18をベースにする
        self.resnet = resnet18(pretrained=pretrained)
        
        # 入力チャンネル数を1に変更する（元は3）
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 最後の全結合層を削除する（特徴量抽出器として使うため）
        self.resnet.fc = nn.Identity()
        
        self.proj = nn.Linear(512, emb_dim)
        
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        
        # ResNetの出力は(B,512)の特徴ベクトルになる
        # 入力画像をみてResNetが特徴抽出を行い、512次元のベクトルを出力する
        h = self.resnet(x)     # (B,512)
        
        # 512次元の特徴ベクトルをemb_dim(256)次元に変換
        z = self.proj(h)       # (B,emb_dim)
        
                # クラス分類のスコアを出力
        logits = self.classifier(z)  # (B, n_classes)
        
        
        return z, logits



train_files = sorted(Path(train_dir).glob("**/*.wav"))

test_files = sorted(Path(test_dir).glob("**/*.wav"))


processor = MelSpectrogramProcessor()


features = []

for path in tqdm(train_files):
    # 音声ファイルからメルスペクトログラムを抽出
    mel = processor.extract_mel_spectrogram(str(path))
    
    # 2次元のメルスペクトログラムを1次元にフラット化する
    feature = mel.flatten()  
    
    features.append(feature)

# featuresは、すべての音声ファイルのメルスペクトログラムをフラット化したものが入ったリストになる
features = np.stack(features)
# print(features.shape)   # 例： (1000, 8192) なら、100ファイル分の特徴量があり、各特徴量は1つにつき8192次元のベクトルになっていることを意味

# PCAを使って特徴量の次元を削減する
# 8192次元 → 50次元に圧縮し、クラスタリングや分類がしやすいようにする
pca = PCA(n_components=50)
X_pca = pca.fit_transform(features)

# print(X_pca.shape)

# PCAで削減した特徴量を使ってK-meansクラスタリングを行い、似ている音をグループ化する
# 50次元の特徴量を5つのクラスタに分ける
kmeans = KMeans(n_clusters=5)

# 似ている音をグループ化する
labels = kmeans.fit_predict(X_pca)
"""
    labels = [0, 2, 1, 0, 3, 2, ...]
    0: クラスタ0に属する音
    1: クラスタ1に属する音
    2: クラスタ2に属する音
    3: クラスタ3に属する音
"""

print(Counter(labels))

path_label = []

for path, label in zip(train_files, labels):
    path_label.append((path, label))
    


train_dataset = TrainDataset(
    path_label=path_label,
    processor=processor
    )


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


model = ResNetFrontend(n_classes=5, emb_dim=256, pretrained=True)
model.train()

# 今回は5クラス分類で、多クラス分類なのでCrossEntropyLossを使う
# CrossEntropyLossは、モデルの出力と正解ラベルの間の損失を計算する関数で、モデルの予測が正解からどれだけ乖離しているかを数値で表す
loss_function = nn.CrossEntropyLoss()


# Adamで重みを更新する
# Adamは、モデルの重みを更新するための最適化アルゴリズムの一つで、学習率の自動調整やモーメンタムなどの機能を持っている
# lrは学習率で、モデルの重みをどれだけ更新するかを決めるハイパーパラメータ　今回は0.0002に設定している
optimizer = optim.Adam(model.parameters(), lr=1e-4)



# 何回学習するか
num_epochs = 6

for epoch in tqdm(range(num_epochs)):
    
    train_loss_sum = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()

        z, logits = model(x)

        loss = loss_function(logits, y)


        # 誤差逆伝播
        # loss.backward() を呼ぶと、モデルの重みを更新するための勾配が計算される　この勾配は、optimizer.step() を呼ぶことで、実際にモデルの重みが更新されるために使われる
        # 勾配とは、モデルの重みを更新するための情報で、誤差逆伝播によって計算される
        # 重みを更新することで、モデルの予測が正解に近づくようになる
        loss.backward()
        
        # パラメータ更新
        # optimizer.step() を呼ぶと、誤差逆伝播で計算された勾配を使って、モデルの重みが更新される　これにより、モデルの予測が正解に近づくようになる
        optimizer.step()

        # 損失を足し合わせる
        # loss.item() は、lossの中身の数値をPythonのfloat型で取り出すためのメソッド　lossはTensor型で、loss.item() を呼ぶと、その中身の数値がfloat型で返される
        train_loss_sum += loss.item()

    # 学習損失の平均を計算
    # train_loss_sum は、そのepochのすべてのバッチの損失を足し合わせたもの　len(train_loader) は、そのepochのバッチの数　これを割ることで、1バッチあたりの平均損失が求められる
    train_loss_avg = train_loss_sum / len(train_loader)

    print(f"Epoch {epoch}: loss={train_loss_avg}")
    
# 学習が終わったら、モデルの重みを保存する
torch.save(model.state_dict(), "cnn_frontend_valve_pseudo_clusters_5.pth")
    
