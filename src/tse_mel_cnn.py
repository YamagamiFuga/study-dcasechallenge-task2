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

# torchvisionの学習済みモデルResNet34を使う
from torchvision.models import resnet34

import torch.nn as nn
from torchvision.models import resnet18


from sklearn.metrics import roc_auc_score

import soundfile as sf


base_dir = os.path.join("dev_fan")

# 学習用の音声ファイルが入っているディレクトリのパス
train_dir = os.path.join(base_dir, "fan/train")

train_wav_path = sorted(Path(train_dir).glob("**/*.wav"))

# 追加データが入っているディレクトリのパス
supplemental_dir = os.path.join(base_dir, "fan/supplemental")

# テスト用の音声ファイルのパスを取得
test_dir = os.path.join(base_dir, "fan/test")

test_wav_path = sorted(Path(test_dir).glob("**/*.wav"))



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
    

# CNNの学習に使うDatasetクラスを定義する
# モデルは(B, 1, 40, 128)の入力を想定しているので、Datasetクラスも同じ形のデータを返すようにする
class MelDataset(Dataset):
    def __init__(self, mel_list, labels=None):
        """
        mel_list : メルスペクトログラムのリスト [(40,128), ...]
        labels   : ラベル（分類用、なければNone）
        """
        self.mel_list = mel_list
        self.labels = labels

    def __len__(self):
        # データ数を返す
        return len(self.mel_list)

    # idx番目のデータを返す関数
    def __getitem__(self, idx):
        # 1つのデータを取り出す
        mel = self.mel_list[idx]  # (40,128)

        # numpy配列をPyTorchのTensorに変換
        # 入力データはメルスペクトログラムの2次元配列なので、dtypeはfloat32にする
        mel = torch.tensor(mel, dtype=torch.float32)

        # チャンネル次元追加 (1,40,128)
        # CNNは通常、(B, C, H, W)の形の入力を想定しているので、チャンネル次元を追加する
        # (40, 128) → (1, 40, 128)
        mel = mel.unsqueeze(0)

        if self.labels is not None:
            label = self.labels[idx]
            return mel, label
        else:
            return mel
    

# 入力されたスペクトログラムから、音の特徴ベクトルとクラス分類のスコアを出力する
# CNNフロントエンドを定義するクラス
class ResNetFrontend(nn.Module):
    """
        メルスペクトログラムを1チャンネル画像として受け取り、
        ResNet18で特徴抽出を行うフロントエンド。
        出力として、
        - z: 異常検知や類似度計算に使う埋め込みベクトル
        - logits: 学習時の分類スコア
        を返す。
        
        入力x
        → ResNetで特徴抽出
        → 512次元ベクトル h
        → 線形層で emb_dim 次元の z に変換
        → 分類層で logits を出す
    """
    
    
    def __init__(self, n_classes=5, emb_dim=256):
        super().__init__()

        # ResNet18読み込み
        self.resnet = resnet18(pretrained=True)

        #1ch対応に変更
        # ResNetの最初の畳み込み層は通常、3チャンネルのカラー画像を想定しているため、入力チャンネル数を1に変更する必要がある
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 最後の全結合層を削除
        # ResNetの最後の全結合層は通常、1000クラスの分類を想定している
        # 今回はそのままの分類は不要なため、特徴ベクトルを直接出力するように変更する
        # これにより、ResNetの出力は512次元の特徴ベクトルになる　→　(B, 512)
        self.resnet.fc = nn.Identity()

        # 特徴ベクトル変換
        # ResNetの出力は512次元なので、emb_dim(256)次元に変換する全結合層を追加
        self.proj = nn.Linear(512, emb_dim)

        # クラス分類用の全結合層
        # softmaxをかける前の値を出力する全結合層を追加
        self.classifier = nn.Linear(emb_dim, n_classes)

    # 前向き伝播
    def forward(self, x):  # (B,1,n_mels,T)

        # ResNetの出力は(B,512)の特徴ベクトルになる
        # 入力画像をみてResNetが特徴抽出を行い、512次元のベクトルを出力する
        h = self.resnet(x)     # (B,512)
        
        # 512次元の特徴ベクトルをemb_dim(256)次元に変換
        z = self.proj(h)       # (B,emb_dim)
        """
        zの値は例えば、[[0.5, -1.2, 0.3, ..., 0.8],
                        [-0.7, 0.4, -0.1, ..., -0.3],
                        ...,
                        [1.0, -0.5, 0.2, ..., 0.6]]
        みたいな感じで、各サンプルに対して256次元の特徴ベクトルが出力される
        """
        
        
        # クラス分類のスコアを出力
        logits = self.classifier(z)
        """
        今回はクラスが5
        logitsの値は[2.1, -0.3, 0.7, 1.8, -1.2]
        みたいな感じで、クラスごとのスコアが出力される
        """
        

        # zは特徴ベクトル、logitsはクラス分類のスコア（学習用）
        return z, logits
    
    
# コサイン距離を計算する関数
def cosine_distance(a, b):
    # np.linalg.norm(a)はベクトルaの大きさを計算する関数
    # ベクトルaをその大きさで割ることで、ベクトルaを単位ベクトルに変換することができる 大きさ　→　ベクトルの長さ
    # 1e-8はゼロ割りを防ぐための小さな値
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    
    # コサイン距離を返す
    # コサイン類似度は、2つのベクトルの内積を計算することで求められる np.dot(a, b)
    # コサイン距離は、コサイン類似度を1から引くことで求められる
    # 1に近いほど類似していることを意味し、0に近いほど類似していないことを意味する
    return 1.0 - float(np.dot(a, b))


# 特徴ベクトルを異常スコアに変換するためのクラス
class MinCosineBackend:
    # Z_trainは学習用の特徴ベクトルの集合 (N,D)
    # Nは学習用サンプルの数、Dは特徴ベクトルの次元数
    """
    例：Z_train = [[0.5, -1.2, 0.3, ..., 0.8], 音の特徴①
                [-0.7, 0.4, -0.1, ..., -0.3], 音の特徴②
                ...,                           ・・・・
                [1.0, -0.5, 0.2, ..., 0.6]]
    """
    # 学習用のベクトルを「長さ1」に正規化して保存する
    # 大きさ　→　音の強さ　向き　→　音の特徴　今回は異常検知なので、音の特徴の向きが重要
    def fit(self, Z_train):  
        self.Z = Z_train / (np.linalg.norm(Z_train, axis=1, keepdims=True) + 1e-8)

    # 入力された特徴ベクトルzと、学習用の特徴ベクトルZ_trainとのコサイン距離を計算し、
    # 最小の距離を異常スコアとして返す
    def score(self, z):       # (D,)
        # テスト用のベクトルも長さ1に正規化する
        z = z / (np.linalg.norm(z) + 1e-8)
        
        # 最小コサイン距離
        # self.Z @ zは、学習用の特徴ベクトルZ_trainとテスト用の特徴ベクトルzとの
        # コサイン類似度を計算する
        # 1.0 - self.Z @ zは、類似度を距離に変換する
        # 1に近いほど類似していることを意味し、0に近いほど類似していないことを意味する
        # np.minは、学習用の特徴ベクトルZ_trainの中で、
        # テスト用の特徴ベクトルzに最も類似しているものを見つけるために使う        
        return float(np.min(1.0 - self.Z @ z))
    


# メルスペクトログラム抽出器のインスタンスを作成
processor = MelSpectrogramProcessor(sr=16000, n_mels=40, n_fft=2048, hop_length=512)
    
# 学習用の音声ファイルのパスを取得
train_denoised_wav_path = sorted(Path("enhanced/train").glob("**/*.wav"))

# メルスペクトログラムを抽出してリストに保存
mel_list = []

# tqdmを使って進捗バーを表示しながら、各音声ファイルからメルスペクトログラムを抽出
for path in tqdm(train_denoised_wav_path):
    # 音声ファイルからメルスペクトログラムを抽出
    mel_spec = processor.extract_mel_spectrogram(str(path))
    mel_list.append(mel_spec)
    
# mel_listは、各音声ファイルから抽出されたメルスペクトログラムのリストになる
# 例：mel_list = [array([[ -80.0,  -75.3,  -70.1, ...,  -60.2], 音の特徴①
# 学習データセット
train_dataset = MelDataset(mel_list)


# DataLoader:
# データをまとめて(batch_size=32件ずつ)取り出すための仕組み
# ここで__getitem__と__len__が裏で自動的に呼ばれる
# 学習用はシャッフルする（shuffle=True）ことで、毎回違う順番でデータが出てくるようにする
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


model = ResNetFrontend()
model.eval()  # 推論モード

z_list = []

with torch.no_grad():  # 勾配計算をオフにする
    for mel in tqdm(train_loader, desc="Extract Z_train"):
        # melは(B, 1, 40, 128)の形
        
        z, _ = model(mel)  # zは(B, emb_dim)の形の特徴ベクトル
        
        z_list.append(z.cpu().numpy())  # CPUに移してNumPy配列に変換して保存
        
# train_loaderに含まれる正常データを、CNN（evalモード）で特徴ベクトルに変換し、それらをまとめたもの
Z_train = np.concatenate(z_list, axis=0)  # (N, emb_dim)の形の学習用特徴ベクトルの集合になる
# 例：Z_train = [[0.5, -1.2, 0.3, ..., 0.8], 音の特徴①
#                 [-0.7, 0.4, -0.1, ..., -0.3], 音の特徴②
#                 ...,                           ・・・・
#                 [1.0, -0.5, 0.2, ..., 0.6]] 音の特徴N]

# 学習用の正常データの特徴ベクトルを使って、MinCosineBackendを学習する
backend = MinCosineBackend()
backend.fit(Z_train)


# 学習用の正常データをCNNで特徴ベクトルに変換し、MinCosineBackendで異常スコアを計算する
scores = []

model.eval()

with torch.no_grad():
    for x in tqdm(train_loader, desc="Scoring train"):
        z, _ = model(x)
        z = z.cpu().numpy()

        for zi in z:
            score = backend.score(zi)
            scores.append(score)
            
            
# 異常スコアの分布をヒストグラムで表示する
plt.hist(scores, bins=50)
plt.title("Anomaly Score Distribution for Normal Data")
plt.xlabel("Anomaly Score (Cosine Distance)")
plt.ylabel("Number of Samples")
plt.show()


#####################
#　ここから先は、テストデータを使って異常スコアを計算するコード
####################

test_denoised_wav_path = sorted(Path("enhanced/test").glob("**/*.wav"))

test_scores = []

# テスト用の音声ファイルをCNNで特徴ベクトルに変換し、MinCosineBackendで異常スコアを計算する

model.eval()

with torch.no_grad():
    for path in tqdm(test_denoised_wav_path, desc="Testing"):
        # メルスペクトログラム
        mel = processor.extract_mel_spectrogram(str(path))

        # サイズ統一してる前提（40,128）
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 特徴抽出
        z, _ = model(mel)
        z = z.cpu().numpy()[0]

        # スコア計算
        test_score = backend.score(z)
        
        # スコアを保存
        test_scores.append(test_score)



normal_scores = []
anomaly_scores = []

for path, score in tqdm(zip(test_denoised_wav_path, test_scores), desc="Categorizing Scores"):
    name = path.name  # ファイル名だけ取得

    if "anomaly" in name:
        anomaly_scores.append(score)
    elif "normal" in name:
        normal_scores.append(score)
        

print("normal:", len(normal_scores))
print("anomaly:", len(anomaly_scores))

# 異常スコアの分布をヒストグラムで表示する
plt.hist(scores, bins=50, alpha=0.5, label="Train Normal")
plt.hist(normal_scores, bins=50, alpha=0.5, label="Test Normal")
plt.hist(anomaly_scores, bins=50, alpha=0.5, label="Anomaly")

plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Number of Samples")
plt.legend()

plt.show()

y_true = [0]*len(normal_scores) + [1]*len(anomaly_scores) # 正常データを0、異常データを1とするラベルのリストを作成
y_score = normal_scores + anomaly_scores    # 正常データと異常データのスコアを結合して、スコアのリストを作成


auc = roc_auc_score(y_true, y_score)
print("AUC:", auc)

pauc = roc_auc_score(y_true, y_score, max_fpr=0.1)
print("pAUC:", pauc)