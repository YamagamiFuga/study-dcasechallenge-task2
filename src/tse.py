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


def get_suffix_type(path):
    """
    ファイル名の末尾 A / B / C を返す
    """
    stem = Path(path).stem
    parts = stem.split("_")
    return parts[-1]


def collect_noise_files(supplemental_dir):
    """
    末尾が B または C の wav をノイズとして集める
    """
    wav_paths = sorted(Path(supplemental_dir).glob("*.wav"))
    noise_files = []

    for path in wav_paths:
        suffix_type = get_suffix_type(path)
        if suffix_type in ["B", "C"]:
            noise_files.append(path)

    return noise_files


def collect_clean_files(supplemental_dir):
    """
    末尾が A の wav をクリーン機械音として集める
    """
    wav_paths = sorted(Path(supplemental_dir).glob("*.wav"))
    clean_files = []

    for path in wav_paths:
        suffix_type = get_suffix_type(path)
        if suffix_type == "A":
            clean_files.append(path)

    return clean_files

# クリーン機械音数: 20
clean_files = collect_clean_files(supplemental_dir)

# ノイズの数: 80
noise_files = collect_noise_files(supplemental_dir)


# ノイズの特徴を平均化してノイズの形を作る関数
def build_average_noise(noise_files, sr=16000, n_fft=512, hop_length=128):
    """
    ノイズファイルから平均的なノイズスペクトルを作る関数
    noise_files: ノイズファイルのパスのリスト
    sr: サンプリングレート
    n_fft: FFTサイズ
    hop_length: フレームのホップ長
    
    戻り値:
    avg_noise_spec: 平均的なノイズスペクトル（numpy配列）
    """
    
    # 各ファイルからノイズスペクトルを計算して保存するリスト
    noise_specs = []
    """
    noise_specs = [
    ファイル1の平均スペクトル,
    ファイル2の平均スペクトル,
    ファイル3の平均スペクトル,
    ...
    ]
    """
    
    # ノイズファイルを1つずつ処理するループ
    for path in noise_files:
        
        # 音声ファイルを読み込む yは音声信号の波形データ、srはサンプリングレート
        y, sr = librosa.load(path, sr=sr)

        # 音声信号からスペクトルを計算する関数 フーリエ変換を行い、時間と周波数の両方の情報を持つスペクトルを得る
        noise_spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # 振幅スペクトルを取得
        # magはその時刻、その周波数にどれくらい音があるかを表す
        mag = np.abs(noise_spec)  
        """
        mag = [
        周波数1  [ t1, t2, t3, t4, ... ]
        周波数2  [ t1, t2, t3, t4, ... ]
        周波数3  [ t1, t2, t3, t4, ... ]
        ...
        ]
        """
        
        # 周波数ごとに平均を取る
        # avg_magは、各周波数に対して、平均的な強さを表すベクトルになる
        avg_mag = np.mean(mag, axis=1)  
        """
        avg_mag = [
        周波数1 の平均  
        周波数2 の平均  
        周波数3 の平均  
        ...
        ]
        """
        
        noise_specs.append(avg_mag)  # 平均スペクトルをリストに追加する
        
    
    # スペクトルを平均化する
    avg_noise_spec = np.mean(noise_specs, axis=0)
    
    print(avg_noise_spec.shape)
    
    return avg_noise_spec

# 学習用とテスト用の音声ファイルのスペクトルから、平均的なノイズスペクトルを引いて、ノイズを減らす関数
def spectral_subtraction_denoise(train_wav_path,test_wav_path, avg_noise_spec):
    """
    スペクトル減算を行う関数
    clean_spec: クリーン機械音のスペクトル（numpy配列）
    avg_noise_spec: 平均的なノイズのスペクトル（numpy配列）
    
    戻り値:
    enhanced_spec: スペクトル減算後のスペクトル（numpy配列）
    """
    
    # train_resultsは、学習用の音声ファイルのスペクトルからノイズを減らしたものを保存するリスト
    train_results = []
    
    # test_resultsは、テスト用の音声ファイルのスペクトルからノイズを減らしたものを保存するリスト
    test_results = []
    
    # avg_noise_specを(257, 1)の形に変換する処理
    # avg_noise_specは(257,)の形であり、後で出てくるmagは(257, T)の形なので、引き算するためにavg_noise_specを(257, 1)の形に変換しておく
    avg_noise_spec = avg_noise_spec[:, np.newaxis]

    # 学習用の音声フォルダ作成
    os.makedirs("enhanced/train", exist_ok=True)
    
    # 学習用の音声ファイルのノイズ除去を行うループ
    for path in train_wav_path:
        # 音声ファイルを読み込む yは音声信号の波形データ、srはサンプリングレート
        y, sr = librosa.load(path, sr=16000)

        # 音声信号からスペクトルを計算する関数 フーリエ変換を行い、時間と周波数の両方の情報を持つスペクトルを得る
        # Dの値は複素数で、振幅スペクトルと位相スペクトルの両方の情報を持っている　Dの形は(257, T)で、257は周波数ビンの数、Tは時間フレームの数
        D = librosa.stft(y, n_fft=512, hop_length=128)
                
        # 振幅スペクトルを取得
        # magはその時刻、その周波数にどれくらい音があるかを表す
        mag = np.abs(D)  
        
        # 位相スペクトルを取得
        # phaseはその時刻、その周波数の音の位相を表す
        phase = np.angle(D)
                
        # スペクトル減算を行う
        # 入力音声の振幅スペクトルmagから、平均的なノイズスペクトルavg_noise_specを引いて、ノイズを減らす
        enhanced_mag = mag - avg_noise_spec
        """
        例：mag = [
        [t1, t2, t3, t4, ...],
        [t1, t2, t3, t4, ...],
        [t1, t2, t3, t4, ...]
        ]
        avg_noise_spec = [
        周波数1 の平均 
        周波数2 の平均
        周波数3 の平均
        ]
        enhanced_mag = [
        [t1 - 周波数1の平均, t2 - 周波数1の平均, t3 - 周波数1の平均, ...],
        [t1 - 周波数2の平均, t2 - 周波数2の平均, t3 - 周波数2の平均, ...],
        [t1 - 周波数3の平均, t2 - 周波数3の平均, t3 - 周波数3の平均, ...]
        ]
        mag = 10, avg_noise_spec = 3 なら、enhanced_magは7になる
        したがって
        ノイズ3を引いて、残り7を機械音寄りの成分とみなす
        """
        
        # スペクトル減算の結果が負の値になることがあるので、0以上にするための処理
        floor = 0.02 * mag
        enhanced_mag = np.maximum(enhanced_mag, floor)

        # 複素数スペクトルを再構築する
        # もとの位相であるphaseを減算後のenhanced_magにかけ、ノイズ除去済みの正しい位相を持つスペクトルを再構築する
        enhanced_D = enhanced_mag * np.exp(1j * phase)
        
        y_enh = librosa.istft(enhanced_D, hop_length=128, length=len(y))

        filename = path.name
        
        save_path = os.path.join("enhanced/train", filename)

        sf.write(save_path, y_enh, sr)
    
    # テスト用の音声フォルダ作成
    os.makedirs("enhanced/test", exist_ok=True)
    
    # テスト用の音声ファイルのノイズ除去を行うループ
    for path in test_wav_path:
        
        # 音声ファイルを読み込む yは音声信号の波形データ、srはサンプリングレート
        y, sr = librosa.load(path, sr=16000)

        # 音声信号からスペクトルを計算する関数 フーリエ変換を行い、時間と周波数の両方の情報を持つスペクトルを得る
        D = librosa.stft(y, n_fft=512, hop_length=128)
                
        # 振幅スペクトルを取得
        # magはその時刻、その周波数にどれくらい音があるかを表す
        mag = np.abs(D)  
        
        # 位相スペクトルを取得
        # phaseはその時刻、その周波数の音の位相を表す
        phase = np.angle(D)
                
        # スペクトル減算を行う
        enhanced_mag = mag - avg_noise_spec
        
        floor = 0.02 * mag
        enhanced_mag = np.maximum(enhanced_mag, floor)

        enhanced_D = enhanced_mag * np.exp(1j * phase)
        
        y_enh = librosa.istft(enhanced_D, hop_length=128, length=len(y))
        
        filename = path.name
        
        save_path = os.path.join("enhanced/test", filename)
        
        sf.write(save_path, y_enh, sr)
        
        print("TSE処理済みの学習用とテスト用の音声ファイルを作成しました。")
    
    return


# ノイズファイルから平均的なノイズスペクトルを作る
avg_noise_spec = build_average_noise(noise_files)

# 学習用とテスト用の音声ファイルのスペクトルから、平均的なノイズスペクトルを引いて、ノイズを減らす
spectral_subtraction_denoise(train_wav_path, test_wav_path, avg_noise_spec)