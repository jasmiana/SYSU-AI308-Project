import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm

# 配置
DATASET_NAME = 'ML-1M'
DATA_DIR = os.path.join('data', DATASET_NAME)
RAW_URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
ZIP_FILE = os.path.join(DATA_DIR, 'ml-1m.zip')
EXTRACT_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_FILE = os.path.join(DATA_DIR, f'{DATASET_NAME}.csv')

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # 1. 下载
    if not os.path.exists(ZIP_FILE):
        print(f"正在下载 {DATASET_NAME} ...")
        urllib.request.urlretrieve(RAW_URL, ZIP_FILE)
    
    # 2. 解压
    print("正在解压...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

def process_data():
    print("正在处理数据...")
    # ML-1M 的 ratings.dat 格式: UserID::MovieID::Rating::Timestamp
    raw_file = os.path.join(EXTRACT_DIR, 'ml-1m', 'ratings.dat')
    
    # 读取数据 (使用 Python 引擎处理 :: 分隔符)
    df = pd.read_csv(raw_file, sep='::', header=None, engine='python', 
                     names=['user_id', 'item_id', 'rating', 'time'])
    
    print(f"原始数据量: {len(df)}")

    # 1. 隐式反馈过滤 (保留评分 >= 4 的交互，根据 ReChorus 惯例)
    # 如果你想保留所有交互，注释掉下面两行
    df = df[df['rating'] >= 4].copy()
    print(f"评分>=4 过滤后: {len(df)}")

    # 2. 5-core 过滤 (用户和商品至少出现 5 次)
    # 循环过滤直到稳定
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        
        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break
            
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    
    print(f"5-core 过滤后: {len(df)} (Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()})")

    # 3. 重新编码 ID (ReChorus 要求 ID 从 1 开始连续)
    # user_id 映射
    unique_users = sorted(df['user_id'].unique())
    user2id = {u: i+1 for i, u in enumerate(unique_users)}
    df['user_id'] = df['user_id'].map(user2id)
    
    # item_id 映射
    unique_items = sorted(df['item_id'].unique())
    item2id = {i: i+1 for i, i in enumerate(unique_items)}
    df['item_id'] = df['item_id'].map(item2id)

    # 4. 保存为 ReChorus 标准格式 (user_id, item_id, time)
    # 使用制表符 \t 分隔 (ReChorus 默认)
    df = df[['user_id', 'item_id', 'time']]
    df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"数据已保存至: {OUTPUT_FILE}")
    print("前 5 行预览:")
    print(df.head())

if __name__ == '__main__':
    download_and_extract()
    process_data()