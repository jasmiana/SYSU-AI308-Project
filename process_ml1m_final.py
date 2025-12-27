import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

# === 配置路径 ===
RAW_FILE = os.path.join('data', 'ML-1M', 'raw', 'ml-1m', 'ratings.dat')
OUTPUT_DIR = os.path.join('data', 'ML-1M')
RANDOM_SEED = 2025

def generate_negatives(user_id, all_items, user_history_set, num_neg=99):
    """为单个用户生成负样本"""
    negatives = []
    while len(negatives) < num_neg:
        # 随机采样
        candidates = np.random.choice(all_items, num_neg * 2)
        for cand in candidates:
            if cand not in user_history_set and cand not in negatives:
                negatives.append(cand)
                if len(negatives) == num_neg:
                    break
    return negatives

def process_and_split():
    if not os.path.exists(RAW_FILE):
        print(f"❌ 错误: 找不到文件 {RAW_FILE}")
        return

    print("Step 1: 读取并清洗数据...")
    # 原始列顺序: user, item, rating, time
    df = pd.read_csv(RAW_FILE, sep='::', header=None, engine='python', 
                     names=['user_id', 'item_id', 'rating', 'time'])
    
    # 论文复现设置：保留全量数据 (Implicit Feedback)，不进行 rating >= 4 过滤
    # df = df[df['rating'] >= 4].copy()
    print(f"当前数据量: {len(df)}")
    
    # 5-core 过滤
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    
    # ID 重映射
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    user2id = {u: i+1 for i, u in enumerate(unique_users)}
    item2id = {i: i+1 for i, i in enumerate(unique_items)}
    
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)
    
    all_items_array = np.array(list(item2id.values()))
    
    # 排序 (关键)
    df = df.sort_values(by=['user_id', 'time']).reset_index(drop=True)
    
    print(f"Step 2: 留一法切分并生成负样本 (1:99)...")
    np.random.seed(RANDOM_SEED)

    train_data = []
    dev_data = []
    test_data = []

    user_interactions = df.groupby('user_id')['item_id'].apply(set).to_dict()

    for user_id, group in tqdm(df.groupby('user_id')):
        if len(group) < 3:
            continue
        
        interactions = group.values.tolist() # [[uid, iid, rating, time], ...]
        
        test_item = interactions[-1]
        dev_item = interactions[-2]
        train_items = interactions[:-2]

        # 生成负样本
        dev_negs = generate_negatives(user_id, all_items_array, user_interactions[user_id])
        test_negs = generate_negatives(user_id, all_items_array, user_interactions[user_id])

        dev_item.append(str(dev_negs)) 
        test_item.append(str(test_negs))
        
        dev_data.append(dev_item)
        test_data.append(test_item)
        train_data.extend(train_items)

    print("Step 3: 保存文件...")
    
    # Train: 保持原始 DataFrame 的列顺序读取
    df_train = pd.DataFrame(train_data, columns=df.columns)[['user_id', 'item_id', 'time']]
    
    # Dev/Test: 必须显式指定正确的列顺序!
    # 原始数据列是 ['user_id', 'item_id', 'rating', 'time']，我们append了 'neg_items'
    correct_cols = ['user_id', 'item_id', 'rating', 'time', 'neg_items']
    
    df_dev = pd.DataFrame(dev_data, columns=correct_cols)[['user_id', 'item_id', 'time', 'neg_items']]
    df_test = pd.DataFrame(test_data, columns=correct_cols)[['user_id', 'item_id', 'time', 'neg_items']]

    # 保存
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), sep='\t', index=False)
    df_dev.to_csv(os.path.join(OUTPUT_DIR, 'dev.csv'), sep='\t', index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), sep='\t', index=False)

    print(f"✅ 完成！")
    print(f"Train: {len(df_train)}, Dev: {len(df_dev)}, Test: {len(df_test)}")

if __name__ == '__main__':
    process_and_split()