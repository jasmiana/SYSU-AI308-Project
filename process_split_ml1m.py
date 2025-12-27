import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# === 配置路径 ===
# 原始数据路径 (请确保该文件存在)
RAW_FILE = os.path.join('data', 'ML-1M', 'raw', 'ml-1m', 'ratings.dat')
# 输出目录
OUTPUT_DIR = os.path.join('data', 'ML-1M')

def process_and_split():
    if not os.path.exists(RAW_FILE):
        print(f"❌ 错误: 找不到文件 {RAW_FILE}")
        return

    print("Step 1: 读取并清洗数据...")
    # 1. 读取数据
    df = pd.read_csv(RAW_FILE, sep='::', header=None, engine='python', 
                     names=['user_id', 'item_id', 'rating', 'time'])
    
    # 2. 过滤评分 < 4
    df = df[df['rating'] >= 4].copy()
    
    # 3. 5-core 过滤
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    
    # 4. ID 重映射 (从 1 开始)
    user2id = {u: i+1 for i, u in enumerate(sorted(df['user_id'].unique()))}
    item2id = {i: i+1 for i, i in enumerate(sorted(df['item_id'].unique()))}
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)
    
    # 5. 排序 (按用户和时间)
    df = df.sort_values(by=['user_id', 'time']).reset_index(drop=True)
    
    print(f"Step 2: 执行留一法切分 (Leave-One-Out)...")
    print(f"总交互数: {len(df)}, 用户数: {df['user_id'].nunique()}, 物品数: {df['item_id'].nunique()}")

    train_data = []
    dev_data = []
    test_data = []

    # 按用户分组处理
    # 最后一个交互 -> Test
    # 倒数第二个交互 -> Dev
    # 剩下的 -> Train
    for user_id, group in tqdm(df.groupby('user_id')):
        if len(group) < 3:
            # 如果序列太短，全放进训练集，或者忽略
            train_data.append(group)
            continue
        
        # 转换为 list 以便切分
        interactions = group.values.tolist() # [[uid, iid, rating, time], ...]
        
        test_data.append(interactions[-1])   # 最后一个
        dev_data.append(interactions[-2])    # 倒数第二个
        train_data.append(interactions[:-2]) # 剩下的所有

    # 重新构建 DataFrame
    # 注意 train_data 是 list of lists of lists，需要展平
    train_flat = [item for sublist in train_data for item in sublist]
    
    df_train = pd.DataFrame(train_flat, columns=df.columns)[['user_id', 'item_id', 'time']]
    df_dev = pd.DataFrame(dev_data, columns=df.columns)[['user_id', 'item_id', 'time']]
    df_test = pd.DataFrame(test_data, columns=df.columns)[['user_id', 'item_id', 'time']]

    print("Step 3: 保存文件...")
    # 保存为 ReChorus 标准的 csv 格式 (tab 分隔)
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), sep='\t', index=False)
    df_dev.to_csv(os.path.join(OUTPUT_DIR, 'dev.csv'), sep='\t', index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), sep='\t', index=False)
    
    # 顺便保存一个总的，以备不时之需
    df[['user_id', 'item_id', 'time']].to_csv(os.path.join(OUTPUT_DIR, 'ML-1M.csv'), sep='\t', index=False)

    print("✅ 完成！已生成 train.csv, dev.csv, test.csv")
    print(f"训练集大小: {len(df_train)}")
    print(f"验证集大小: {len(df_dev)}")
    print(f"测试集大小: {len(df_test)}")

if __name__ == '__main__':
    process_and_split()