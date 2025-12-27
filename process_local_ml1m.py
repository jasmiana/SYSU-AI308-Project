import os
import pandas as pd

# 配置路径 (根据你提供的目录结构)
# 原始数据路径: data/ML-1M/raw/ml-1m/ratings.dat
RAW_FILE = os.path.join('data', 'ML-1M', 'raw', 'ml-1m', 'ratings.dat')
# 目标输出路径: data/ML-1M/ML-1M.csv
OUTPUT_FILE = os.path.join('data', 'ML-1M', 'ML-1M.csv')

def process_data():
    if not os.path.exists(RAW_FILE):
        print(f"错误: 找不到原始文件 {RAW_FILE}")
        print("请确认 ratings.dat 是否在 data/ML-1M/raw/ml-1m/ 目录下")
        return

    print("正在读取 ratings.dat ...")
    # ML-1M 的格式: UserID::MovieID::Rating::Timestamp
    # 这里的 engine='python' 是为了处理多字符分隔符 '::'
    df = pd.read_csv(RAW_FILE, sep='::', header=None, engine='python', 
                     names=['user_id', 'item_id', 'rating', 'time'])
    
    print(f"原始交互数量: {len(df)}")

    # 1. 过滤评分 (保留 Rating >= 4) - ReChorus 通用惯例
    df = df[df['rating'] >= 4].copy()
    print(f"保留评分>=4后: {len(df)}")

    # 2. 5-core 过滤 (确保每个用户和物品至少有5条记录)
    print("正在进行 5-core 过滤...")
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        
        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            break
            
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    
    print(f"过滤完成: {len(df)} 条交互")
    print(f"用户数: {df['user_id'].nunique()}, 物品数: {df['item_id'].nunique()}")

    # 3. ID 重映射 (ReChorus 要求 ID 从 1 开始连续)
    print("正在重新映射 ID...")
    user2id = {u: i+1 for i, u in enumerate(sorted(df['user_id'].unique()))}
    item2id = {i: i+1 for i, i in enumerate(sorted(df['item_id'].unique()))}
    
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)

    # 4. 保存文件 (Tab 分隔)
    # 只保留 ReChorus 需要的三列
    out_df = df[['user_id', 'item_id', 'time']]
    out_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    
    print("-" * 30)
    print(f"✅ 成功! 数据已保存至: {OUTPUT_FILE}")
    print("现在你可以运行 ML-1M 的实验了！")

if __name__ == '__main__':
    process_data()