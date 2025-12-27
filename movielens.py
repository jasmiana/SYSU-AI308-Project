import numpy as np
import pandas as pd
import os
import sys
import zipfile
import urllib.request
import json
from datetime import datetime
from tqdm import tqdm

# --- 配置部分 ---
DATASET = 'ml-1m'
RAW_PATH = os.path.join('./', DATASET)
CTR_PATH = './ML_1MCTR/'
TOPK_PATH = './ML_1MTOPK/'

RANDOM_SEED = 0
NEG_ITEMS = 99

# 设置随机种子
np.random.seed(RANDOM_SEED)

def download_and_extract():
    """下载并解压数据，替代原Notebook中的curl命令"""
    if not os.path.exists(RAW_PATH):
        os.makedirs(RAW_PATH)
    
    zip_path = os.path.join(RAW_PATH, DATASET + '.zip')
    
    if not os.path.exists(zip_path):
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        print(f'正在下载数据到 {RAW_PATH} ...')
        try:
            urllib.request.urlretrieve(url, zip_path)
            print('下载完成。')
        except Exception as e:
            print(f"下载失败: {e}")
            sys.exit(1)
            
    # 检查是否已经解压（简单检查是否存在ratings.dat）
    if not os.path.exists(os.path.join(RAW_PATH, 'ratings.dat')):
        print('正在解压文件...')
        try:
            with zipfile.ZipFile(zip_path, 'r') as f:
                # 原始压缩包内包含 ml-1m 文件夹，我们需要将其内容提取到 RAW_PATH
                # 原代码逻辑是将压缩包内容提取到 RAW_PATH 下
                # 这里的逻辑稍微调整以确保路径正确
                for file in f.namelist():
                    if file.startswith('ml-1m/'):
                        f.extract(file, './') # 解压到当前目录，会生成 ml-1m 文件夹，正好对应 RAW_PATH
            print("解压完成。")
        except zipfile.BadZipFile:
            print("压缩文件损坏，请删除后重试。")
            sys.exit(1)

def process_data():
    print("开始处理交互数据...")
    
    # --- 1. 读取并过滤数据 ---
    interactions = []
    user_freq, item_freq = dict(), dict()
    file_path = os.path.join(RAW_PATH, "ratings.dat")
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print("读取 ratings.dat...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as F:
        # 这里的tqdm用于显示进度
        lines = F.readlines()
        for line in tqdm(lines, desc="解析原始数据"):
            line = line.strip().split("::")
            if len(line) < 4: continue
            uid, iid, rating, time = line[0], line[1], float(line[2]), float(line[3])
            
            # 标签生成逻辑
            if rating >= 4:
                label = 1
            else:
                label = 0
            
            interactions.append([uid, time, iid, label])
            if int(label) == 1:
                user_freq[uid] = user_freq.get(uid, 0) + 1
                item_freq[iid] = item_freq.get(iid, 0) + 1

    # --- 2. 5-core 过滤 (迭代直到所有用户和物品至少有5次交互) ---
    print("执行 5-core 过滤...")
    select_uid = []
    select_iid = []
    
    # 初始检查
    while len(select_uid) < len(user_freq) or len(select_iid) < len(item_freq):
        select_uid = []
        select_iid = []
        
        for u in user_freq:
            if user_freq[u] >= 5:
                select_uid.append(u)
        for i in item_freq:
            if item_freq[i] >= 5:
                select_iid.append(i)
                
        print(f"当前状态 -> 用户: {len(select_uid)}/{len(user_freq)}, 物品: {len(select_iid)}/{len(item_freq)}")
        
        select_uid_set = set(select_uid)
        select_iid_set = set(select_iid)
        
        if len(select_uid) == len(user_freq) and len(select_iid) == len(item_freq):
            break
            
        user_freq, item_freq = dict(), dict()
        interactions_5core = []
        
        for line in tqdm(interactions, desc="过滤中"):
            uid, iid, label = line[0], line[2], line[-1]
            if uid in select_uid_set and iid in select_iid_set:
                interactions_5core.append(line)
                if int(label) == 1:
                    user_freq[uid] = user_freq.get(uid, 0) + 1
                    item_freq[iid] = item_freq.get(iid, 0) + 1
        
        interactions = interactions_5core

    print(f"过滤完成。保留交互数: {len(interactions)}")

    # --- 3. 特征工程 (时间戳处理) ---
    print("处理时间特征...")
    ts = []
    for item in tqdm(interactions, desc="转换时间戳"):
        ts.append(datetime.fromtimestamp(item[1]))

    # 构建 DataFrame
    interaction_df = pd.DataFrame(interactions, columns=["user_id", "time", "news_id", "label"])
    interaction_df['timestamp'] = ts
    interaction_df['hour'] = interaction_df['timestamp'].apply(lambda x: x.hour)
    interaction_df['weekday'] = interaction_df['timestamp'].apply(lambda x: x.weekday())
    interaction_df['date'] = interaction_df['timestamp'].apply(lambda x: x.date())

    def get_time_range(hour):
        if 5 <= hour <= 8: return 0
        if 8 < hour < 11: return 1
        if 11 <= hour <= 12: return 2
        if 12 < hour <= 15: return 3
        if 15 < hour <= 17: return 4
        if 18 <= hour <= 19: return 5
        if 19 < hour <= 21: return 6
        if hour > 21: return 7
        return 8 # 0-4 am

    interaction_df['period'] = interaction_df.hour.apply(lambda x: get_time_range(x))
    min_date = interaction_df.date.min()
    interaction_df['day'] = (interaction_df.date - min_date).apply(lambda x: x.days)

    # 保存中间结果
    interaction_df.to_csv("interaction_5core.csv", index=False)
    
    # 修复原Notebook中的拼写错误 (interaciton_df -> interaction_df)
    # 并且确保ID是整型，以便后续排序正确
    interaction_df["user_id"] = interaction_df["user_id"].astype(int)
    interaction_df["news_id"] = interaction_df["news_id"].astype(int) # 原代码中用了 item_id 和 news_id 混用，这里列名是 news_id

    return interaction_df

def generate_ctr_data(interaction_df):
    """生成 CTR (点击率预测) 任务所需的数据"""
    print("\n正在生成 CTR 任务数据...")
    os.makedirs(CTR_PATH, exist_ok=True)
    
    interaction_ctr = interaction_df.copy()
    # 重命名列
    interaction_ctr.rename(columns={
        'hour': 'c_hour_c',
        'weekday': 'c_weekday_c',
        'period': 'c_period_c',
        'day': 'c_day_f',
        'user_id': 'original_user_id'
    }, inplace=True)

    # 重新映射 User ID
    user2newid_ctr = dict(zip(sorted(interaction_ctr.original_user_id.unique()), 
                          range(1, interaction_ctr.original_user_id.nunique() + 1)))
    interaction_ctr['user_id'] = interaction_ctr.original_user_id.apply(lambda x: user2newid_ctr[x])

    # 重新映射 Item ID (news_id)
    item2newid_ctr = dict(zip(sorted(interaction_ctr.news_id.unique()), 
                          range(1, interaction_ctr.news_id.nunique() + 1)))
    interaction_ctr['item_id'] = interaction_ctr['news_id'].apply(lambda x: item2newid_ctr[x])

    # 排序
    interaction_ctr.sort_values(by=['user_id', 'time'], inplace=True)
    interaction_ctr = interaction_ctr.reset_index(drop=True)

    # 保存映射关系
    nu2nid = {int(k): v for k, v in user2newid_ctr.items()}
    ni2nid = {int(k): v for k, v in item2newid_ctr.items()}
    with open(os.path.join(CTR_PATH, "user2newid.json"), 'w') as f:
        json.dump(nu2nid, f)
    with open(os.path.join(CTR_PATH, "item2newid.json"), 'w') as f:
        json.dump(ni2nid, f)

    # 划分数据集 (Train/Val/Test)
    split_time1 = interaction_ctr.c_day_f.max() * 0.8
    train = interaction_ctr.loc[interaction_ctr.c_day_f <= split_time1].copy()
    val_test = interaction_ctr.loc[(interaction_ctr.c_day_f > split_time1)].copy()
    
    split_time2 = interaction_ctr.c_day_f.max() * 0.9
    val = val_test.loc[val_test.c_day_f <= split_time2].copy()
    test = val_test.loc[val_test.c_day_f > split_time2].copy()

    # 过滤 Val/Test 中未在 Train 出现的用户/物品
    train_u, train_i = set(train.user_id.unique()), set(train.item_id.unique())
    val_sel = val.loc[(val.user_id.isin(train_u)) & (val.item_id.isin(train_i))].copy()
    test_sel = test.loc[(test.user_id.isin(train_u)) & (test.item_id.isin(train_i))].copy()

    print(f"[CTR] Train: {len(train)}, Val: {len(val_sel)}, Test: {len(test_sel)}")

    # 生成 Impression IDs
    print("[CTR] 生成 Impression IDs...")
    for interaction_partial in [train, val_sel, test_sel]:
        interaction_partial['last_user_id'] = interaction_partial['user_id'].shift(1)
        impression_ids = []
        impression_len = 0
        current_impid = 0
        max_imp_len = 20
        
        # 使用 numpy 数组加速遍历
        data_values = interaction_partial[['user_id', 'last_user_id']].to_numpy()
        
        for uid, last_uid in tqdm(data_values, desc="Impression ID"):
            if uid == last_uid:
                if impression_len >= max_imp_len:
                    current_impid += 1
                    impression_len = 1
                else:
                    impression_len += 1
                impression_ids.append(current_impid)
            else:
                current_impid += 1
                impression_len = 1
                impression_ids.append(current_impid)
        
        interaction_partial.loc[:, 'impression_id'] = impression_ids

    # 保存 CSV
    select_columns = ['user_id','item_id','time','label','c_hour_c','c_weekday_c','c_period_c','c_day_f','impression_id']
    train[select_columns].to_csv(os.path.join(CTR_PATH, 'train.csv'), sep="\t", index=False)
    val_sel[select_columns].to_csv(os.path.join(CTR_PATH, 'dev.csv'), sep="\t", index=False)
    test_sel[select_columns].to_csv(os.path.join(CTR_PATH, 'test.csv'), sep="\t", index=False)
    print(f"[CTR] 文件已保存至 {CTR_PATH}")
    
    return item2newid_ctr

def generate_topk_data(interaction_df):
    """生成 Top-K 推荐任务所需的数据"""
    print("\n正在生成 Top-K 任务数据...")
    os.makedirs(TOPK_PATH, exist_ok=True)

    # 仅保留正样本 (label == 1)
    interaction_pos = interaction_df.loc[interaction_df.label == 1].copy()
    interaction_pos.rename(columns={
        'hour': 'c_hour_c',
        'weekday': 'c_weekday_c',
        'period': 'c_period_c',
        'day': 'c_day_f',
        'user_id': 'original_user_id'
    }, inplace=True)

    # 划分数据集
    split_time1 = int(interaction_pos.c_day_f.max() * 0.8)
    train = interaction_pos.loc[interaction_pos.c_day_f <= split_time1].copy()
    val_test = interaction_pos.loc[(interaction_pos.c_day_f > split_time1)].copy()
    val_test.sort_values(by='time', inplace=True)
    
    split_time2 = int(interaction_pos.c_day_f.max() * 0.9)
    val = val_test.loc[val_test.c_day_f <= split_time2].copy()
    test = val_test.loc[val_test.c_day_f > split_time2].copy()

    # 过滤
    train_u, train_i = set(train.original_user_id.unique()), set(train.news_id.unique())
    val_sel = val.loc[(val.original_user_id.isin(train_u)) & (val.news_id.isin(train_i))].copy()
    test_sel = test.loc[(test.original_user_id.isin(train_u)) & (test.news_id.isin(train_i))].copy()

    # 重新映射 ID (统一映射)
    all_df = pd.concat([train, val_sel, test_sel], axis=0)
    user2newid_topk = dict(zip(sorted(all_df.original_user_id.unique()), 
                           range(1, all_df.original_user_id.nunique() + 1)))
    
    item2newid_topk = dict(zip(sorted(all_df.news_id.unique()), 
                           range(1, all_df.news_id.nunique() + 1)))

    # 应用映射
    for df in [train, val_sel, test_sel]:
        df['user_id'] = df.original_user_id.apply(lambda x: user2newid_topk[x])
        df['item_id'] = df['news_id'].apply(lambda x: item2newid_topk[x])

    # 保存映射
    nu2nid = {int(k): v for k, v in user2newid_topk.items()}
    ni2nid = {int(k): v for k, v in item2newid_topk.items()}
    with open(os.path.join(TOPK_PATH, "user2newid.json"), 'w') as f:
        json.dump(nu2nid, f)
    with open(os.path.join(TOPK_PATH, "item2newid.json"), 'w') as f:
        json.dump(ni2nid, f)

    # 生成负样本 (最耗时的部分)
    print("[Top-K] 生成负样本 (可能需要几分钟)...")
    
    def generate_negative(data_df, all_items, clicked_item_set, random_seed, neg_item_num=99):
        np.random.seed(random_seed)
        neg_items = np.random.choice(all_items, (len(data_df), neg_item_num))
        # 转换为列表以提高迭代修改速度
        neg_items = neg_items.tolist()
        
        user_ids = data_df['user_id'].values
        
        for i, uid in tqdm(enumerate(user_ids), total=len(user_ids), desc="负采样"):
            user_clicked = clicked_item_set[uid]
            current_row_negatives = neg_items[i]
            
            # 检查生成的负样本是否其实被用户点击过
            for j in range(neg_item_num):
                while current_row_negatives[j] in user_clicked:
                    # 如果撞车了，重新随机选一个
                    current_row_negatives[j] = np.random.choice(all_items)
            neg_items[i] = current_row_negatives
            
        return neg_items

    # 准备负采样所需的数据结构
    # 这里需要重新构建 all_df，因为上面的 df 是拷贝
    all_df_mapped = pd.concat([train, val_sel, test_sel], axis=0)
    
    clicked_item_set = dict()
    # 优化：groupby 比较慢，预先构建字典
    for user_id, seq_df in all_df_mapped.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())
        
    all_items = all_df_mapped.item_id.unique()
    
    val_sel['neg_items'] = generate_negative(val_sel, all_items, clicked_item_set, random_seed=1)
    test_sel['neg_items'] = generate_negative(test_sel, all_items, clicked_item_set, random_seed=2)

    # 保存 CSV
    select_columns = ['user_id','item_id','time','c_hour_c','c_weekday_c','c_period_c','c_day_f']
    train[select_columns].to_csv(os.path.join(TOPK_PATH, 'train.csv'), sep="\t", index=False)
    val_sel[select_columns + ['neg_items']].to_csv(os.path.join(TOPK_PATH, 'dev.csv'), sep="\t", index=False)
    test_sel[select_columns + ['neg_items']].to_csv(os.path.join(TOPK_PATH, 'test.csv'), sep="\t", index=False)
    
    print(f"[Top-K] 文件已保存至 {TOPK_PATH}")
    return item2newid_topk

def save_metadata(interaction_df, ctr_map, topk_map):
    """保存电影元数据"""
    print("正在保存元数据...")
    movies_path = os.path.join(RAW_PATH, "movies.dat")
    if not os.path.exists(movies_path):
        return

    item_meta = pd.read_csv(
        movies_path, 
        sep='::', 
        names=['movieId', 'title', 'genres'], 
        encoding='latin-1', 
        engine='python'
    )
    
    # 保存 CTR 的元数据
    if ctr_map:
        item_select = item_meta.loc[item_meta.movieId.isin(interaction_df.news_id.unique())].copy()
        # 这里的 map key 是 int，movieId 是 int
        item_select['item_id'] = item_select.movieId.apply(lambda x: ctr_map.get(x, -1))
        # 过滤掉没有ID映射的项 (理论上不应该有)
        item_select = item_select[item_select['item_id'] != -1]
        
        genres2id = dict(zip(sorted(item_select.genres.unique()), range(1, item_select.genres.nunique() + 1)))
        item_select['i_genre_c'] = item_select['genres'].apply(lambda x: genres2id[x])
        
        title2id = dict(zip(sorted(item_select.title.unique()), range(1, item_select.title.nunique() + 1)))
        item_select['i_title_c'] = item_select['title'].apply(lambda x: title2id[x])
        
        item_select[['item_id', 'i_genre_c', 'i_title_c']].to_csv(
            os.path.join(CTR_PATH, 'item_meta.csv'), sep="\t", index=False
        )

    # 保存 Top-K 的元数据
    if topk_map:
        # Top-K 只用了正样本，所以 item 集合可能比 interaction_df 小
        # 我们使用 topk_map 的 keys 来过滤
        valid_items = set(topk_map.keys())
        item_select = item_meta.loc[item_meta.movieId.isin(valid_items)].copy()
        
        item_select['item_id'] = item_select.movieId.apply(lambda x: topk_map.get(x, -1))
        
        genres2id = dict(zip(sorted(item_select.genres.unique()), range(1, item_select.genres.nunique() + 1)))
        item_select['i_genre_c'] = item_select['genres'].apply(lambda x: genres2id[x])
        
        title2id = dict(zip(sorted(item_select.title.unique()), range(1, item_select.title.nunique() + 1)))
        item_select['i_title_c'] = item_select['title'].apply(lambda x: title2id[x])
        
        item_select[['item_id', 'i_genre_c', 'i_title_c']].to_csv(
            os.path.join(TOPK_PATH, 'item_meta.csv'), sep="\t", index=False
        )

if __name__ == "__main__":
    download_and_extract()
    base_df = process_data()
    if base_df is not None:
        ctr_mapping = generate_ctr_data(base_df)
        topk_mapping = generate_topk_data(base_df)
        save_metadata(base_df, ctr_mapping, topk_mapping)
        print("\n=== 全部任务完成 ===")