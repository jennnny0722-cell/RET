import pandas as pd
import numpy as np


def clean_and_resample_data(file_path):
    """
    数据清洗与降频
    """
    print("正在加载数据...")
    df = pd.read_parquet(file_path)


    df = df[~df['code'].astype(str).str.startswith('920')] #删北交所
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')

    df = df.drop_duplicates(subset=['code', 'date']).sort_values(['code', 'date'])


    monthly_df = df.groupby(['code', 'year_month']).tail(1).copy()
    keep_cols = ['code', 'year_month', 'close', 'size']
    if 'is_st' in df.columns:
        keep_cols.append('is_st')
    monthly_df = monthly_df[keep_cols]


    codes = monthly_df['code'].unique()
    all_periods = pd.period_range(monthly_df['year_month'].min(), monthly_df['year_month'].max(), freq='M')
    mux = pd.MultiIndex.from_product([codes, all_periods], names=['code', 'year_month'])

    monthly_df = monthly_df.set_index(['code', 'year_month']).reindex(mux).reset_index()

    valid_bounds = monthly_df.dropna(subset=['close']).groupby('code')['year_month'].agg(
        first_month='min',
        last_month='max'
    ).reset_index()

    monthly_df = monthly_df.merge(valid_bounds, on='code')
    monthly_df = monthly_df[(monthly_df['year_month'] >= monthly_df['first_month']) &
                            (monthly_df['year_month'] <= monthly_df['last_month'])].copy()
    monthly_df.drop(columns=['first_month', 'last_month'], inplace=True)


    monthly_df['close'] = monthly_df.groupby('code')['close'].ffill()
    monthly_df['size'] = monthly_df.groupby('code')['size'].ffill()
    if 'is_st' in monthly_df.columns:
        monthly_df['is_st'] = monthly_df.groupby('code')['is_st'].ffill().fillna(0)

    # 计算月度收益率 (当月月末收盘价 / 上月月末收盘价 - 1)
    monthly_df['monthly_ret'] = monthly_df.groupby('code')['close'].pct_change().fillna(0.0)
    # =====================================================================

    print("正在进行极值处理...")
    monthly_df['monthly_ret'] = monthly_df['monthly_ret'].clip(-0.95, 3.0)

    # 截面缩尾
    lower_bound = monthly_df.groupby('year_month')['monthly_ret'].transform(lambda x: x.quantile(0.01))
    upper_bound = monthly_df.groupby('year_month')['monthly_ret'].transform(lambda x: x.quantile(0.99))
    monthly_df['monthly_ret'] = monthly_df['monthly_ret'].clip(lower_bound, upper_bound)

    return monthly_df


def generate_paper_features(monthly_df):
    """
    合成 16 个截面特征与分类 Target
    """
    print("正在合成特征与目标变量...")
    monthly_df = monthly_df.sort_values(['code', 'year_month'])
    monthly_df['ret_plus_1'] = monthly_df['monthly_ret'] + 1

    # 1. 动量特征计算
    monthly_df['MOM1m'] = monthly_df['monthly_ret']

    def calc_lagged_momentum(s, window):
        return s.rolling(window).apply(np.prod, raw=True).shift(1) - 1

    monthly_df['MOM3m'] = monthly_df.groupby('code')['ret_plus_1'].transform(lambda x: calc_lagged_momentum(x, 2))
    monthly_df['MOM6m'] = monthly_df.groupby('code')['ret_plus_1'].transform(lambda x: calc_lagged_momentum(x, 5))
    monthly_df['MOM9m'] = monthly_df.groupby('code')['ret_plus_1'].transform(lambda x: calc_lagged_momentum(x, 8))
    monthly_df['MOM12m'] = monthly_df.groupby('code')['ret_plus_1'].transform(lambda x: calc_lagged_momentum(x, 11))

    # 2. 全市场截面标准化计算
    ms = ['MOM1m', 'MOM3m', 'MOM6m', 'MOM9m', 'MOM12m']
    for m in ms:
        grouped = monthly_df.groupby('year_month')[m]
        mean_col = f'M_{m}'
        std_col = f'S_{m}'
        z_col = f'z{m}'

        monthly_df[mean_col] = grouped.transform('mean')
        monthly_df[std_col] = grouped.transform('std')
        monthly_df[z_col] = (monthly_df[m] - monthly_df[mean_col]) / (monthly_df[std_col] + 1e-8)

    # 3. 市值特征: 全市场十分组
    def safe_size_qcut(x):
        if x.notna().sum() > 10:
            return pd.qcut(x, 10, labels=False, duplicates='drop') + 1
        return np.nan

    monthly_df['SIZE'] = monthly_df.groupby('year_month')['size'].transform(safe_size_qcut)

    # 4. 目标变量: shift(-1) 提取次月真实执行收益率
    monthly_df['next_ret'] = monthly_df.groupby('code')['monthly_ret'].shift(-1)

    # 5. Target 标签: 全市场收益率十分组
    def safe_target_qcut(x):
        if x.notna().sum() > 10:
            return pd.qcut(x, 10, labels=False, duplicates='drop')
        return np.nan

    monthly_df['target_class'] = monthly_df.groupby('year_month')['next_ret'].transform(safe_target_qcut)

    # 剔除ST
    if 'is_st' in monthly_df.columns:
        monthly_df = monthly_df[monthly_df['is_st'] == 0]

    return monthly_df
