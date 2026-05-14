import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def run_decile_backtest(data_path):
    """
    基于Deep Momentum预测结果执行十分组多空回测，并输出核心指标与图表。
    """
    print(f"正在加载预测数据: {data_path} ...")

    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    df['year_month'] = pd.to_datetime(df['year_month'])

    # 1. 数据清洗与过滤
    initial_len = len(df)
    df = df.dropna(subset=['predicted_ret', 'next_ret'])  # 剔除缺乏下期真实收益率或预测收益率的无效行（例如最新一期待预测数据）

    print(f"数据清洗完毕。参与回测的有效观测样本数为: {len(df)} (剔除无 next_ret 标签的数据)。")

    if df.empty:
        raise ValueError("有效回测数据为空，请检查数据源的 next_ret 列。")

    # 2. 横截面十分组 (Decile Sorting)
    # 按月根据 predicted_ret 降序分组。10 组为 Winner (多头)，1 组为 Loser (空头)
    df['decile'] = df.groupby('year_month')['predicted_ret'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
    )

    # 3. 计算各分组的等权平均真实收益率
    # 行索引为 year_month，列索引为 1 到 10 的分组编号
    decile_returns = df.groupby(['year_month', 'decile'])['next_ret'].mean().unstack()

    # 4. 构建组合收益
    # 纯多头组合
    long_only_ret = decile_returns[10]
    # 多空组合
    long_short_ret = decile_returns[10] - decile_returns[1]

    # 计算对数累计净值 (基准设定为 1)
    cum_ls_ret = (1 + long_short_ret).cumprod()
    cum_long_ret = (1 + long_only_ret).cumprod()

    # 5. 计算回测指标
    def calculate_metrics(returns, name):
        ann_ret = returns.mean() * 12
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan

        cum_r = (1 + returns).cumprod()
        roll_max = cum_r.cummax()
        drawdown = cum_r / roll_max - 1
        mdd = drawdown.min()

        return ann_ret, ann_vol, sharpe, mdd, drawdown

    ls_ann_ret, ls_ann_vol, ls_sharpe, ls_mdd, ls_drawdown = calculate_metrics(long_short_ret, "Long-Short")
    lo_ann_ret, lo_ann_vol, lo_sharpe, lo_mdd, _ = calculate_metrics(long_only_ret, "Long-Only")

    # 输出报告
    print("\n" + "=" * 45)
    print("      Deep Momentum 策略回测结果")
    print("=" * 45)
    print(f"{'指标':<15} | {'多空组合 (L/S)':<12} | {'纯多头 (Long)':<12}")
    print("-" * 45)
    print(f"{'年化收益率':<15} | {ls_ann_ret:>10.2%} | {lo_ann_ret:>10.2%}")
    print(f"{'年化波动率':<15} | {ls_ann_vol:>10.2%} | {lo_ann_vol:>10.2%}")
    print(f"{'夏普比率':<15} | {ls_sharpe:>10.3f} | {lo_sharpe:>10.3f}")
    print(f"{'最大回撤':<15} | {ls_mdd:>10.2%} | {lo_mdd:>10.2%}")
    print("=" * 45)

    # 6. 可视化图表生成
    fig = plt.figure(figsize=(15, 12))

    # 图 1: 累计净值曲线
    ax1 = plt.subplot(2, 2, (1, 2))
    ax1.plot(cum_ls_ret.index, cum_ls_ret, label='Long-Short (Top 10% - Bottom 10%)', color='#1f77b4', linewidth=2)
    ax1.plot(cum_long_ret.index, cum_long_ret, label='Long Only (Top 10%)', color='#ff7f0e', linewidth=1.5,
             linestyle='--')
    ax1.set_title('Cumulative Return (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative NAV')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 图 2: 十分组收益单调性柱状图
    ax2 = plt.subplot(2, 2, 3)
    avg_decile_ret = decile_returns.mean() * 12  # 转化为年化
    avg_decile_ret.plot(kind='bar', ax=ax2, color='#2ca02c', alpha=0.8, edgecolor='black')
    ax2.set_title('Annualized Return by Decile', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Decile (1 = Lowest Pred, 10 = Highest Pred)')
    ax2.set_ylabel('Annualized Return')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    # 图 3: 动态回撤图
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(ls_drawdown.index, ls_drawdown, color='#d62728', alpha=0.8, linewidth=1.2)
    ax3.fill_between(ls_drawdown.index, ls_drawdown, 0, color='#d62728', alpha=0.3)
    ax3.set_title('Long-Short Portfolio Drawdown', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Drawdown')
    # 设置 Y 轴格式为百分比
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return decile_returns, long_short_ret
