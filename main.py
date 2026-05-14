import os
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import clean_and_resample_data, generate_paper_features
from xgb import train_and_evaluate_deep_momentum
from backtest import run_decile_backtest


def main():
    # ==========================================
    # 路径配置
    # ==========================================
    BASE_DIR = "D:/Quant/Quant/xgb_code"
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data_new.parquet")
    FEATURES_OUT_PATH = os.path.join(BASE_DIR, "features_processed.parquet")
    PREDICTIONS_OUT_PATH = os.path.join(BASE_DIR, "predictions.parquet")

    # ==========================================
    # 数据清洗与特征生成
    # ==========================================
    if not os.path.exists(FEATURES_OUT_PATH):
        print("\n[Step 1/3] 开始执行数据清洗与特征生成...")
        cleaned_df = clean_and_resample_data(RAW_DATA_PATH)
        features_df = generate_paper_features(cleaned_df)
        features_df.to_parquet(FEATURES_OUT_PATH)
        print(f"特征已生成: {FEATURES_OUT_PATH}")
    else:
        print(f"\n[Step 1/3] 本地已存在特征缓存，跳过数据处理。")
        print(f"读取路径: {FEATURES_OUT_PATH}")

    # ==========================================
    # XGBoost 滚动训练与预测阶段
    # ==========================================
    print("\n[Step 2/3] 开始执行 XGBoost 模型训练与样本外预测...")
    predictions_df = train_and_evaluate_deep_momentum(
        data_path=FEATURES_OUT_PATH,
        min_train_years=10,
        n_ensemble=5  # n_ensemble: 集成树的随机种子循环次数。论文中设置为100
    )

    # 存入信号数据
    predictions_df.to_parquet(PREDICTIONS_OUT_PATH)
    print(f"信号已生成: {PREDICTIONS_OUT_PATH}")

    # ==========================================
    # 策略回测
    # ==========================================
    print("\n[Step 3/3] 开始回测...")
    # 传入刚生成的预测文件，执行十分组截面回测并渲染图表
    run_decile_backtest(PREDICTIONS_OUT_PATH)


if __name__ == "__main__":
    main()