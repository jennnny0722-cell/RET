import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def train_and_evaluate_deep_momentum(data_path, min_train_years=10, n_ensemble=100):
    """
    滚动训练 XGBoost，计算 RET 重分类预测及各分组概率。
    """
    df = pd.read_parquet(data_path)
    if not pd.api.types.is_datetime64_any_dtype(df['year_month']):
        df['year_month'] = pd.to_datetime(df['year_month'].astype(str))

    features = [
        'zMOM1m', 'zMOM3m', 'zMOM6m', 'zMOM9m', 'zMOM12m',
        'M_MOM1m', 'M_MOM3m', 'M_MOM6m', 'M_MOM9m', 'M_MOM12m',
        'S_MOM1m', 'S_MOM3m', 'S_MOM6m', 'S_MOM9m', 'S_MOM12m',
        'SIZE'
    ]

    df_valid = df.copy()
    df_valid['year'] = df_valid['year_month'].dt.year

    years = sorted(df_valid['year'].dropna().unique())
    if len(years) < min_train_years + 1:
        raise ValueError("数据时间跨度不足以进行10年滚动训练。")

    out_of_sample_results = []

    for train_end_year in range(years[0] + min_train_years - 1, years[-1]):
        test_year = train_end_year + 1
        print(f" -> 训练期: {years[0]}-{train_end_year} | 预测测试期: {test_year}")

        train_data = df_valid[df_valid['year'] <= train_end_year].copy()
        train_data = train_data.dropna(subset=['target_class', 'next_ret'])

        test_data = df_valid[df_valid['year'] == test_year].copy()

        if test_data.empty:
            continue

        X_train_full = train_data[features]
        y_train_full = train_data['target_class'].astype(int)
        past_10_data = train_data[train_data['year'] > train_end_year - 10]
        mu_k = past_10_data.groupby('target_class')['next_ret'].mean().to_dict()
        mu_k_array = np.array([mu_k.get(k, 0) for k in range(10)])

        X_test = test_data[features]
        test_probs = np.zeros((len(X_test), 10))

        for i in range(n_ensemble):
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=i
            )

            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=10,
                eval_metric='mlogloss',
                early_stopping_rounds=10,
                n_estimators=500,
                learning_rate=0.1,
                max_depth=4,
                random_state=i,
                n_jobs=-1,
                missing=np.nan
            )

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            probs = model.predict_proba(X_test)
            for idx, cls in enumerate(model.classes_):
                test_probs[:, int(cls)] += probs[:, idx]

        test_probs /= n_ensemble

        test_data['predicted_ret'] = np.dot(test_probs, mu_k_array)
        test_data['predicted_class'] = np.argmax(test_probs, axis=1)

        prob_cols = []
        for c in range(10):
            col_name = f'prob_class_{c}'
            test_data[col_name] = test_probs[:, c]
            prob_cols.append(col_name)

        output_cols = ['code', 'year_month', 'next_ret', 'predicted_ret', 'predicted_class'] + prob_cols
        out_of_sample_results.append(test_data[output_cols])

    all_predictions = pd.concat(out_of_sample_results)
    return all_predictions
