import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class OlsModel:
    def __init__(self):
        self.train_data_path = "/kaggle/input/avenir-hku-web/kline_data/train_data"
        self.submission_id_path = "/kaggle/input/avenir-hku-web/submission_id.csv"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.all_train_data = []

    def get_all_symbol_list(self):
        try:
            parquet_name_list = os.listdir(self.train_data_path)
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list if parquet_name.endswith(".parquet")]
            return symbol_list
        except Exception as e:
            print(f"get_all_symbol_list error: {e}")
            return []

    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df = df.set_index("timestamp")
            df = df.astype(np.float64)
            df['vwap'] = np.where(df['volume'] == 0, np.nan, df['amount'] / df['volume'])
            df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan).ffill()
            df.columns.name = symbol
            return df
        except Exception as e:
            print(f"get_single_symbol_kline_data error (symbol: {symbol}): {e}")
            return pd.DataFrame()

    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        all_symbol_list = self.get_all_symbol_list()
        print(f"[get_all_symbol_kline] 开始读取币种数据，共{len(all_symbol_list)}个币种...")
        if not all_symbol_list:
            return [], [], [], [], [], [], [], []
        df_list = []
        for idx, symbol in enumerate(all_symbol_list):
            if (idx+1) % 100 == 0 or idx == 0 or idx == len(all_symbol_list)-1:
                print(f"[get_all_symbol_kline] 进度: {idx+1}/{len(all_symbol_list)}")
            df = self.get_single_symbol_kline_data(symbol)
            df_list.append(df)
        print(f"[get_all_symbol_kline] 数据拼接完成, 用时: {datetime.datetime.now() - t0}")
        df_open_price = pd.concat([df['open_price'] for df in df_list], axis=1)
        df_high_price = pd.concat([df['high_price'] for df in df_list], axis=1)
        df_low_price = pd.concat([df['low_price'] for df in df_list], axis=1)
        df_close_price = pd.concat([df['close_price'] for df in df_list], axis=1)
        df_vwap = pd.concat([df['vwap'] for df in df_list], axis=1)
        df_amount = pd.concat([df['amount'] for df in df_list], axis=1)
        df_open_price = df_open_price.sort_index(ascending=True)
        time_arr = pd.to_datetime(df_open_price.index, unit='ms').values
        open_price_arr = df_open_price.values.astype(float)
        high_price_arr = df_high_price.sort_index(ascending=True).values.astype(float)
        low_price_arr = df_low_price.sort_index(ascending=True).values.astype(float)
        close_price_arr = df_close_price.sort_index(ascending=True).values.astype(float)
        vwap_arr = df_vwap.sort_index(ascending=True).values.astype(float)
        amount_arr = df_amount.sort_index(ascending=True).values.astype(float)
        return all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr

    def weighted_spearmanr(self, y_true, y_pred):
        n = len(y_true)
        if n < 2: return 0.0
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x ** 2
        w_sum = w.sum()
        if w_sum == 0: return 0.0
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true) ** 2).sum()
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()
        if var_true == 0 or var_pred == 0: return 0.0
        return cov / np.sqrt(var_true * var_pred)

    def train(self, df_target, df_factor1, df_factor2, df_factor3, df_factor4, df_factor5, batch_results=None, save_result=False):
        t0 = time.time()
        factor1_long = df_factor1.stack()
        factor2_long = df_factor2.stack()
        factor3_long = df_factor3.stack()
        factor4_long = df_factor4.stack()
        factor5_long = df_factor5.stack()
        target_long = df_target.stack()
        data = pd.concat([
            factor1_long.rename('factor1'),
            factor2_long.rename('factor2'),
            factor3_long.rename('factor3'),
            factor4_long.rename('factor4'),
            factor5_long.rename('factor5'),
            target_long.rename('target')
        ], axis=1).dropna()
        if data.empty: return None
        X = data[['factor1', 'factor2', 'factor3', 'factor4', 'factor5']]
        y = data['target']
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        model.fit(X, y)
        data['y_pred'] = model.predict(X)
        importance = model.feature_importances_
        print(f"[train] 特征重要性: {dict(zip(X.columns, importance))}")
        print(f"[train] BTC市场因子重要性: {importance[3]:.4f}")
        print(f"[train] 新增主动买入因子重要性 (factor5 - buy_ratio): {importance[4]:.4f}")
        print(f"[train] Weighted Spearman: {self.weighted_spearmanr(data['target'], data['y_pred']):.4f} | 批耗时: {time.time() - t0:.1f}s")
        self.all_train_data.append(data[['target', 'y_pred']])
        df_submit = data.reset_index().rename(columns={'level_0': 'datetime', 'level_1': 'symbol'})
        df_submit['datetime'] = pd.to_datetime(df_submit['datetime'])
        df_submit['id'] = df_submit['datetime'].astype(str) + "_" + df_submit['symbol']
        df_submit = df_submit[['id', 'y_pred']].rename(columns={'y_pred': 'predict_return'})
        if batch_results is not None:
            batch_results.append(df_submit)
        if save_result:
            try:
                df_submission_id = pd.read_csv(self.submission_id_path)
                id_list = df_submission_id["id"].tolist()
                df_submit_competion = df_submit[df_submit['id'].isin(id_list)].copy()
                missing = list(set(id_list) - set(df_submit_competion['id']))
                df_missing = pd.DataFrame({'id': missing, 'predict_return': [0] * len(missing)})
                df_submit_competion = pd.concat([df_submit_competion, df_missing], ignore_index=True)
                df_submit_competion = df_submit_competion.set_index('id').loc[id_list].reset_index()
                df_submit_competion.to_csv("submit.csv", index=False)
                print("[train] Submit file saved as 'submit.csv'.")
            except Exception as e:
                print(f"Error saving submit file: {e}")
        return df_submit

    def run(self):
        print("[run] 开始读取K线数据...")
        all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr = self.get_all_symbol_kline()
        if not all_symbol_list:
            print("[run] No data loaded. Exiting run().")
            return
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        df_close = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr)

        # 读取 buy_volume 数据
        df_buy_volume_list = []
        for symbol in all_symbol_list:
            try:
                df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
                df = df.set_index("timestamp").sort_index().astype(np.float64)
                df_buy_volume_list.append(df["buy_volume"].rename(symbol))
            except Exception as e:
                print(f"[run] 读取 buy_volume 失败: {symbol}, 错误: {e}")
                df_buy_volume_list.append(pd.Series(dtype=float, name=symbol))
        df_buy_volume = pd.concat(df_buy_volume_list, axis=1)
        df_buy_volume.index = pd.to_datetime(df_buy_volume.index, unit='ms')

        df_vwap.index = df_amount.index = df_close.index = pd.to_datetime(df_vwap.index)

        print("[run] 数据加载完毕，开始训练...")
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        batch_size = 10
        time_index = df_vwap.index
        months = sorted(set([(d.year, d.month) for d in time_index]))
        batch_results = []

        for i in range(0, len(all_symbol_list), batch_size):
            batch_symbols = all_symbol_list[i:i+batch_size]
            print(f"[run] 批次: {i//batch_size+1}/{len(all_symbol_list)//batch_size+1}")
            df_vwap_batch = df_vwap[batch_symbols]
            df_amount_batch = df_amount[batch_symbols]
            df_close_batch = df_close[batch_symbols]
            df_buy_volume_batch = df_buy_volume[batch_symbols]
            for year, month in months:
                mask = (time_index.year == year) & (time_index.month == month)
                if not mask.any(): continue
                df_vwap_month = df_vwap_batch.loc[mask]
                df_amount_month = df_amount_batch.loc[mask]
                df_close_month = df_close_batch.loc[mask]
                df_buy_volume_month = df_buy_volume_batch.loc[mask]

                btc_7d_return = df_close['BTCUSDT'].loc[mask] / df_close['BTCUSDT'].loc[mask].shift(windows_7d) - 1
                df_btc_7d_return = pd.DataFrame(np.tile(btc_7d_return.values[:, None], (1, len(batch_symbols))),
                                                index=btc_7d_return.index, columns=batch_symbols)

                df_24hour_rtn = df_vwap_month / df_vwap_month.shift(windows_1d) - 1
                df_15min_rtn = df_vwap_month / df_vwap_month.shift(1) - 1
                df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
                df_7d_momentum = df_vwap_month / df_vwap_month.shift(windows_7d) - 1
                df_amount_sum = df_amount_month.rolling(windows_7d).sum()

                df_buy_ratio = df_buy_volume_month / df_amount_month.replace(0, np.nan)
                print(f"[run] buy_ratio月度统计: 均值={df_buy_ratio.mean().mean():.4e}, 方差={df_buy_ratio.var().mean():.4e}")

                self.train(
                    df_target=df_24hour_rtn.shift(-windows_1d),
                    df_factor1=df_7d_volatility,
                    df_factor2=df_7d_momentum,
                    df_factor3=df_amount_sum,
                    df_factor4=df_btc_7d_return,
                    df_factor5=df_buy_ratio,
                    batch_results=batch_results,
                    save_result=False
                )

        if batch_results:
            df_submit_all = pd.concat(batch_results, ignore_index=True).drop_duplicates(subset=['id'], keep='last')
            df_submit_all.to_csv("submit_all.csv", index=False)
            print("[run] All batch results saved as 'submit_all.csv'.")
            df_submission_id = pd.read_csv(self.submission_id_path)
            id_list = df_submission_id["id"].tolist()
            df_submit_competion = df_submit_all[df_submit_all['id'].isin(id_list)].copy()
            missing = list(set(id_list) - set(df_submit_competion['id']))
            df_missing = pd.DataFrame({'id': missing, 'predict_return': [0] * len(missing)})
            df_submit_competion = pd.concat([df_submit_competion, df_missing], ignore_index=True)
            df_submit_competion = df_submit_competion.set_index('id').loc[id_list].reset_index()
            df_submit_competion.to_csv("/kaggle/working/submission.csv", index=False)
            print(f"[run] Final submission shape: {df_submit_competion.shape} (expected: {len(id_list)})")
        else:
            print("[run] No batch results generated.")

        if self.all_train_data:
            all_data = pd.concat(self.all_train_data, ignore_index=True)
            score = self.weighted_spearmanr(all_data['target'], all_data['y_pred'])
            print(f'[run] 全局Weighted Spearman correlation coefficient: {score:.4f}')


if __name__ == '__main__':
    model = OlsModel()
    model.run()
