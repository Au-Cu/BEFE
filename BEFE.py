from datetime import datetime
import requests, json, time, re, warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 设定全局常变量
global targets, MT_features, MA_features
targets = []
start = '20241021'
MT_features = ['涨跌额', '涨跌幅']
MA_features = ['开盘', '收盘', '最低', '最高', '成交量(手)', '成交金额(万)', '换手率']
hold = ['买入', '持有']
empty = ['卖出', '空仓']


# 标的代码输入函数
def get_targets():
    print('请输入标的六位代码')
    while True:
        code = input('以换行继续输入下一个标的代码或结束\n')
        if code == '':
            break
        targets.append(code)
    print('标的代码获取完成')
    return


# 数据获取函数
def get_data(target):
    end = datetime.now().date().strftime('%Y%m%d')
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0'}
    colnames = '日期  开盘  收盘  涨跌额 涨跌幅 最低  最高  成交量(手)  成交金额(万) 换手率'.split()
    url = 'https://q.stock.sohu.com/hisHq?code=cn_' + target + '&start=' + start + '&end=' + end +'&stat=1&order=D&period=d'
    r = requests.get(url, headers = headers)
    try:
        data = json.loads(r.text)[0]['hq']
    except:
        print(json.loads(r.text))
        raise ValueError
    try:
        data = pd.DataFrame(data, columns = colnames) # 主板标的
    except:
        data = pd.DataFrame(data, columns = colnames + ['盘后量(手)']) # 创业板标的
    finally:
        data['日期'] = pd.to_datetime(data['日期'])
        data = data.sort_values('日期').reset_index(drop = True)
        time.sleep(0.1)
    return data


# 特征构造函数
def create_features(df):
    windows = range(1, 31)
    MT = {}
    MA = {}
    for window in windows:
        for feature in MT_features:
            MT[feature + '_MT' + str(window) + '_lag1'] = df[feature].rolling(window).sum()
        for feature in MA_features:
            MA[feature + '_MA' + str(window) + '_lag1'] = df[feature].rolling(window).mean()
        feature = '盘后量(手)'
        if feature in df.columns: # 创业板标的
            MA[feature + '_MA' + str(window) + '_lag1'] = df[feature].rolling(window).mean()
    return pd.concat([pd.DataFrame(MA), pd.DataFrame(MT)], axis=1).dropna()


# 主成分提取函数
def get_principal_component(df, feature, n = 4):
    # 数据处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # 建模降维
    model = PCA(n_components = n)
    principle_component = model.fit_transform(X_scaled)
    compressed_df = pd.DataFrame(principle_component,
                                 columns = [feature + 'PC_' + str(i + 1) for i in range(n)],
                                 index = df.index)
    return compressed_df


# 特征降维函数
def compress_features(df):
    compressed_df_list = []
    for feature in MT_features + MA_features:
        regular = f'(?=.*{re.escape(feature)})'
        cols = df.columns[df.columns.str.contains(regular, regex = True)].tolist()
        compressed_df = get_principal_component(df = df[cols], feature = feature)
        compressed_df_list.append(compressed_df)
    return pd.concat(compressed_df_list, axis = 1)
    

# 数据读取-数据清洗-特征构造-特征降维-对齐
def render_X_y(target):
    df = get_data(target)
    df[['涨跌幅', '换手率']] = df[['涨跌幅', '换手率']].apply(lambda x: x.apply(lambda y:float(y[:-2])))
    features = create_features(df = df)
    compressed_features = compress_features(features)
    today_features = compressed_features[-1:]
    X = compressed_features.shift(1).dropna()
    y = df['涨跌幅'][X.index]
    return today_features, X, y


# 数据分割函数
def split_train_test(X, y, cut):
    return X[:-cut], X[-cut:], y[:-cut], y[-cut:]


# 多模型集成预测——贝叶斯更新模型
def bagged_pred_bayes():
    cut = 40
    bayes_update_window = 20
    ensembled_pred_list = []

    get_targets()
    print('正在进行建模预测，请稍候\n')
    for target in targets:
        today_features, X, y = render_X_y(target)
        X_train, X_test, y_train, y_test = split_train_test(X, y, cut)
        models = {'Lasso': Lasso(alpha = 0.001),
                  'Ridge': Ridge(alpha = 1.0),
                  'RF': RandomForestRegressor(n_estimators = 200,
                                              max_depth = 5,
                                              random_state = 42),
                  'XGBoost': xgb.XGBRegressor(n_estimators=200,
                                              max_depth=3,
                                              learning_rate=0.05,
                                              subsample=0.8,
                                              colsample_bytree=0.8,
                                              random_state=42),
                  'LightGBM': lgb.LGBMRegressor(n_estimators=200,
                                                max_depth=3,
                                                learning_rate=0.05,
                                                random_state=42,
                                                verbose=-1)}
        for name, model in models.items():
            model.fit(X_train, y_train)
        weights = {name: 1 / len(models) for name in models}
        history_errors = {name: [] for name in models}
        for t in range(cut):
            X_temp = X_test.iloc[t:t + 1]
            y_temp = y_test.iloc[t]
            preds = {name: model.predict(X_temp)[0] for name, model in models.items()}
            for name in models:
                history_errors[name].append(y_temp - preds[name])
            if t >= bayes_update_window:
                likelihood = {}
                for name in models:
                    recent_errors = np.array(history_errors[name][-bayes_update_window:])
                    sigma2 = np.var(recent_errors) + 1e-6
                    likelihood[name] = np.exp(- (y_temp - preds[name]) ** 2 / (2 * sigma2))
                total = sum(likelihood[name] * weights[name] for name in models)
                weights = {name: likelihood[name] * weights[name] / total for name in models}
        final_preds = {}
        for name, model in models.items():
            final_preds[name] = model.predict(today_features)[0]
        ensembled_pred = sum(weights[name] * final_preds[name] for name in models)
        ensembled_pred_list.append(round(ensembled_pred, 2) / 100)
        print()
        print(f'标的代码：{target}')
        print('\n最终模型权重：')
        for name, weight in weights.items():
            print(f'{name}: {weight:.3f}')
        print('\n模型预测涨跌幅：')
        for name in models:
            print(f'{name}: {final_preds[name]:.2f}%')
        print('\n集成预测涨跌幅:', round(ensembled_pred, 2), '%')
        print()
    return ensembled_pred_list


# 主函数
bagged_pred_bayes()