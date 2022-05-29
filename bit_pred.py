import pandas as pd

### 代码开始 ### (≈ 2 行代码)
df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/1081/challenge-2-bitcoin.csv")
df.head()
### 代码结束 ###

### 代码开始 ### (≈ 1 行代码)
data = df[['btc_market_price','btc_total_bitcoins','btc_transaction_fees']]
### 代码结束 ###

data.head()

from matplotlib import pyplot as plt
# %matplotlib inline


# x_temp = np.linspace(0, 3000, 10000)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

### 代码开始 ### (≈ 9 行代码)
# axes[0, 0].plot(x, fit_func(n_poly(4), x), 'g')
# axes[0, 0].scatter(x, y)
axes[0].plot([i for i in range(0,len(data))],data['btc_market_price'], color='green')
axes[0].set_ylabel("btc_market_price")
axes[0].set_xlabel("time")

# axes[0, 1].plot(x_temp, fit_func(n_poly(2), x_temp), 'b')
axes[1].plot([i for i in range(0,len(data))],data['btc_total_bitcoins'], color='blue')
# axes[1].scatter(x, y)
axes[1].set_xlabel("btc_total_bitcoins")
axes[1].set_xlabel("time")

# axes[0, 2].plot(x_temp, fit_func(n_poly(3), x_temp), 'r')
#axes[0, 2].scatter(x, y)
axes[2].plot([i for i in range(0,len(data))],data['btc_transaction_fees'], color='red')
axes[2].set_xlabel("btc_transaction_fees")
axes[2].set_xlabel("time")

### 代码结束 ###

def split_dataset():
    """
    参数:
    无

    返回:
    X_train, y_train, X_test, y_test -- 训练集特征、训练集目标、测试集特征、测试集目标
    """

    ### 代码开始 ### (≈ 6 行代码)
    train_df = df[:int(len(df) * 0.7)]
    test_df = df[int(len(df) * 0.7):]

    X_train = train_df[['btc_market_price', 'btc_total_bitcoins']].values
    y_train = train_df['btc_transaction_fees']
    X_test = test_df[['btc_market_price', 'btc_total_bitcoins']].values
    y_test = test_df['btc_transaction_fees']

    ### 代码结束 ###

    return X_train, y_train, X_test, y_test

    print(len(split_dataset()[0]),
          len(split_dataset()[1]),
          len(split_dataset()[2]),
          len(split_dataset()[3]),
          split_dataset()[0].shape,
          split_dataset()[1].shape,
          split_dataset()[2].shape,
          split_dataset()[3].shape)


# 加载数据
X_train = split_dataset()[0]
y_train = split_dataset()[1]
X_test = split_dataset()[2]
y_test = split_dataset()[3]


def poly3():
    """
    参数:
    无

    返回:
    mae -- 预测结果的 MAE 评价指标
    """

    ### 代码开始 ### (≈ 7 行代码)
    poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
    poly_X_train_3 = poly_features_3.fit_transform(X_train)
    poly_X_test_3 = poly_features_3.fit_transform(X_test)

    model = LinearRegression()
    model.fit(poly_X_train_3, y_train)  # 训练模型

    results_3 = model.predict(poly_X_test_3)  # 预测结果

    # results_3.flatten()  # 打印扁平化后的预测结果
    mae = mean_absolute_error(y_test, results_3.flatten())

    ### 代码结束 ###

    return mae


poly3()

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


def poly_plot(N):
    """
    参数:
    N -- 标量, 多项式次数

    返回:
    mse -- N 次多项式预测结果的 MSE 评价指标列表
    """

    m = 1
    mse = []
    m_max = 10

    ### 代码开始 ### (≈ 6 行代码)
    while m <= m_max:
        model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
        model.fit(X_train, y_train)  # 训练模型
        pre_y = model.predict(X_test)  # 测试模型
        mse.append(mean_squared_error(y_test, pre_y.flatten()))  # 计算 MSE
        m = m + 1

    ### 代码结束 ###

    return mse

poly_plot(10)[:10:3]


mse = poly_plot(10)

### 代码开始 ### (≈ 2 行代码)
plt.plot([i for i in range(1, 11)], mse, 'r')
plt.scatter([i for i in range(1, 11)], mse)
### 代码结束 ###

plt.title("MSE")
plt.xlabel("N")
plt.ylabel("MSE")

