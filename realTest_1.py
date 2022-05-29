import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
# 加载数据集
df = pd.read_csv(
    "https://labfile.oss.aliyuncs.com/courses/1081/course-6-vaccine.csv", header=0)
print(df)

# 定义 x, y 的取值
x = df['Year']
y = df['Values']
# 绘图
plt.plot(x, y, 'r')
plt.scatter(x, y)

# 首先划分 dateframe 为训练集和测试集
train_df = df[:int(len(df)*0.7)]
test_df = df[int(len(df)*0.7):]

# 定义训练和测试使用的自变量和因变量
X_train = train_df['Year'].values
y_train = train_df['Values'].values

X_test = test_df['Year'].values
y_test = test_df['Values'].values

# 建立线性回归模型
model = LinearRegression()
model.fit(X_train.reshape(len(X_train), 1), y_train.reshape(len(y_train), 1))
results = model.predict(X_test.reshape(len(X_test), 1))
print(results)  # 线性回归模型在测试集上的预测结果


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("线性回归平均绝对误差: ", mean_absolute_error(y_test, results.flatten()))
print("线性回归均方误差: ", mean_squared_error(y_test, results.flatten()))


from sklearn.preprocessing import PolynomialFeatures
# 2 次多项式回归特征矩阵
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_X_train_2 = poly_features_2.fit_transform(
    X_train.reshape(len(X_train), 1))
poly_X_test_2 = poly_features_2.fit_transform(X_test.reshape(len(X_test), 1))

# 2 次多项式回归模型训练与预测
model = LinearRegression()
model.fit(poly_X_train_2, y_train.reshape(len(X_train), 1))  # 训练模型

results_2 = model.predict(poly_X_test_2)  # 预测结果

results_2.flatten()  # 打印扁平化后的预测结果

print("2 次多项式回归平均绝对误差: ", mean_absolute_error(y_test, results_2.flatten()))
print("2 次多项式均方误差: ", mean_squared_error(y_test, results_2.flatten()))

# 线性回归模型的预测结果要优于 2 次多项式回归模型的预测结果。
'''
线性回归平均绝对误差:  6.011979515629853
线性回归均方误差:  43.53185829515393

2 次多项式回归平均绝对误差:  19.792070829567653
2 次多项式均方误差:  464.3290384751541
'''