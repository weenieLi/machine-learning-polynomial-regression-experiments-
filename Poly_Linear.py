import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


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



from sklearn.pipeline import make_pipeline

X_train = X_train.reshape(len(X_train), 1)
X_test = X_test.reshape(len(X_test), 1)
y_train = y_train.reshape(len(y_train), 1)

for m in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(
        m, include_bias=False), LinearRegression())
    model.fit(X_train, y_train)
    pre_y = model.predict(X_test)
    print("{} 次多项式回归平均绝对误差: ".format(m),
          mean_absolute_error(y_test, pre_y.flatten()))
    print("{} 次多项式均方误差: ".format(m), mean_squared_error(y_test, pre_y.flatten()))
    print("---")

'''
3 次多项式回归平均绝对误差:  4.5476919429640335
3 次多项式均方误差:  29.933056670525307
---
4 次多项式回归平均绝对误差:  4.424336223586132
4 次多项式均方误差:  29.02874897635671
---
5 次多项式回归平均绝对误差:  4.341616358034744
5 次多项式均方误差:  28.221927580869462
---

3，4，5 次多项式回归的结果均优于线性回归模型。所以，多项式回归还是有其优越性的
'''

# 计算 m 次多项式回归预测结果的 MSE 评价指标并绘图
mse = []  # 用于存储各最高次多项式 MSE 值
m = 1  # 初始 m 值
m_max = 10  # 设定最高次数
while m <= m_max:
    model = make_pipeline(PolynomialFeatures(
        m, include_bias=False), LinearRegression())
    model.fit(X_train, y_train)  # 训练模型
    pre_y = model.predict(X_test)  # 测试模型
    mse.append(mean_squared_error(y_test, pre_y.flatten()))  # 计算 MSE
    m = m + 1

print("MSE 计算结果: ", mse)
# 绘图
plt.plot([i for i in range(1, m_max + 1)], mse, 'r')
plt.scatter([i for i in range(1, m_max + 1)], mse)

# 绘制图名称等
plt.title("MSE of m degree of polynomial regression")
plt.xlabel("m")
plt.ylabel("MSE")




# 加载必要模块
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error




