import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model

#从文件导入数据
datafile = './housing.data'
housing_data = np.fromfile(datafile,sep=' ')
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_len = len(feature_names)
# print(housing_data.shape)
#将数据reshape成(m,n)的形式，m是样本数，n是特征数

housing_data = housing_data.reshape([housing_data.shape[0] // feature_len , feature_len])
# print(housing_data)

#定义归一化操作：取最大最小均值操作
feature_max = housing_data.max(axis=0)
feature_min = housing_data.min(axis=0)
feature_avg = housing_data.sum(axis=0) / housing_data.shape[0]

#归一化
def feature_norm(input):
    f_size = input.shape
    output_feature = np.zeros(f_size,np.float32)
    for batch_id in range(f_size[0]):
        for index in range(13):
            output_feature[batch_id][index] = (input[batch_id][index] - feature_avg[index]) / (feature_max[index] - feature_min[index])

    return output_feature

housing_feature = feature_norm(housing_data[:,:13])
# print(housing_feature)
housing_data = np.c_[housing_feature,housing_data[:,-1]].astype(np.float32)#拼接数据

ratio = 0.8 # 分割为训练集和测试集
offset = int(ratio * housing_data.shape[0])
train_data = housing_data[:offset]
test_data = housing_data[offset:]
print(train_data[:2])
#到此数据处理完毕

def Model():
    model = linear_model.LinearRegression()#使用该模型实现线性回归
    return model

#拟合函数
def train(model,x,y):
    model.fit(x,y)

x,y = train_data[:,:13],train_data[:,-1:]
model = Model()
train(model,x,y)

#模型评估
def draw_infer_result(ground_truths,infer_results):
    title = '房价'
    plt.title(title, fontsize=24)
    x = np.arange(1,40)
    y = x
    # 绘制x与y的关系线，这是一条通过原点的45度线，表示完美的预测
    plt.plot(x,y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    # 在图表上绘制散点图，表示真实值与预测值的关系
    # ground_truths是真实值数组，infer_results是预测值数组
    plt.scatter(ground_truths,infer_results,color = 'green',label = 'training cost')
    # 显示网格
    plt.grid()
    plt.show()

x_test,y_test = test_data[:,:13],test_data[:,-1:]
predict = model.predict(x_test)
draw_infer_result(y_test,predict)