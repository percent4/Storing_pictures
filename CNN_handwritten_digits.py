"""
数字识别
利用CNN对handwritten数字的数据集进行多分类
"""
from TF_CNN import CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_FILE_PATH = 'E://data/digits_4_clf.csv'          # CSV 文件路径
digits = pd.read_csv(CSV_FILE_PATH)       # 读取CSV文件

# 数据集的特征
features = ['v'+str(i+1) for i in range(256)]
y_bin_label = ['y'+str(i) for i in range(10)]

# 添加类别变量label
label = []
for i in range(digits.shape[0]):
    label.append(list(digits[y_bin_label].iloc[i,:]).index(1))

digits['label'] = label

# 数据是否标准化
# x_bar = (x-mean)/std
IS_STANDARD = 'no'
if IS_STANDARD == 'yes':
    for feature in features:
        mean = digits[feature].mean()
        std = digits[feature].std()
        digits[feature] = (digits[feature]-mean)/std

# 将数据集分为训练集和测试集，训练集70%, 测试集30%
x_train, x_test, y_train, y_test = train_test_split(digits[features], digits[y_bin_label], \
                                                    train_size = 0.99, test_size=0.01, random_state=1234)

# 使用CNN进行预测
# 构建CNN网络

# 模型保存地址
MODEL_SAVE_PATH = 'E://logs/cnn_digits.ckpt'
# CNN初始化
cnn = CNN(1000, 0.001, MODEL_SAVE_PATH)

# 训练CNN
cnn.train(x_train, y_train)
# 预测数据
y_pred = cnn.predict(x_test)

# 预测分类
prediction = []
for pred in y_pred:
    prediction.append(list(pred).index(max(pred)))

# 计算预测的准确率
x_test['prediction'] = prediction
x_test['label'] = digits['label'][y_test.index]
print(x_test.head())
accuracy = accuracy_score(x_test['prediction'], x_test['label'])
print('CNN的预测准确率为%.2f%%.'%(accuracy*100))
