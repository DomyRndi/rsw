import os
import numpy as np
import jieba as jb
import jieba.analyse
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

number_to_author = ['LX', 'MY', 'QZS', 'WXB', 'ZAL']  # 作家集合
author_number = len(number_to_author)  # 作家数量
author_to_number = {author: i for i, author in enumerate(number_to_author)}  # 建立作家数字映射，从0开始

# 读入数据集
data_begin = []  # 初始数据集合
path = 'dataset/'  # 数据路径
#path = 'test_data/test_case1_data/'  # 数据路径
for file in os.listdir(path):
    if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
        with open(os.path.join(path, file), 'r',  encoding='UTF-8') as f:  # 打开文件
            for line in f.readlines():
                data_begin.append((line, author_to_number[file[:-4]]))

# 将片段组合在一起后进行词频统计
fragment = ['' for _ in range(author_number)]
for sentence, label in data_begin:
    fragment[label] += sentence   # 每个作家的所有作品组合到一起

# 词频特征统计，取出各个作家前 200 的词
high_words = set()
for label, text in enumerate(fragment):  # 提取每个作家频率前200的词汇，不返回关键词权重值
    for word in jb.analyse.extract_tags(text, topK=500, withWeight=False):
        if word in high_words:
            high_words.remove(word)
        else:
            high_words.add(word) # 将高频词汇存入

number_to_word = list(high_words) 
word_number = len(number_to_word)  # 所有高频词汇的个数
word_to_number = {word: i for i, word in enumerate(number_to_word)} # 建立高频词汇字典，一一对应

features = torch.zeros((len(data_begin), word_number))
labels = torch.zeros(len(data_begin))
for i, (sentence, author_belong) in enumerate(data_begin):
    feature = torch.zeros(word_number, dtype=torch.float)
    for word in jb.lcut(sentence):  # jb.lcut 直接生成的就是一个list，尝试jb.lcut_for_search()搜索引擎模式和jb.lcut精确分词模式
        if word in high_words:
            feature[word_to_number[word]] += 1
    if feature.sum():
        feature /= feature.sum()
        features[i] = feature
        labels[i] = author_belong
    else:
        labels[i] = 5  # 表示识别不了作者

dataset = data.TensorDataset(features, labels)

# 划分数据集
valid_weight = 0.25  # 30%验证集
train_size = int((1 - valid_weight) * len(dataset))  # 训练集大小
valid_size = len(dataset) - train_size  # 验证集大小
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
# 创建一个 DataLoader 对象
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # batch_size=32
valid_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=True)  # batch_size=1000

# 设定模型参数，使用ReLU作为激活函数，简单顺序连接模型
model = nn.Sequential(
    nn.Linear(word_number, 700),  # 三个隐含层神经网络，尝试（512，1024，1024）
    nn.ReLU(),  # 激活函数尝试ReLU，
    nn.Linear(700, 6),
    # nn.ReLU(),
    # nn.Linear(1024, 6),
      # 最后一个隐含层不需要激活函数
).to(device)

epochs = 60  # 设定训练轮次
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数（尝试nn.CrossEntropyLoss()和nn.NLLLoss(),二者多用于多分类任务）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 定义优化器（adam），初始学习率为1e-4
best_acc = 0   # 优化器尝试RMSProp（）、Adam（）、Adamax（）
history_acc = []
history_loss = []
best_model = model.cpu().state_dict().copy()  # 最优模型

for epoch in range(epochs):  # 开始训练
    for step, (word_x, word_y) in enumerate(train_loader):
        word_x = word_x.to(device)  # 传递数据
        word_y = word_y.to(device)
        out = model(word_x)
        loss = loss_fn(out, word_y.long())  # 计算损失
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()
        train_acc = np.mean((torch.argmax(out, 1) == word_y).cpu().numpy())

        with torch.no_grad():  # 上下文管理器，被包裹语句不会被track
            for word_x, word_y in valid_loader:
                word_x = word_x.to(device)
                word_y = word_y.to(device)
                out = model(word_x)  
                valid_acc = np.mean((torch.argmax(out, 1) == word_y).cpu().numpy())  # 准确率求平均
        if valid_acc > best_acc:  # 记录最佳模型
            best_acc = valid_acc
            best_model = model.cpu().state_dict().copy()
    print('epoch:%d | valid_acc:%.4f' % (epoch, valid_acc))   # 展示训练过程
    history_acc.append(valid_acc)
    history_loss.append(loss)

print('best accuracy:%.4f' % (best_acc, ))
torch.save({
    'word2int': word_to_number,
    'int2author': number_to_author,
    'model': best_model,
}, 'results/test1.pth')  # 保存模型

plt.plot(history_acc,label = 'valid_acc')
plt.plot(history_loss,label = 'valid_loss')
plt.legend()
plt.show()
