# import numpy as np
# import torch
# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification
# import os
# import jieba as jb
# import jieba.analyse
# import torch.nn as nn
# from torch.utils import data
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# from torch.utils.data import TensorDataset, DataLoader, random_split
# from transformers import BertForSequenceClassification, AdamW
# from transformers import get_linear_schedule_with_warmup



# model_path = './results/'


# def predict(text):
#     Tokenizer = BertTokenizer.from_pretrained(model_path)
#     model = BertForSequenceClassification.from_pretrained(model_path)
#     text_list = []
#     labels = []
#     text_list.append(text)
#     label = 0
#     labels.append(label)
#     tokenizer = Tokenizer(
#         text_list,
#         padding=True,
#         truncation=True,
#         max_length=128,
#         return_tensors='pt'  # 返回的类型为 pytorch tensor
#     )
#     input_ids = tokenizer['input_ids']
#     token_type_ids = tokenizer['token_type_ids']
#     attention_mask = tokenizer['attention_mask']

#     # model = model.cuda()
#     model.eval()
#     preds = []
#     # for i, batch in enumerate(pred_dataloader):
#     with torch.no_grad():
#         outputs = model(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#             attention_mask=attention_mask
#         )

#     logits = outputs[0]
#     logits = logits.detach().cpu().numpy()
#     preds += list(np.argmax(logits, axis=1))
#     labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
#     prediction = labels[preds[0]]
#     return prediction


# if __name__ == '__main__':
#     target_text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
#             骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
#             立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
#             一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
#             小禽，他决不会飞鸣，也不会跳跃。"

#     print(predict(target_text))



'''
以下部分为pytorch部分的模型预测代码，如需使用请取消注释
'''
import torch
import torch.nn as nn
import jieba as jb


config_path = 'results/test1.pth'
config = torch.load(config_path)

word2int = config['word2int']
int2author = config['int2author']
word_num = len(word2int)
model = nn.Sequential(
    nn.Linear(word_num, 700),
    nn.ReLU(),
    # nn.Linear(512, 1024),
    # nn.ReLU(),
    # nn.Linear(1024, 1024),
    # nn.ReLU(),
    # nn.Linear(512, 1024),
    # nn.ReLU(),
    nn.Linear(700, 6),
)
# model = nn.Sequential(
#     nn.Linear(word_num, 512),  # 三个隐含层神经网络，尝试（512，1024，1024）
#     nn.ReLU(),  # 激活函数尝试ReLU，
#     nn.Linear(512, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 6),
#       # 最后一个隐含层不需要激活函数
# )
model.load_state_dict(config['model'])
int2author.append(int2author[0])


def predict(text):
    feature = torch.zeros(word_num)
    for word in jb.lcut(text):
        if word in word2int:
            feature[word2int[word]] += 1
    feature /= feature.sum()
    model.eval()
    out = model(feature.unsqueeze(dim=0))
    pred = torch.argmax(out, 1)[0]
    return int2author[pred]


# if __name__ == '__main__':
#     target_text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
#             骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
#             立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
#             一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
#             小禽，他决不会飞鸣，也不会跳跃。"

#     print(predict(target_text))
