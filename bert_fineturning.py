import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import pickle
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from tqdm import tqdm_notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.metrics import precision_recall_curve,classification_report
import matplotlib.pyplot as plt


# pandas读取数据
data = pd.read_csv('tbrain_train_final_0610_0723.csv')
# 列名重新命名
#content = data.columns['content','full_content']
#data = data[['text','y']]
le = LabelEncoder()
data = data[['news_ID','full_content']]
#data.columns = ['news_ID','full_content']
le.fit(data.full_content.tolist())
data['full_content'] = le.transform(data.full_content.tolist())
#data["full_content"] = data["full_content"].apply(lambda x: list(map(int, x)))
#print(data.content.head())
print(data.head())

# 分词工具
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

class DataPrecessForSingleSentence(object):
    """
        对文本进行处理
        """

    def __init__(self, bert_tokenizer, max_workers=10):
        """
        bert_tokenizer :分词器
        dataset        :包含列名为'text'与'label'的pandas dataframe
        """
        self.bert_tokenizer = bert_tokenizer
        # 创建多线程池
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        # 获取文本与标签

    def get_input(self, dataset, max_seq_len=30):
        """
        通过多线程（因为notebook中多进程使用存在一些问题）的方式对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。

        入参:
            dataset     : pandas的dataframe格式，包含两列，第一列为文本，第二列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。


        """

        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                      [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, labels

    def trunate_and_pad(self, seq, max_seq_len):
        """
        1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。

        入参:
            seq         : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。

        """
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment
# 类初始化
processor = DataPrecessForSingleSentence(bert_tokenizer= bert_tokenizer)
# 产生输入ju 数据
seqs, seq_masks, seq_segments, labels = processor.get_input(
    dataset=data, max_seq_len=30)
# 加载预训练的bert模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese', num_labels=28)
# 转换为torch tensor
t_seqs = torch.tensor(seqs, dtype=torch.long)
t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
t_labels = torch.tensor(labels, dtype = torch.long)

train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
train_sampler = RandomSampler(train_data)
train_dataloder = DataLoader(dataset= train_data, sampler= train_sampler,batch_size = 256)
# 将模型转换为trin mode
model.train()


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 待优化的参数
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            0.01
    },
    {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }
]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-05,
                     warmup= 0.1 ,
                     t_total= 2000)

device = 'cpu'

##
# from tqdm.notebook import tqdm
# pbar = tqdm(filenames, disable=False)
# 存储每一个batch的loss
loss_collect = []
# for i in trange(10, desc='Epoch'):
#     for step, batch_data in enumerate(
#             tqdm_notebook(train_dataloder, desc='Iteration')):
#         #tqdm.notebook.tqdm(train_dataloder, desc='Iteration')):
#         batch_data = tuple(t.to(device) for t in batch_data)
#         batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
#         # 对标签进行onehot编码
#         one_hot = torch.zeros(batch_labels.size(0), 28).long()
#         one_hot_batch_labels = one_hot.scatter_(
#             dim=1,
#             index=torch.unsqueeze(batch_labels, dim=1),
#             src=torch.ones(batch_labels.size(0), 28).long())
#
#         logits = model(
#             batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
#         logits = logits.softmax(dim=1)
#         loss_function = CrossEntropyLoss()
#         loss = loss_function(logits, batch_labels)
#         loss.backward()
#         loss_collect.append(loss.item())
#         print("\r%f" % loss, end='')
#         optimizer.step()
#         optimizer.zero_grad()
#torch.save(model,open("fine_tuned_chinese_bert.bin","wb"))
### 推論

# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
le_test = LabelEncoder()

test_data = pd.read_csv('tbrain_train_final_0610_0723.csv')
#test_data.columns = ['text','label']
# 标签ID化
#test_data['label'] = test_data.label.tolist()
test_data['full_content'] = le_test.fit(data.full_content.tolist()).transform(test_data.full_content.tolist())
# 转换为tensor
test_seqs, test_seq_masks, test_seq_segments, test_labels = processor.get_input(
    dataset=test_data, max_seq_len=30)
test_seqs = torch.tensor(test_seqs, dtype=torch.long)
test_seq_masks = torch.tensor(test_seq_masks, dtype = torch.long)
test_seq_segments = torch.tensor(test_seq_segments, dtype = torch.long)
test_labels = torch.tensor(test_labels, dtype = torch.long)
test_data = TensorDataset(test_seqs, test_seq_masks, test_seq_segments, test_labels)
test_dataloder = DataLoader(dataset= train_data, batch_size = 256)
# 用于存储预测标签与真实标签
true_labels = []
pred_labels = []
#model.eval()
# 预测
for batch_data in tqdm_notebook(test_dataloder, desc = 'TEST'):
    batch_data = tuple(t.to(device) for t in batch_data)
    batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
    #logits = model(test_data,labels=None)
    logits = model(batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
    logits = logits.softmax(dim=1).argmax(dim = 1)
    pred_labels.append(logits.detach().numpy())
    true_labels.append(batch_labels.detach().numpy())

# 預測值
#np.concatenate(pred_labels)
#print(np.concatenate(true_labels))
#print(np.concatenate(pred_labels))
# 查看各个类别的准召
print(classification_report(np.concatenate(true_labels), np.concatenate(pred_labels)))
