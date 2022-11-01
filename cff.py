import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

# 数据说明
dataset = pd.read_csv('train_dataset/Train.csv')
mirna_seqdf = pd.read_csv('train_dataset/mirna_seq.csv')  # ['mirna', 'seq']
gene_seqdf = pd.read_csv('train_dataset/gene_seq.csv')  # 'label', 'sequence'

dataset_mirna = dataset['miRNA']
dataset_gene = dataset['gene']
dataset_label = dataset['label']
gene_index = gene_seqdf['label'].values.tolist()
gene_seq = gene_seqdf['sequence']
mirna_index = mirna_seqdf['mirna'].values.tolist()
mirna_seq = mirna_seqdf['seq']

# 数据预处理
key_set = {}
key_set_T = {}
# itertools.product() 创建一个迭代器 repeat指定重复生成序列的顺序
for i in itertools.product('UCGA', repeat=3):  # UUU UUC UGA...
    # print(i)
    obj = ''.join(i)
    # print(obj)
    ky = {'{}'.format(obj): 0}  # {UUU:0}
    key_set.update(ky)
for i in itertools.product('TCGA', repeat=3):  # itertools.product('BCDEF', repeat = 2):
    # print(i)
    obj = ''.join(i)
    # print(obj)
    ky = {'{}'.format(obj): 0}
    key_set_T.update(ky)


def clean_key_set(key_set):
    for i, key in enumerate(key_set):
        # print(i,key,key_set[key])
        key_set[key] = 0
    return key_set


# 训练集mirna的特征
def return_features(n, seq):
    clean_key_set(key_set)
    if '\n' in seq:
        seq = seq[0:-1]
    for i in range(len(seq) + 1 - n):
        win = seq[i:i + n]
        # print(win)
        ori = key_set['{}'.format(win)]
        key_set['{}'.format(win)] = ori + 1
    return key_set


# 训练集gene的特征
def return_gene_features(n, seq):
    clean_key_set(key_set_T)
    if '\n' in seq:
        seq = seq[0:-1]
    for i in range(len(seq) + 1 - n):
        win = seq[i:i + n]
        # print(win)
        ori = key_set_T['{}'.format(win)]
        key_set_T['{}'.format(win)] = ori + 1
    return key_set_T


# 使用拼接方法构造数据集
def construct_dataset(dataset_mirna, dataset_gene):
    list_mirna_feature = []
    list_gene_feature = []
    for i in range(0, len(dataset_mirna)):
        try:
            mirna = dataset_mirna[i]
            m_index = mirna_index.index(mirna)  # 训练集的mirna在表中的位置
            # print(m_index)
            mirna_f = return_features(3, mirna_seq[m_index])
            gene = dataset_gene[i]
            g_index = gene_index.index(gene)

            gene_f = return_gene_features(3, gene_seq[g_index])
            # print(gene_f)
            mirna_feature = mirna_f.copy()
            gene_feature = gene_f.copy()
            list_mirna_feature.append(mirna_feature)
            list_gene_feature.append(gene_feature)
        except:
            mirna = dataset_mirna[i]
            gene = dataset_gene[i]
            print('error detected', i, mirna, gene)
    lmpd = pd.DataFrame(list_mirna_feature)
    lgpd = pd.DataFrame(list_gene_feature)
    X = pd.concat([lmpd, lgpd], axis=1)  # 横向拼接
    return X


Y = []
for i, label in enumerate(dataset_label):
    if label == 'Functional MTI':
        Y.append(1)
    else:
        Y.append(0)

X = construct_dataset(dataset_mirna, dataset_gene)

# 模型训练切分训练集调参
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=2)

def train():
    # n_estimators:决策树的个数。
    clf = RandomForestClassifier(n_estimators=75)
    clf.fit(X_train, y_train)
    y_p = clf.predict(X_test)  # 返回对测试机的预测标签
    # acc = metrics.accuracy_score(y_test,y_p)
    # print('RF_ACC',acc)
    y_pb = clf.predict_proba(X_test)
    f1score = metrics.f1_score(y_test, y_p)
    print('RF_F1', f1score)


train()

# 最终模型
clf_final = RandomForestClassifier(n_estimators=75)
clf_final.fit(X, Y)

# 存储模型与重新调用
joblib.dump(clf_final, "model/train_model.m")  # 存储
clf_final = joblib.load("model/train_model.m")  # 调用

# 加载测试数据
df_predict = pd.read_csv('test_dataset.csv')

predict_mirna = df_predict['miRNA']
predict_gene = df_predict['gene']

X_predict = construct_dataset(predict_mirna, predict_gene)

final_result = clf_final.predict(X_predict)

# 生成提交文件
df_predict['results'] = final_result
df_predict.to_csv('submission.csv', index=None)
