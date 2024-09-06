import random
import jieba
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re, string
lab = {}
# 读取数据
def text_to_words(file_path):
    """
    将文本文件中的句子进行分词，并去除标点符号，返回分词后的句子列表和标签列表。

    参数:
    file_path (str): 文本文件的路径。

    返回:
    tuple: 包含两个列表的元组，第一个列表是分词后的句子列表，第二个列表是标签列表。
    """
    sentences_arr = []  # 分词后的句子列表
    lab_arr = []  # 标签列表
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            x = line.split('_!_')[1]
            # print(x)
            lab[int(x)] = line.split('_!_')[2]
            lab_arr.append(x)  # 获取标签
            sentence = line.split('_!_')[-1].strip()  # 获取句子并去除前后空白字符
            # 去除标点符号
            sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+<|>|《》“”‘’…", "", sentence)
            sentence = jieba.lcut(sentence, cut_all=False)  # 精确模式分词

            sentences_arr.append(sentence)
        return sentences_arr, lab_arr

# 加载停用词表
def load_stopwords(file_path):
    """
    从文件中加载停用词表。

    参数:
    file_path (str): 停用词表文件的路径。

    返回:
    list: 包含停用词的列表。
    """
    stopwords = [line.strip() for line in open(file_path, 'r', encoding='utf-8').readlines()]
    return stopwords

# 词频统计
def get_dict(sentences_arr, stopwords):
    """
    统计句子列表中每个词语的词频，并返回按词频降序排序的词频字典。

    参数:
    sentences_arr (list): 包含分词后句子的列表。
    stopwords (list): 包含停用词的列表。

    返回:
    list: 包含按词频降序排序的词语及其词频的列表。
    """
    word_dic = {}
    for sentence in sentences_arr:
        for word in sentence:
            if word != '' and word.isalpha():
                if word not in stopwords:
                    word_dic[word] = word_dic.get(word, 1) + 1
    word_dic = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)
    return word_dic

# 构建词表, 过滤掉频率低于 word_num 的词
def get_feature_words(word_dic, word_num):
    """
    从词频字典中提取频率高于指定阈值的词语，构建特征词表。

    参数:
    word_dic (list): 包含按词频降序排序的词语及其词频的列表。
    word_num (int): 词频阈值，过滤掉频率低于该值的词语。

    返回:
    list: 包含特征词的列表。
    """
    n = 0
    feature_words = []
    for word in word_dic:
        if n < word_num:
            feature_words.append(word[0])
        n += 1
    return feature_words

# 文本特征表示
def get_text_features(train_data_list, test_data_list, feature_words):
    """
    将训练数据和测试数据转换为特征向量。

    参数:
    train_data_list (list): 训练数据列表。
    test_data_list (list): 测试数据列表。
    feature_words (list): 特征词列表。

    返回:
    tuple: 包含训练特征向量列表和测试特征向量列表的元组。
    """
    def text_features(text, feature_words):
        """
        将文本转换为特征向量。

        参数:
        text (list): 分词后的文本。
        feature_words (list): 特征词列表。

        返回:
        list: 特征向量。
        """
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_features_list = [text_features(text, feature_words) for text in train_data_list]
    test_features_list = [text_features(text, feature_words) for text in test_data_list]
    return train_features_list, test_features_list

def load_sentence(sentence):
    """去除标点符号"""
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+]|[+——！，。？、~@#￥%……&*（）]+<|>|【】《》“”‘’…", "", sentence)
    sentence = jieba.lcut(sentence, cut_all=False)  # 精确模式分词
    return sentence



# 读取数据并进行预处理
sentences_arr, lab_arr = text_to_words('./news_classify_data.txt')
stopwords = load_stopwords('./stopwords_cn.txt')
word_dic = get_dict(sentences_arr, stopwords)
feature_words = get_feature_words(word_dic, 10000)
# print(lab_arr)
# 划分训练集和测试集
train_data_list, test_data_list, train_class_list, test_class_list = model_selection.train_test_split(sentences_arr, lab_arr, test_size=0.1)

# 获取文本特征表示
train_features_list, test_features_list = get_text_features(train_data_list, test_data_list, feature_words)

# 训练朴素贝叶斯分类器
classifier = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
classifier.fit(train_features_list, train_class_list)

print(lab)
# 评估分类器性能
test_accuracy = classifier.score(test_features_list, test_class_list)
print("测试集准确率:", test_accuracy)

# 进行预测并输出分类报告
predict = classifier.predict(test_features_list)
print("分类报告:\n", classification_report(test_class_list, predict))
# print(lab_arr)
# lab = list(set(lab))
p_data = '【中国稳健前行】应对风险挑战必须发挥制度优势'
sentence = load_sentence(p_data)
sentence = [sentence]
print('分词后句子:',sentence)
#形成特征向量
p_words = get_text_features(sentence,sentence, feature_words)
res = classifier.predict(p_words[0])
print(lab[int(res)])



