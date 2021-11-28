# 最大正向切分的第一种实现方式
import re
import time


# 加载词典
def load_word_dict(path):
    # 计算词典中的 最大词长度
    max_word_length = 0
    word_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length


dict_tmp = ['北京', '大学生', '前来', '报道']

str1 = '北京大学生前来报到'


def foo1(string):
    words= []
    max_word_length =3
    while len(string)!=0:
        lens = min(max_word_length,len(string))
        word =string[:lens]
        while  word not in dict_tmp:
            if len(word) == 1:break
            word = word[:len(word)-1]
        words.append(word)
        string = string[len(word):]
    return words


print(foo1(str1))