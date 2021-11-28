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


# dict_tmp = ['北京', '大学生', '前来', '报道']
dict_tmp = {
    '北':0,
    '北京':1,
    '大学生':1,
    '大学':1,
    '大':0,
    '前':0,
    '前来':1,
    '报':0,
    '报道':1,

}

str1 = '我们北京大学生前来报到'


def foo2(string):
    words= []
    start_index ,end_index=0,1
    window = string[start_index:end_index]
    find_word = window
    while start_index<len(string):
        if window not in dict_tmp or end_index > len(string):
            words.append(find_word)
            start_index+=len(find_word)
            end_index=start_index+1
            window = string[start_index:end_index]
            find_word = window
        elif dict_tmp.get(window) ==1:
            find_word= window
            end_index +=1
            window = string[start_index:end_index]

        elif dict_tmp[window] ==0:
            end_index+=1
            window = string[start_index:end_index]
        print(window)
        print(words)
    return words


print(foo2(str1))