from collections import defaultdict
from ngram_language_model import MyLanguageModel
import copy


class SentenceCorrection():
    def __init__(self, corpus, sub_corpus_path):
        self.lm = MyLanguageModel(corpus)
        self.sub_corpus = dict()
        self.threshold = 10
        self.setup_sub_corpus(sub_corpus_path)

    def setup_sub_corpus(self, sub_corpus_path):
        sub_corpus = open(sub_corpus_path, 'r', encoding='utf-8').readlines()
        for sub_c in sub_corpus:
            keys, values = sub_c.split()
            self.sub_corpus[keys] = list(values)

    def correct(self, sentence):
        self.baseline_prob = self.lm.predict(sentence)
        print('baseline,', self.baseline_prob, sentence)
        sentence_list = list(sentence)
        new_sentence = copy.deepcopy(sentence_list)
        for index, word in enumerate(sentence_list):
            tongyin_list = self.sub_corpus.get(word, [])
            if tongyin_list == []: continue
            result = self.cal_correct_prob(copy.deepcopy(sentence_list), tongyin_list, index)

            if max(result) > self.threshold:
                fix_word = tongyin_list[result.index(max(result))]
                print('第 %s 个字建议修改：'%(index),word,'->',fix_word,'概率提升：',max(result))

                new_sentence[index]= fix_word
        print('修改前：',''.join(sentence_list))
        print('修改前：',''.join(new_sentence))

    def cal_correct_prob(self, sentence_list, tongyin_list, index):
        # print(sentence_list, tongyin_list, index)
        if tongyin_list is []:
            return [-1]
        result = []
        for tyc in tongyin_list:
            sentence_list[index] = tyc
            after_prob = self.lm.predict(''.join(sentence_list))
            # print('after', after_prob)
            after_prob -= self.baseline_prob
            result.append(after_prob)
        return result


if __name__ == '__main__':
    corpus = open('财经.txt', 'r', encoding='utf-8').readlines()
    SentenceCorrection(corpus, 'tongyin.txt').correct('因为其他频种都遭宇了政策调控')
