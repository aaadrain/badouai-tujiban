import math
from collections import defaultdict


class MyLanguageModel():
    def __init__(self, corpus, n=3):
        self.n = n
        self.unk_prob = 1e-5
        self.back_off_prob = 0.4
        self.sos = '<sos>'
        self.eos = '<eos>'
        self.sep = '◕'
        self.ngram_count_dict = dict((x + 1, defaultdict(int)) for x in range(self.n))
        self.ngram_count_prob_dict = dict((x + 1, defaultdict(int)) for x in range(self.n))
        self.ngram_count(corpus)
        self.ngram_count_prob()

    def sentence_segment(self, sentence):
        return list(sentence)

    def ngram_count(self, corpus):
        for sentence in corpus:
            ngram_list = self.sentence_segment(sentence)
            ngram_lists = [self.sos] + ngram_list + [self.eos]
            for window_size in range(1, self.n + 1):
                for index, word in enumerate(ngram_lists):
                    if window_size != len(ngram_lists[index:index + window_size]):
                        # print(ngram_lists[index:index+window_size], window_size)
                        continue
                    ngram = self.sep.join(ngram_lists[index:index + window_size])
                    self.ngram_count_dict[window_size][ngram] += 1
            self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        # print(self.ngram_count_dict)

    def ngram_count_prob(self):
        for window_size in range(1, self.n + 1):
            for word, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    ngrams = word.split(self.sep)
                    pre_ngrams = self.sep.join(ngrams[:-1])
                    pre_fix_count = self.ngram_count_dict[window_size - 1][pre_ngrams]
                else:
                    pre_fix_count = self.ngram_count_dict[0]
                self.ngram_count_prob_dict[window_size][word] = count / pre_fix_count

    def get_gram_prob(self, gram):
        # print(gram)
        n = len(gram.split(self.sep))
        if gram in self.ngram_count_prob_dict[n]:
            return self.ngram_count_prob_dict[n][gram]
        elif len(gram):
            return self.unk_prob
        else:
            gram = self.sep.join(gram.split(self.sep)[1:])
            return self.back_off_prob * self.get_gram_prob(gram)

    def predict(self, sentence):
        sentence = self.sentence_segment(sentence)
        sentence_list = [self.sos] + sentence + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(sentence_list):
            gram = self.sep.join(sentence_list[max(0, index - self.n + 1):index + 1])

            prob = self.get_gram_prob(gram)
            # print(prob)
            sentence_prob += math.log(prob)
        return sentence_prob


if __name__ == '__main__':
    corpus = open('财经.txt', 'r', encoding='utf-8').readlines()

    print(MyLanguageModel(corpus).predict('因为其他频率种都遭宇了政策调控'))
