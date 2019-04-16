#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
from collections import defaultdict

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:

    def word_dictionary(self, corpus):
        word_dict=defaultdict(int)
        for s in corpus:
            for w in s:
                word_dict[w]+=1
        self.word_dict = word_dict


    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        self.word_dictionary(corpus)
        num_words = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
        self.num_words = num_words
        for s in corpus:
            self.fit_sentence(s)
        #self.norm()
        #print(self.model)



    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)

        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        #print(p)
        return p

    def logprob_sentence_(self, sentence):
        #print('in logprob_sentence')

        p = 0.0
        p += self.cond_logprob(sentence[0],'*','*')
        if len(sentence)>1:
            p += self.cond_logprob(sentence[1],sentence[0],'*')
        for i in range(2,len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[i-1], sentence[i-2])
        p += self.cond_logprob('END_OF_SENTENCE', sentence[len(sentence)-1], sentence[len(sentence)-2])
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

class Trigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = defaultdict(float)
        self.lbackoff = log(backoff, 2)
        self.tot = 0.0
        self.num_words = 0
        self.word_dict = dict()

    def total(self):
        tot = 0.0
        for word in self.model:
            if isinstance(word,str):
                tot += self.model[word]
        self.tot = tot
        return tot


    def inc_word(self, w, prev1, prev2):
        if (w,prev1,prev2) in self.model:
            self.model[(w,prev1,prev2)] += 1.0
        else:
            self.model[(w,prev1,prev2)] = 1.0

        if (prev1,prev2) in self.model:
            self.model[(prev1,prev2)] += 1.0
        else:
            self.model[(prev1,prev2)] = 1.0

        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0


    def fit_sentence(self, sentence):
        for j in range(len(sentence)):
            if self.word_dict[sentence[j]]<5:
                sentence[j] = 'UNK'
        self.inc_word(sentence[0],'*','*')
        if len(sentence)>1:
            self.inc_word(sentence[1],sentence[0],'*')
        for i in range(2,len(sentence)):
            self.inc_word(sentence[i],sentence[i-1],sentence[i-2])
        self.inc_word('END_OF_SENTENCE',sentence[len(sentence)-1],sentence[len(sentence)-2])

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            if isinstance(word,str):
                tot += self.model[word]
        self.tot = tot
        #print('total: ',tot)
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) 

    def cond_logprob(self, word, sentence, unk_flag = 0):
        if unk_flag ==0:
            for j in range(len(sentence)):
                if self.word_dict[sentence[j]]<5:
                    sentence[j] = 'UNK'
        if len(sentence)==0:
            prev1 = '*'
            prev2 = '*'
        elif len(sentence)==1:
            prev1 = sentence[0]
            prev2 = '*'
        else:
            prev1 = sentence[-1]
            prev2 = sentence[-2]
        alpha = 0.005
        lamb1=0.6
        lamb2=0.2
        lamb3 = 1.0 - (lamb1 + lamb2)
        self.model['*'] = self.model['END_OF_SENTENCE']
        if self.model[(word,prev1,prev2)]==0:
            triplet_component = 0
        else: 
            triplet_component = lamb1*(self.model[(word,prev1,prev2)]/self.model[(prev1,prev2)])
        if self.model[(word,prev1)]==0:
            tuple_component = 0
        else: 
            #print('one:  ',self.model[prev1], prev1)
            #print('two: ',self.model[(word,prev1)], word)
            tuple_component = lamb2*(self.model[(word,prev1)]/self.model[prev1])
        if self.model[word]==0:
            uni_component = 0
        else: 
            uni_component = lamb3*(self.model[word]/len(self.word_dict))
        if triplet_component + tuple_component + uni_component != 0:
            return log( triplet_component + tuple_component + uni_component  ,2)
        else:
            return self.lbackoff

        '''
        if (word,prev1,prev2) in self.model:
            #return log( (self.model[(word,prev1,prev2)])/( self.model[(prev1,prev2)] ), 2). #no smoothing
            #return log( lamb1*(self.model[(word,prev1,prev2)]/self.model[(prev1,prev2)]) + lamb2*(self.model[(word,prev1)]/self.model[prev1]) + lamb3*(self.model[word]/len(self.word_dict))   ,2)
            #return log( (self.model[(word,prev1,prev2)]+1)/( self.model[(prev1,prev2)] + alpha*len(self.word_dict) ), 2)
        elif (prev1, prev2) in self.model:
            #return log(1.0/( self.model[(prev1,prev2)]), 2)
            return log(1.0/( self.model[(prev1,prev2)] + alpha*len(self.word_dict)), 2)
        else:
            #return self.lbackoff
            return log( 1.0/(alpha*len(self.word_dict) ) , 2)
        '''


    def cond_logprob_(self, word, prev1, prev2):
        #print(self.tot)
        if (word,prev1,prev2) in self.model:
            return (self.model[(word,prev1,prev2)]+1)/(self.model[(prev1,prev2)]+self.tot)
        else:
            return self.lbackoff

    def vocab(self):
        #vocab = []
        #for word in self.model:
        #    if isinstance(word,str):
        #        vocab.append(word)
        return self.word_dict.keys()

