import time
import logging
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from .word2vec_fast import train_sentence, train_tuple
from nlip import Embeddings
from nlip.utils import si, smart_open, Timer

floatX = np.float32

logging.basicConfig(level=logging.INFO,format="[%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

class Word2Vec():

    def __init__(self, dimension=100, alpha=0.025, window=5, negative=5,
            sample=1e-3, dev_data=None):
        self.word2index = {} # string -> word index
        self.index2word = [] # word index -> string
        self.index2count = [] # word index -> word count
        self.index2sample = [] # word index -> word sampling threshold
        self.ns_table = None
        self.dim = dimension
        self.alpha = alpha
        self.cur_alpha = alpha
        self.window = window
        self.negative = negative
        self.sample = sample
        self.dev = False
        if dev_data:
            self.dev_words = [(x,y) for x,y,*_ in dev_data]
            self.devsims = np.asarray([float(x) for _,_,x,*_ in dev_data])
            self.dev = True

    def _downsample_vocab(self):
        retain_total = sum(self.index2count)
        # Precalculate each vocabulary item's threshold for sampling
        if not self.sample:
            # no words downsampled
            threshold_count = retain_total
        else:
            # set parameter as proportion of total
            threshold_count = self.sample * retain_total

        self.index2sample_id = []
        downsample_total, downsample_unique = 0, 0
        for w in range(len(self.index2word)):
            v = self.index2count[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            self.index2sample.append(int(round(word_probability * 2**32)))

        logger.info("sample=%g downsampled the %s most common words", self.sample, si(downsample_unique))
        logger.info("downsampling will decrease corpus size by approximately %.1f%%",
                    downsample_total * 100.0 / max(retain_total, 1))

    def _make_ns_table(self, power=0.75, domain=2**31 - 1):
        vocabsize = len(self.index2word)
        self.ns_table = np.zeros(vocabsize, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.index2count[index]**power for index in range(vocabsize)]))
        cumulative = 0.0
        for word_index in range(vocabsize):
            cumulative += self.index2count[word_index]**power / train_words_pow
            self.ns_table[word_index] = round(cumulative * domain)
        if len(self.ns_table) > 0:
            assert self.ns_table[-1] == domain

    def _finalise_vocab(self):
        # precalculate sampling thresholds for words
        self._downsample_vocab()
        # build the table for drawing random words (for negative sampling)
        self._make_ns_table()
        # set initial input/projection and hidden weights
        self.reset_weights()

    def load_vocab(self, vocab_infile):
        self.index2word = []
        self.index2count = []
        with smart_open(vocab_infile, 'r') as f:
            for line_num,line in enumerate(f):
                tokens = line.strip().split()
                self.index2word.append(tokens[0])
                self.index2count.append(int(tokens[1]))
        logger.info("loaded a word vocabulary of size %s", si(len(self.index2word)))
        self.word2index = {e:i for i,e in enumerate(self.index2word)}
        self._finalise_vocab()

    def reset_weights(self, what=None):
        vocabsize = len(self.index2word)
        self.contexts = np.zeros((vocabsize, self.dim), dtype=floatX, order='C')
        logger.info('initialised a %s x %s context matrix', si(vocabsize), si(self.dim))

    def test_dev(self, embed):
        # array of dot products between pairs of (normalised) noun vectors
        # in the development set
        scores = np.asarray([1-cosine(embed[self.word2index[a]],embed[self.word2index[b]]) for a,b in self.dev_words])
        # return spearman rank between human similarity judgements and
        # our own similarity judgements
        return spearmanr(scores,self.devsims)

    def train_sentences(self, corpus_infile, epochs=1, report_freq=20):
        if len(self.index2sample) == 0:
            logger.error("attempted to start training but vocabulary has not been loaded")
            raise RuntimeError("You must build/load the vocabulary before training the model")
        epochs = int(epochs) or 1
        # initialise temporary work memory and word vectors
        work = np.zeros(self.dim, dtype=floatX)
        embeddings = np.ascontiguousarray((np.random.rand(len(self.index2word), self.dim) - 0.5) / self.dim,dtype=floatX)
        logger.info("initialised a %s x %s embedding matrix", si(len(self.index2word)), si(self.dim))
        with smart_open(corpus_infile, 'r') as fin:
            total_words = 0
            # read the number of sentences in the corpus
            corpus_sentences = int(next(fin).strip())
            total_sentences = epochs * corpus_sentences
            logger.info("loaded corpus with %s sentences, training for %d epochs", si(corpus_sentences), epochs)

            t = Timer(interval=report_freq)
            t.tic()
            word_count = 0
            for epoch in range(epochs):
                fin.seek(0)
                next(fin) # skip first line with number of sentences
                for sentence_num, line in enumerate(fin,start=epoch*corpus_sentences):
                    alpha = self.alpha * (1 - sentence_num / total_sentences)
                    sentence = list(map(int,line.strip().split()))
                    word_count += len(sentence)
                    train_sentence(self, sentence, alpha, embeddings, work)
                    if t.ready():
                        t.toc()
                        if self.dev:
                            cor = self.test_dev(embeddings)
                            logger.info("%.2f%% sentences @ %s words/s, alpha %.6f, corr %.5f (p %.2e)" %
                                (100 * sentence_num / total_sentences, si(word_count / t.interval), alpha, cor[0], cor[1]))
                        else:
                            logger.info("%.2f%% sentences @ %s words/s, alpha %.6f" %
                                (100 * sentence_num / total_sentences, si(word_count / t.interval), alpha))
                        total_words += word_count
                        word_count = 0
                total_words += word_count
        logger.info("trained on %s sentences (%s words) in %s @ %s words/s" %
                (si(total_sentences), si(total_words), t.toc(hms=True),
                    si(total_words / elapsed if elapsed else 0.0)))
        cor = self.test_dev(embeddings)
        logger.info("correlation on development set %.5f (p %.2e)" % cor)
        return Embeddings(A=embeddings, index2word=self.index2word)

    def train_tuples(self, corpus_infile, compound_vocab_infile,
            epochs=1, report_freq=20):
        if len(self.index2sample) == 0:
            logger.error("attempted to start training but vocabulary has not been loaded")
            raise RuntimeError("You must build/load the vocabulary before training the model")
        epochs = int(epochs) or 1
        index2word_comp = []
        index2count_comp = []
        # count the number of compound vectors to be learned
        vocabsize = 0
        with smart_open(compound_vocab_infile, 'r') as fvoc:
            for line_num,line in enumerate(fvoc):
                tokens = line.strip().split()
                index2word_comp.append(tokens[0])
                index2count_comp.append(tokens[1])
                vocabsize += 1

        # initialise temporary work memory and compound vectors
        work = np.zeros(self.dim, dtype=floatX)
        embeddings = np.ascontiguousarray((np.random.rand(vocabsize, self.dim) - 0.5) / self.dim,dtype=floatX)
        logger.info("initialised a %s x %s compound matrix", si(vocabsize), si(self.dim))

        with smart_open(corpus_infile, 'r') as fin:
            total_words = 0
            # read the number of sentences in the corpus
            corpus_sentences = int(next(fin).strip())
            total_sentences = epochs * corpus_sentences
            logger.info("loaded corpus with %s examples, training for %d epochs", si(corpus_sentences), epochs)

            t = Timer(interval=report_freq)
            t.tic()
            word_count = 0
            for epoch in range(epochs):
                fin.seek(0)
                next(fin) # skip first line with number of sentences
                for sentence_num, line in enumerate(fin,start=epoch*corpus_sentences):
                    if len(sentence) <= 1: continue
                    alpha = self.alpha * (1 - sentence_num / total_sentences)
                    sentence = list(map(int,line.strip().split()))
                    word_count += len(sentence)-1
                    train_tuple(self, sentence, alpha, embeddings, work)
                    if t.ready():
                        t.toc()
                        logger.info("%.2f%% examples @ %s words/s, alpha %.6f" %
                            (100 * sentence_num / total_sentences, si(word_count / t.interval), alpha))
                        next_report = elapsed + report_freq
                        total_words += word_count
                        word_count = 0
                total_words += word_count
        logger.info("trained on %s words (%s examples) in %s @ %s words/s" %
                (si(total_words), si(total_sentences), t.toc(hms=True),
                    si(total_words / elapsed if elapsed else 0.0)))
        return Embeddings(A=embeddings, index2word=index2word_comp, index2count=index2count_comp)

    # save/load context vectors
    def contexts(self):
        return Embeddings(A=self.contexts, index2word=self.index2word,
                index2count=self.index2count)
    def load_contexts(self, contexts):
        shape = contexts.A.shape
        if shape[0] != len(contexts.index2word) or shape[1] != self.dim:
            logger.error("vocabulary/contexts shape mismatch")
            raise RuntimeError("vocabulary/contexts shape mismatch")
        self.contexts = contexts.A
        self.index2word = contexts.index2word
        self.index2count = contexts.index2count
