import networkx as nx
import numpy as np
import spacy
from itertools import combinations
import re
from nlp_rake import Rake
import yake
from gensim.models import Word2Vec
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import torch
from transformers import logging, BertModel, BertTokenizerFast
from rusenttokenize import ru_sent_tokenize
import emoji
import os

logging.set_verbosity_warning()
logging.set_verbosity_error()

nlp = spacy.load("ru_core_news_lg")
all_stopwords = nlp.Defaults.stop_words

if __name__ == '__main__':
    with open("configs/superwords.txt", 'r') as file:
        superwords = file.read().split('\n')
else:
    with open(os.path.abspath('.') + "/searchformeaning/summarizer/configs/superwords.txt", 'r') as file:
        superwords = file.read().split('\n')


class SearchForMeaning:
    """
    SearchForMeaning:  Extractive Text Summarization based on the following algorithms: Importance Calculation,
                       TextRank, LexRank, Latent Semantic Analysis, Luhn Algorithm, The Minto Pyramid (Lead-based),
                       Features Calculation, Word2Vec, RuBERT.

    License:           The MIT License.

    Author:            Shamil Zaripov (@tonyfierro), 2023.
    """

    def __init__(self, txt: str, extract_percentage: int | float) -> None:
        """
        Initialize class objects and making those objects ready to use.

        :param txt: (str)
            A text that will be analysed.
        :param extract_percentage: (int | float)
            Percentage part of text that will be extracted.
        """

        validity, self.raw_sentences, self._tokens_of_sentences = self._check_validity(txt)
        if validity is True:
            self._plaintext_sentences = PlaintextParser.from_string(txt, Tokenizer("russian"))
            self._AMOUNT_OF_SENTENCES = self._calculate_amount_of_sentences(extract_percentage)
            self._MINTO_PERCENTAGE = 0.25
        else:
            raise ValueError(
                "The text is incorrect! The text must be satisfied the next conditions: length must be more than 50 "
                "cyrillic characters, number of sentences must be more than 3 and the text mustn't have sentences that "
                "contains only latin characters."
            )

    def _check_validity(self, txt: str) -> (bool, list[str] | None, list[list] | None):
        """
        This method checks the admissibility of the text.

        :param txt: (str)
            String in Russian language.
        :return: (bool, list[str] | NoneType, list[list] | NoneType)
            Returns (True, split sentences, tokens of sentences) if the text is allowable,
            otherwise returns (False, None, None).

        """

        for character in txt:
            if character in emoji.EMOJI_DATA:
                return False, None, None

        if len(self._process_text(txt).replace('.', '').replace(' ', '')) > 50:
            raw_sentences = ru_sent_tokenize(txt.replace('\n', ' '))
            if len(raw_sentences) > 3:
                tokens_of_sentences = self._prepare_text(txt)
                if tokens_of_sentences.count([]) == 0:
                    return True, raw_sentences, tokens_of_sentences
        return False, None, None

    def _calculate_amount_of_sentences(self, percent: int | float) -> int:
        """
        This method calculates the number of sentences that will be extracted.

        :param percent: (int | float)
            The percentage range: 1 <= percent <= 100. If percent > 100 then percent will be counted as 100.
        :return: (int)
            The number of sentences.
        """

        if 1 <= percent <= 100:
            return round(len(self.raw_sentences) * (percent / 100))
        else:
            return len(self.raw_sentences)

    @staticmethod
    def _process_text(txt: str) -> str:
        """
        This method removes the junk characters in text.

        :param txt: (str)
            String in Russian language, latin characters will be deleted though.
        :return: (str)
            Returns a string without unnecessary characters.
        """

        return re.sub(r'[A-Za-z0-9#$%&\'()*+«»,…/:;<=>№@[\]^_`{|}~—\"\-]+', '', txt).replace('\n', ' ')

    @staticmethod
    def _split_sentences(txt: str) -> list[str]:
        """
        This method splits text into sentences.

        :param txt: (str)
            Text that will be split.
        :return: (list[str])
            A List which contains the sentences.
        """

        return ru_sent_tokenize(txt)

    @staticmethod
    def _tokenize_sentence(sentence: str) -> list[str]:
        """
        Used to split sentences into tokens (words).

        :param sentence: (str)
            Sentence that needs to be split into tokens.
        :return: (list[str])
            A List that contains tokens.
        """

        return sentence[:-1].split()

    @staticmethod
    def _identify_pos(tokens: list[str]) -> list[list[str]]:
        """
        Used to identify part of speech in the tokens.

        :param tokens: (list[str])
            A List that contains words-tokens.
        :return: (list[list[str]])
            List that contains the infinitive forms of the tokens that was entranced.
        """

        result = list()
        for token in tokens:
            doc = nlp(token.lower())
            for doc_elem in doc:
                result.append([doc_elem.text, doc_elem.pos_, doc_elem.lemma_])
        return result

    @staticmethod
    def _remove_stopwords(tokens_list: list[str]) -> list[str]:
        """
        This method removes the stopwords.

        :param tokens_list: (list[str])
            A list that contains tokens.
        :return: (list[str])
            A list without the stopwords.
        """

        new_tokens_list = list()
        for token in tokens_list:
            if token not in all_stopwords:
                new_tokens_list.append(token)
        return new_tokens_list

    def _prepare_text(self, txt: str) -> list[list[str]]:
        """
        This method unions three methods which perform text-preprocessing and preparing text for further analysis.

        :param txt: (str)
            A raw text.
        :return: (list[str])
            Tokens in a list.
        """

        txt = self._process_text(txt)
        txt = self._split_sentences(txt)
        txt = [self._tokenize_sentence(snt) for snt in txt]
        return txt

    @staticmethod
    def _calculate_importance_coefficient(wb: dict[str]) -> dict[str, float]:
        """
        This method calculates the importance-coefficient of the word.

        :param wb: (dict[str])
            Words-base that contains words with their counts.
        :return: (dict[str, float])
            Words-base that contains words with their importance-coefficients.
        """

        wb_dict = dict()
        for key, value in wb.items():
            wb_dict[key] = value / len(wb)
        return wb_dict

    @staticmethod
    def _calculate_similarity_coefficient(a: list[str], b: list[str]) -> int | float:
        """
        This method calculates the similarity-coefficient of the sentences.

        :param a: (list[str])
            A list that contains tokens of the sentence #1.
        :param b: (list[str])
            A list that contains tokens of the sentence #2.
        :return: (int | float)
            The similarity-coefficient.
        """

        if not len(a) or not len(b):
            return 0
        return len(set(a).intersection(set(b))) / (len(a) + len(b))

    def _create_wordsbase(self, tokens_of_sentences: list[list[str]]) -> dict[str, int]:
        """
        This method creates a "words-base" - words with their counts.

        :param tokens_of_sentences: (list[str])
            A list that contains raw tokens (words).
        :return: (dict[str])
            A dictionary that contains the words with their counts.
        """

        wordsbase = dict()
        for snt in tokens_of_sentences:
            tokens_without_sw = self._remove_stopwords(snt)
            part_of_speech = self._identify_pos(tokens_without_sw)
            for word in part_of_speech:
                if word[2] not in wordsbase:
                    wordsbase[word[2]] = 1
                else:
                    wordsbase[word[2]] += 1
        return wordsbase

    def extract_sentences(self, weights: list[float]) -> list[tuple[int, str]]:
        """
        This method extracts sentences.

        :param weights: (list[float])
            A list that contains scores (weights) of the sentences.
        :return: (list[tuple[int, str]])
            Returns a list that contains the most important sentences.
        """

        sentences = sorted(zip(weights, enumerate(self.raw_sentences)),
                           reverse=True)[:self._AMOUNT_OF_SENTENCES]
        # noinspection PyTypeChecker
        return sorted([snt[1] for snt in sentences])

    def _join_sentences_with_indexes(self, summary: tuple) -> list[tuple[int, str]]:
        """
        Used to join sentences with their indexes.

        :param summary: (tuple)
            A tuple that contains sentences that needs to be joined with their indexes.
        :return: (list[tuple[int, str]])
            A list that contains sentences with indexes.
        """

        sentences = [str(row) for row in summary]
        sentences_indexed = list()
        for row in sentences:
            for idx, raw_row in enumerate(self.raw_sentences):
                if row == raw_row:
                    sentences_indexed.append((idx, raw_row))
        return sentences_indexed

    def _transform_pagerank_to_weights(self, pr_graph: dict[int]) -> list[float]:
        """
        Used to transform pagerank's format into list format.

        :param pr_graph: (dict[int])
            PageRank's output (a dictionary that contains scores).
        :return: (list[float])
            A list that contains scores (weights) of the sentences.
        """

        weights = [0] * len(self.raw_sentences)
        for key in pr_graph.keys():
            weights[key] = pr_graph[key]
        return weights

    def use_importance_algorithm(self) -> list[tuple[int, str]]:
        """
        The algorithm based on importance.

        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        wordsbase = self._create_wordsbase(self._tokens_of_sentences)
        wordsbase_importance_values = self._calculate_importance_coefficient(wordsbase)

        importance_of_sentences = [0] * len(self.raw_sentences)
        for idx, tokens in enumerate(self._tokens_of_sentences):
            tokens = self._identify_pos(tokens)

            for token in tokens:
                if token[2] in wordsbase_importance_values:
                    importance_of_sentences[idx] += wordsbase_importance_values[token[2]]
        return self.extract_sentences(importance_of_sentences)

    def use_textrank(self) -> list[tuple[int, str]]:
        """
        The TextRank algorithm.

        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        words = list()
        for snt in self._tokens_of_sentences:
            part_of_speech = [row[2] for row in self._identify_pos(snt)]
            words.append(part_of_speech)

        pairs = combinations(range(len(self.raw_sentences)), 2)
        similarity_result = [[i, j, self._calculate_similarity_coefficient(words[i], words[j])] for i, j in pairs]
        similarity_result_filtered = filter(lambda x: x[1], similarity_result)
        graph = nx.Graph()
        graph.add_weighted_edges_from(similarity_result_filtered)
        pr_graph = nx.pagerank(graph)
        weights = self._transform_pagerank_to_weights(pr_graph)
        return self.extract_sentences(weights)

    def use_lexrank(self) -> list[tuple[int, str]]:
        """
        The LexRank algorithm.

        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        lexrank = LexRankSummarizer()
        summary = lexrank(self._plaintext_sentences.document, self._AMOUNT_OF_SENTENCES)
        return self._join_sentences_with_indexes(summary)

    def use_lsa(self, sw: bool = True) -> list[tuple[int, str]]:
        """
        The Latent Semantic Analysis (LSA algorithm).

        :param sw: (bool)
            The parameter that enable or disable stopwords.
        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        lsa = LsaSummarizer(Stemmer("russian"))
        if sw is True:
            lsa.stop_words = all_stopwords

        summary = lsa(self._plaintext_sentences.document, self._AMOUNT_OF_SENTENCES)
        return self._join_sentences_with_indexes(summary)

    def use_luhn(self) -> list[tuple[int, str]]:
        """
        The Luhn algorithm.

        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        luhn = LuhnSummarizer()
        summary = luhn(self._plaintext_sentences.document, self._AMOUNT_OF_SENTENCES)
        return self._join_sentences_with_indexes(summary)

    def use_minto(self) -> list[tuple[int, str]]:
        """
        The algorithm based on The Minto Pyramid.

        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        # noinspection PyTypeChecker
        return list(enumerate(self.raw_sentences))[:round(len(self.raw_sentences) * self._MINTO_PERCENTAGE)]

    def use_features_algorithm(self) -> list[tuple[int, str]]:
        """
        The algorithm based on the features of the text. This algorithm checking for: the dates, the super-words and
        find keywords. Then counts the numbers of features and forms the scores of the sentences.

        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        coefficients_of_sentences = [0] * len(self.raw_sentences)

        for idx, snt in enumerate(self.raw_sentences):

            # Check for dates.
            date = re.findall(r'\d{2,4}.\d{2}.\d{2,4}', snt)
            if len(date) != 0:
                coefficients_of_sentences[idx] += len(date)

            #  Check for super-words.
            lemmatized_words = self._identify_pos(self._tokens_of_sentences[idx])
            lemmatized_sentence = ' '.join([row[2] for row in lemmatized_words])

            for word in superwords:
                if word in lemmatized_sentence:
                    coefficients_of_sentences[idx] += lemmatized_sentence.count(word)

            #  Check for keywords.
            rake_result = Rake(stopwords=all_stopwords, max_words=3).apply(snt)
            rake_sum = sum([kw[1] for kw in rake_result])
            coefficients_of_sentences[idx] += rake_sum
            yake_result = yake.KeywordExtractor(lan='ru', n=3, dedupLim=0.1, top=3).extract_keywords(snt)
            yake_sum = sum([kw[1] for kw in yake_result])
            coefficients_of_sentences[idx] += yake_sum

        return self.extract_sentences(coefficients_of_sentences)

    def use_word2vec(self, model_w2v: Word2Vec) -> list[tuple[int, str]]:
        """
        The Word2Vec algorithm.

        :param model_w2v: (Word2Vec)
            The parameter that receives a Word2Vec model.
        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        vectors = list()
        for sentence in self._tokens_of_sentences:
            tokens = [tkn[2] for tkn in self._identify_pos(sentence) if tkn[2] not in all_stopwords]
            vector = np.zeros(300)
            for token in tokens:
                try:
                    vector += model_w2v.wv[token]
                except KeyError:
                    pass

            if len(tokens) != 0:
                vector /= len(tokens)
            vectors.append(vector)

        pairs = combinations(range(len(self.raw_sentences)), 2)
        similarity_matrix = [[i, j, np.dot(vectors[i], vectors[j]) /
                              (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))] for i, j in pairs]
        similarity_matrix_filtered = filter(lambda x: x[1], similarity_matrix)
        graph = nx.Graph()
        graph.add_weighted_edges_from(similarity_matrix_filtered)
        pr_graph = nx.pagerank(graph)
        weights = self._transform_pagerank_to_weights(pr_graph)

        return self.extract_sentences(weights)

    def use_rubert(self, model_rubert: BertModel, tokenizer_rubert: BertTokenizerFast) -> list[tuple[int, str]]:
        """
        The algorithm based on RuBERT model.

        :param model_rubert: (BertModel)
            The parameter that receives a RuBERT model.
        :param tokenizer_rubert: (BertTokenizerFast)
            The parameter that receives a tokenizer of the RuBERT model.
        :return: (list[tuple[int, str]])
            Returns a list with sentences and their indexes.
        """

        sentences = [re.sub(r'[A-Za-z0-9#$%&\'()*+«»,…/.!?:;<=>№@[\]^_`{|}~—\"\-]+', ' ', snt)
                     for snt in self.raw_sentences]

        vectors = list()

        for sentence in sentences:
            tensor = tokenizer_rubert(sentence, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model_rubert(**{k: v.to(model_rubert.device) for k, v in tensor.items()})
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)
            vectors.append(embeddings[0].cpu().numpy())

        pairs = combinations(range(len(self.raw_sentences)), 2)
        similarity_matrix = [[i, j, np.dot(vectors[i], vectors[j]) /
                              (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))] for i, j in pairs]
        similarity_matrix_filtered = filter(lambda x: x[1], similarity_matrix)

        graph = nx.Graph()
        graph.add_weighted_edges_from(similarity_matrix_filtered)
        pr_graph = nx.pagerank(graph)
        weights = self._transform_pagerank_to_weights(pr_graph)

        return self.extract_sentences(weights)
