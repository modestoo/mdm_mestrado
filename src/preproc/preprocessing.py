import re

import distance
import nltk
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords


class Preprocessing:
    def __init__(self):
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger")

    def get_token_features(self, q1, q2):

        safe_div = 0.0001

        stop_words = stopwords.words("english")

        stop_words.append("difference")
        stop_words.append("different")
        stop_words.append("best")

        token_features = [0.0] * 14

        q1 = q1.split()
        q2 = q2.split()

        q1_stops = set([word for word in q1 if word in stop_words])
        q2_stops = set([word for word in q2 if word in stop_words])
        common_stops = q1_stops & q2_stops

        q1 = [word for word in q1 if word not in stop_words]
        q2 = [word for word in q2 if word not in stop_words]

        q1_stemmed = " ".join([word for word in q1])
        q2_stemmed = " ".join([word for word in q2])

        if len(q1) == 0 or len(q2) == 0:
            return (token_features, q1_stemmed, q2_stemmed)

        q1_tagged = nltk.pos_tag(q1)
        q2_tagged = nltk.pos_tag(q2)

        q1_adj = set()
        q2_adj = set()
        q1_prn = set()
        q2_prn = set()
        q1_n = set()
        q2_n = set()

        for word in q1_tagged:
            if word[1] == "JJ" or word[1] == "JJR" or word[1] == "JJS":
                q1_adj.add(word[0])
            elif word[1] == "NNP" or word[1] == "NNPS":
                q1_prn.add(word[0])
            elif word[1] == "NN" or word[1] == "NNS":
                q1_n.add(word[0])

        for word in q2_tagged:
            if word[1] == "JJ" or word[1] == "JJR" or word[1] == "JJS":
                q2_adj.add(word[0])
            elif word[1] == "NNP" or word[1] == "NNPS":
                q2_prn.add(word[0])
            elif word[1] == "NN" or word[1] == "NNS":
                q2_n.add(word[0])

        q1 = set(q1)
        q2 = set(q2)
        common_tokens = q1 & q2

        q1_words = set(q1)
        q2_words = set(q2)
        common_words = q1_words & q2_words

        # Features: counter tokens
        token_features[0] = len(q1) * 1.0
        token_features[1] = len(q2) * 1.0
        token_features[2] = len(q1_stemmed) * 1.0
        token_features[3] = len(q2_stemmed) * 1.0
        token_features[4] = len(common_words) * 1.0
        token_features[5] = len(q1_adj & q2_adj)
        token_features[6] = len(q1_prn & q2_prn)
        token_features[7] = len(q1_n & q2_n)

        # Features: proportions tokens
        token_features[8] = (len(common_stops) * 1.0) / (
            min(len(q1_stops), len(q2_stops)) + safe_div
        )
        token_features[9] = (len(common_stops) * 1.0) / (
            max(len(q1_stops), len(q2_stops)) + safe_div
        )
        token_features[10] = (len(common_tokens) * 1.0) / (
            min(len(q1), len(q2)) + safe_div
        )
        token_features[11] = (len(common_tokens) * 1.0) / (
            max(len(q1), len(q2)) + safe_div
        )

        # Features: conditional tokens
        token_features[12] = int(q1[0] == q2[0])
        token_features[13] = int(q1[-1] == q2[-1])

        return (token_features, q1_stemmed, q2_stemmed)

    def get_fuzzy_features(self, q1, q2):

        fuzzy_features = [0.0] * 3

        fuzzy_features[0] = fuzz.partial_ratio(q1, q2)
        fuzzy_features[1] = fuzz.token_sort_ratio(q1, q2)
        fuzzy_features[2] = fuzz.token_set_ratio(q1, q2)

        return fuzzy_features

    def get_length_features(self, q1, q2):

        safe_div = 0.0001

        length_features = [0.0] * 4

        q1_list = q1.strip(" ")
        q2_list = q2.strip(" ")

        length_features[0] = (len(q1_list) + len(q2_list)) / 2
        length_features[1] = (len(q1_list) + len(q2_list)) / 2
        substr_len = distance.lcsubstrings(q1, q2, positions=True)[0]
        if substr_len == 0:
            length_features[2] = 0
        else:
            length_features[2] = substr_len / (
                min(len(q1_list), len(q2_list)) + safe_div
            )
        length_features[3] = abs(len(q1_list) - len(q2_list))

        return length_features

    def extract_features(self, data):

        data["question1"] = data["question1"].apply(self.preprocess)
        data["question2"] = data["question2"].apply(self.preprocess)

        token_features = data.apply(
            lambda x: self.get_token_features(x["question1"], x["question2"]), axis=1
        )

        q1_stemmed = list(map(lambda x: x[1], token_features))
        q2_stemmed = list(map(lambda x: x[2], token_features))
        token_features = list(map(lambda x: x[0], token_features))

        # Features: counter tokens
        data["question1"] = q1_stemmed
        data["question2"] = q2_stemmed
        data["len_question1"] = list(map(lambda x: x[0], token_features))
        data["len_question2"] = list(map(lambda x: x[1], token_features))
        data["words_question1"] = list(map(lambda x: x[2], token_features))
        data["words_question2"] = list(map(lambda x: x[3], token_features))
        data["mutual_words"] = list(map(lambda x: x[4], token_features))
        data["count_mutual_adjective"] = list(map(lambda x: x[5], token_features))
        data["count_mutual_proper_name"] = list(map(lambda x: x[6], token_features))
        data["count_mutual_noun"] = list(map(lambda x: x[7], token_features))

        # Features: proportions tokens
        data["ratio_min_stop_words_question"] = list(
            map(lambda x: x[8], token_features)
        )
        data["ratio_max_stop_words_question"] = list(
            map(lambda x: x[9], token_features)
        )
        data["ratio_min_stop_words"] = list(map(lambda x: x[10], token_features))
        data["ratio_max_stop_words"] = list(map(lambda x: x[11], token_features))

        # Features: conditional tokens
        data["first_token_equals"] = list(map(lambda x: x[12], token_features))
        data["last_token_equals"] = list(map(lambda x: x[13], token_features))

        # Features: token length
        length_features = data.apply(
            lambda x: self.get_length_features(x["question1"], x["question2"]), axis=1
        )
        data["len_mean"] = list(map(lambda x: x[0], length_features))
        data["len_median"] = list(map(lambda x: x[1], length_features))
        data["ratio_max_min"] = list(map(lambda x: x[2], length_features))
        data["absolute_difference"] = list(map(lambda x: x[3], length_features))

        # Features: token fuzzy
        fuzzy_features = data.apply(
            lambda x: self.get_fuzzy_features(x["question1"], x["question2"]), axis=1
        )
        data["ratio_fuzz_partial"] = list(map(lambda x: x[1], fuzzy_features))
        data["ratio_sort_token"] = list(map(lambda x: x[2], fuzzy_features))
        data["ratio_set_token"] = list(map(lambda x: x[3], fuzzy_features))

        return data

    def preprocess(self, q):

        q = str(q).lower().strip()

        q = q.replace("%", " percent")
        q = q.replace("$", " dollar ")
        q = q.replace("₹", " rupee ")
        q = q.replace("€", " euro ")
        q = q.replace("@", " at ")

        q = q.replace("[math]", "")

        q = q.replace(",000,000,000 ", "b ")
        q = q.replace(",000,000 ", "m ")
        q = q.replace(",000 ", "k ")
        q = re.sub(r"([0-9]+)000000000", r"\1b", q)
        q = re.sub(r"([0-9]+)000000", r"\1m", q)
        q = re.sub(r"([0-9]+)000", r"\1k", q)

        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "can not",
            "can't've": "can not have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have",
        }

        q_decontracted = []

        for word in q.split():
            if word in contractions:
                word = contractions[word]

            q_decontracted.append(word)

        q = " ".join(q_decontracted)
        q = q.replace("'ve", " have")
        q = q.replace("n't", " not")
        q = q.replace("'re", " are")
        q = q.replace("'ll", " will")

        q = BeautifulSoup(q)
        q = q.get_text()

        pattern = re.compile("\W")
        q = re.sub(pattern, " ", q).strip()

        return q
