import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler


class DataInit:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.q1_w2v_train = None
        self.q2_w2v_train = None
        self.q1_w2v_test = None
        self.q2_w2v_test = None
        self.q1_vecs_tfidf_train = None
        self.q2_vecs_tfidf_train = None
        self.q1_vecs_tfidf_test = None
        self.q2_vecs_tfidf_test = None
        self.load_data_into_memory()

    def load_data_into_memory(self):
        PATH_DATA = "outputs/"

        PATH_TRAIN_DATA = PATH_DATA + "train_data.csv"
        PATH_TEST_DATA = PATH_DATA + "test_data.csv"
        PATH_TRAIN_LABELS = PATH_DATA + "train_labels.npy"
        PATH_TEST_LABELS = PATH_DATA + "test_labels.npy"

        PATH_Q1_W2V_TRAIN = PATH_DATA + "q1_w2v_train.csv"
        PATH_Q2_W2V_TRAIN = PATH_DATA + "q2_w2v_train.csv"
        PATH_Q1_W2V_TEST = PATH_DATA + "q1_w2v_test.csv"
        PATH_Q2_W2V_TEST = PATH_DATA + "q2_w2v_test.csv"

        PATH_Q1_VECS_TFIDF_TRAIN = PATH_DATA + "q1_tfidf_train.csv"
        PATH_Q2_VECS_TFIDF_TRAIN = PATH_DATA + "q2_tfidf_train.csv"
        PATH_Q1_VECS_TFIDF_TEST = PATH_DATA + "q1_tfidf_test.csv"
        PATH_Q2_VECS_TFIDF_TEST = PATH_DATA + "q2_tfidf_test.csv"

        self.train_data = pd.read_csv(PATH_TRAIN_DATA)
        self.test_data = pd.read_csv(PATH_TEST_DATA)
        self.train_labels = np.load(PATH_TRAIN_LABELS)
        self.test_labels = np.load(PATH_TEST_LABELS)

        self.q1_w2v_train = pd.read_csv(PATH_Q1_W2V_TRAIN)
        self.q2_w2v_train = pd.read_csv(PATH_Q2_W2V_TRAIN)
        self.q1_w2v_test = pd.read_csv(PATH_Q1_W2V_TEST)
        self.q2_w2v_test = pd.read_csv(PATH_Q2_W2V_TEST)

        self.q1_tfidf_train = pd.read_csv(PATH_Q1_VECS_TFIDF_TRAIN)
        self.q2_tfidf_train = pd.read_csv(PATH_Q2_VECS_TFIDF_TRAIN)
        self.q1_tfidf_test = pd.read_csv(PATH_Q1_VECS_TFIDF_TEST)
        self.q2_tfidf_test = pd.read_csv(PATH_Q2_VECS_TFIDF_TEST)

    def model_start(self, name_model):
        if name_model == "model_1":
            return self.model_1()
        elif name_model == "model_2":
            return self.model_2()
        elif name_model == "model_3":
            return self.model_3()
        elif name_model == "model_4":
            return self.model_4()
        elif name_model == "model_5":
            return self.model_5()

    def model_1(self):
        # Modelo 1: Features (token e fuzzy)
        scaler = StandardScaler()

        scaled = self.train_data.loc[
            :,
            (self.train_data.columns != "last_token_equals")
            & (self.train_data.columns != "first_token_equals"),
        ]
        scaled = scaler.fit_transform(scaled)

        model_train_stack = np.column_stack(
            (
                scaled,
                self.train_data["last_token_equals"],
                self.train_data["first_token_equals"],
            )
        )

        scaled = self.test_data.loc[
            :,
            (self.test_data.columns != "last_token_equals")
            & (self.test_data.columns != "first_token_equals"),
        ]
        scaled = scaler.transform(scaled)
        model_test_stack = np.column_stack(
            (
                scaled,
                self.test_data["last_token_equals"],
                self.test_data["first_token_equals"],
            )
        )

        return model_train_stack, model_test_stack, self.train_labels, self.test_labels

    def model_2(self):
        # Modelo 2: TF-IDF
        model_train_stack = hstack((self.q1_vecs_tfidf_train, self.q2_vecs_tfidf_train))
        model_test_stack = hstack((self.q1_vecs_tfidf_test, self.q2_vecs_tfidf_test))
        return model_train_stack, model_test_stack, self.train_labels, self.test_labels

    def model_3(self):
        # Modelo 3: Média ponderada IDF do Word2Vec
        list_q1 = self.q1_w2v_train[self.q1_w2v_train.isnull().any(axis=1)].index
        list_q2 = self.q2_w2v_train[self.q2_w2v_train.isnull().any(axis=1)].index
        list_q1 = self.q1_w2v_test[self.q1_w2v_test.isnull().any(axis=1)].index
        list_q2 = self.q2_w2v_test[self.q2_w2v_test.isnull().any(axis=1)].index

        lista_total_train = list(list_q1) + (list(list_q2))
        lista_total_train = self.remove_repeated_numbers(lista_total_train)
        lista_total_test = list(list_q1) + (list(list_q2))
        lista_total_test = self.remove_repeated_numbers(lista_total_test)

        q1_w2v_train = self.q1_w2v_train.drop(lista_total_train, axis=0)
        q2_w2v_train = self.q2_w2v_train.drop(lista_total_train, axis=0)
        q1_w2v_test = self.q1_w2v_test.drop(lista_total_test, axis=0)
        q2_w2v_test = self.q2_w2v_test.drop(lista_total_test, axis=0)

        model_train_stack = np.hstack((q1_w2v_train.values, q2_w2v_train.values))
        model_test_stack = np.hstack((q1_w2v_test.values, q2_w2v_test.values))

        new_train_labels_truncated = []
        for index, element in enumerate(self.train_labels):
            if index not in lista_total_train:
                new_train_labels_truncated.append(element)

        new_test_labels_truncated = []
        for index, element in enumerate(self.test_labels):
            if index not in lista_total_test:
                new_test_labels_truncated.append(element)

        return (
            model_train_stack,
            model_test_stack,
            new_train_labels_truncated,
            new_test_labels_truncated,
        )

    def model_4(self):
        # Modelo 4: TF-IDF + Features
        scaler = StandardScaler()

        scaled = self.train_data.loc[
            :,
            (self.train_data.columns != "last_token_equals")
            & (self.train_data.columns != "first_token_equals"),
        ]
        scaled = scaler.fit_transform(scaled)

        model_train_stack = np.column_stack(
            (
                scaled,
                self.train_data["last_token_equals"],
                self.train_data["first_token_equals"],
            )
        )

        scaled = self.test_data.loc[
            :,
            (self.test_data.columns != "last_token_equals")
            & (self.test_data.columns != "first_token_equals"),
        ]
        scaled = scaler.transform(scaled)

        model_test_stack = np.column_stack(
            (
                scaled,
                self.test_data["last_token_equals"],
                self.test_data["first_token_equals"],
            )
        )

        model_train_stack = hstack(
            (model_train_stack, self.q1_vecs_tfidf_train, self.q2_vecs_tfidf_train)
        )
        model_test_stack = hstack(
            (model_test_stack, self.q1_vecs_tfidf_test, self.q2_vecs_tfidf_test)
        )

        return model_train_stack, model_test_stack, self.train_labels, self.test_labels

    def model_5(self):
        # Modelo 5: Média ponderada IDF do Word2Vec + Features
        scaler = StandardScaler()

        scaled = self.df_train.loc[
            :,
            (self.df_train.columns != "last_token_equals")
            & (self.df_train.columns != "first_token_equals"),
        ]
        scaled = scaler.fit_transform(scaled)

        model_train_stack = np.column_stack(
            (
                scaled,
                self.df_train["last_token_equals"],
                self.df_train["first_token_equals"],
            )
        )

        scaled = self.df_test.loc[
            :,
            (self.df_test.columns != "last_token_equals")
            & (self.df_test.columns != "first_token_equals"),
        ]
        scaled = scaler.transform(scaled)

        model_test_stack = np.column_stack(
            (
                scaled,
                self.df_test["last_token_equals"],
                self.df_test["first_token_equals"],
            )
        )

        model_train_stack = np.hstack(
            (model_train_stack, self.q1_w2v_train.values, self.q2_w2v_train.values)
        )
        model_test_stack = np.hstack(
            (model_test_stack, self.q1_w2v_test.values, self.q2_w2v_test.values)
        )

        return model_train_stack, model_test_stack, self.train_labels, self.test_labels

    def remove_repeated_numbers(self, list_number):
        list_aux = []
        for i in list_number:
            if i not in list_aux:
                list_aux.append(i)
        list_aux.sort()
        return list_aux
