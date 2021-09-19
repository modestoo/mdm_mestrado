import en_core_web_lg
import numpy as np
import pandas as pd
from preproc.preprocessing import Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm

nlp = en_core_web_lg.load()
preprocessing = Preprocessing()


def data_init():
    PATH_DATASET_TRAIN = "dataset/new_train.csv"
    PATH_DATASET_TEST = "dataset/new_test.csv"

    df_train = pd.read_csv(PATH_DATASET_TRAIN, index=False)
    df_test = pd.read_csv(PATH_DATASET_TEST, index=False)

    df_train.columns = ["id", "qid1", "qid2", "question1", "question2", "duplicada"]
    df_test.columns = ["id", "qid1", "qid2", "question1", "question2", "duplicada"]

    train_data, train_labels = df_train.iloc[:, :-1], df_train["duplicada"]
    test_data, test_labels = df_test.iloc[:, :-1], df_test["duplicada"]

    return train_data, train_labels, test_data, test_labels


# Realiza o preprocessamento dos dados
def preprocessing_init(train_data, train_labels, test_data, test_labels):
    train_data = preprocessing.extract_features(train_data)
    train_data.to_csv("outputs/cleaned_featurized_train.csv", index=False)
    np.save("outputs/train_labels.npy", train_labels)

    test_data = preprocessing.extract_features(test_data)
    test_data.to_csv("outputs/cleaned_featurized_test.csv", index=False)
    np.save("outputs/test_labels.npy", test_labels)

    # Encoding das duplas de questões
    train_data["question1"] = train_data["question1"].apply(lambda x: str(x))
    train_data["question2"] = train_data["question2"].apply(lambda x: str(x))
    test_data["question1"] = test_data["question1"].apply(lambda x: str(x))
    test_data["question2"] = test_data["question2"].apply(lambda x: str(x))

    train_data = train_data.drop(
        columns=[
            "question1",
            "question2",
            "qid1",
            "qid2",
            "duplicada",
            "id",
            "is_duplicate",
        ]
    )
    test_data = test_data.drop(
        columns=["question1", "question2", "qid1", "qid2", "duplicada", "id"]
    )

    train_data.to_csv("outputs/train_data.csv", index=False)
    test_data.to_csv("outputs/test_data.csv", index=False)

    return train_data, train_labels, test_data, test_labels


def tfidf_vetorization(train_data, test_data):
    # Vetorização dos dados com TF-IDF
    vectorizer = TfidfVectorizer(lowercase=False)

    questions = list(list(train_data["question1"]) + list(train_data["question2"]))
    vectorizer.fit(questions)

    q1_vecs_tfidf_train = vectorizer.transform(train_data["question1"].values)
    q1_vecs_tfidf_test = vectorizer.transform(test_data["question1"].values)
    q2_vecs_tfidf_train = vectorizer.transform(train_data["question2"].values)
    q2_vecs_tfidf_test = vectorizer.transform(test_data["question2"].values)

    idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    q1_tfidf_train = pd.DataFrame.sparse.from_spmatrix(q1_vecs_tfidf_train)
    q2_tfidf_train = pd.DataFrame.sparse.from_spmatrix(q2_vecs_tfidf_train)
    q1_tfidf_test = pd.DataFrame.sparse.from_spmatrix(q1_vecs_tfidf_test)
    q2_tfidf_test = pd.DataFrame.sparse.from_spmatrix(q2_vecs_tfidf_test)

    q1_tfidf_train.to_csv("outputs/q1_tfidf_train.csv", index=False)
    q2_tfidf_train.to_csv("outputs/q2_tfidf_train.csv", index=False)
    q1_tfidf_test.to_csv("outputs/q1_tfidf_test.csv", index=False)
    q2_tfidf_test.to_csv("outputs/q2_tfidf_test.csv", index=False)

    return idf


def word2vec_vetorization(train_data, test_data, idf):

    # Vetorização dos dados com o Word2Vec
    q1_train_vecs = []
    q2_train_vecs = []
    q1_test_vecs = []
    q2_test_vecs = []

    q1_train_vecs = apply_word2vec(train_data["question1"], idf)
    q2_train_vecs = apply_word2vec(train_data["question1"], idf)
    q1_test_vecs = apply_word2vec(test_data["question1"], idf)
    q2_test_vecs = apply_word2vec(test_data["question2"], idf)

    q1_w2v_train = pd.DataFrame(q1_train_vecs, index=train_data.index)
    q2_w2v_train = pd.DataFrame(q2_train_vecs, index=train_data.index)
    q1_w2v_test = pd.DataFrame(q1_test_vecs, index=test_data.index)
    q2_w2v_test = pd.DataFrame(q2_test_vecs, index=test_data.index)

    q1_w2v_train.to_csv("outputs/q1_w2v_train.csv", index=False)
    q2_w2v_train.to_csv("outputs/q2_w2v_train.csv", index=False)
    q1_w2v_test.to_csv("outputs/q1_w2v_test.csv", index=False)
    q2_w2v_test.to_csv("outputs/q2_w2v_test.csv", index=False)


def apply_word2vec(data, idf):
    len_vector_doc = 300
    output_list = []

    for q in tqdm(list(data)):
        doc = nlp(q)
        mean_vec = np.zeros((len_vector_doc))

        for word in doc:
            vector = word.vector
            if str(word) in idf:
                idf_weight = idf[str(word)]
            else:
                idf_weight = 0
            mean_vec += vector * idf_weight

        mean_vec /= len(doc)
        output_list.append(mean_vec)

    return output_list


def preprocessing_data():

    train_data, train_labels, test_data, test_labels = data_init()
    train_data, train_labels, test_data, test_labels = preprocessing_init(
        train_data, train_labels, test_data, test_labels
    )
    idf = tfidf_vetorization(train_data, test_data)
    word2vec_vetorization(train_data, test_data, idf)
