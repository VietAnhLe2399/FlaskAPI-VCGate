import pandas as pd
import numpy as np
import os
from ngrams import ngrams_per_line, preprocess_text
from collections import defaultdict
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC

typeList = ['scopus', 'isi']

clfs = {
    # for ISI
    "SVM-rbf-isi": SVC(C=1000, gamma=0.01, probability=True),
    # for Scopus
    "SVM-rbf-scopus": SVC(C=100, gamma=0.01, probability=True)
}

tfidf = TfidfVectorizer(
        min_df = 1,
        max_df = 0.8,
        ngram_range=(1,3),
        use_idf = True,
        sublinear_tf = True,
        analyzer=ngrams_per_line
    )

encode = {}
encode_rev = {}

def readData(type):
    df = pd.DataFrame(columns=["content", "original", "label"])
    index = 0
    i = 0

    BASE_DIR = "./patterns/%s" % type
    concatenated_patterns = defaultdict(str)

    for file in os.listdir(BASE_DIR): 
        if ".txt" not in file: 
            continue
        label = file.split("_")[0] 
        encode[label] = index
        print(label)        
        with open(os.path.join(BASE_DIR, file), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                text = preprocess_text(line)
                if not text:
                    raise Exception
                df.loc[i] = [text, line[:-1], index]
                concatenated_patterns[label] += text + ","
                i += 1
        index += 1
    # random dataframe
    df = df.sample(frac=1).reset_index(drop = True)

    # get encode reverse
    for k, v in encode.items():
        encode_rev[v] = k

    return trainModel(type, df, concatenated_patterns)

# Training
def trainModel(type, df, concatenated_patterns):
    df["label"] = df["label"].astype("int")
    x_train_text, x_test_text, y_train, y_test = train_test_split(df[["content", "original"]], df["label"], test_size = 0.2)

    tfidf.fit(concatenated_patterns.values())
    x_train = tfidf.transform(x_train_text["content"])
    # uncomment this to take all dataset as input (overfitting train)
    # x_train = tfidf.transform(df["content"])
    # y_train = df["label"]
    x_test = tfidf.transform(x_test_text["content"])

    clfName = "SVM-rbf-" + type
    clf = clfs[clfName]
    print("*"*5 + clfName + "*"*5)
    print("TRAIN")
    clf.fit(x_train, y_train)
    print("-"*10)

    # Finish training, save model
    with open('./pythonModels/%s_tfidf.pkl' % type, 'wb') as f:
        pickle.dump(tfidf, f)
    with open('./pythonModels/%s_svm.pkl' % type, 'wb') as f:
        pickle.dump(clf, f)
    with open('./pythonModels/%s_encode_rev.pkl' % type, 'wb') as f:
        pickle.dump(encode_rev, f)

    return getResult(x_test_text,  x_test, y_test, clf)
    
def getResult(x_test_text, x_test, y_test, clf):
    result = {}
    trainPredict = defaultdict(str)
    testPredict = defaultdict(str)
    y_pred = clf.predict(x_test)
    for input, label, prediction in zip(x_test_text["original"], y_test, y_pred):
        if label != prediction:
            # print(input,"*****", encode_rev[label], "->", encode_rev[prediction])
            trainPredict[encode_rev[prediction]] += input + ";"

    print("TEST")
    print("precision", precision_score(y_test, y_pred, average="weighted"))
    result["precision"] = precision_score(y_test, y_pred, average="weighted")
    print("recall", recall_score(y_test, y_pred, average="weighted"))
    result["recall"] = recall_score(y_test, y_pred, average="weighted")
    print("F1", f1_score(y_test, y_pred, average="weighted"))
    result["F1"] = f1_score(y_test, y_pred, average="weighted")
    print("*"*20)

    for text, x in zip(x_test_text["original"], clf.predict_proba(x_test)):
        highest_prob = max(x)
        if highest_prob < 0.7:
            testPredict[encode_rev[prediction]] += text + ","
            # print(text, "***", encode_rev[clf.predict(tfidf.transform([text]))[0]], round(highest_prob, 2))
    
    result["train"] = trainPredict
    result["test"] = testPredict
    # print(result)
    return result

if __name__ == "__main__":
    for type in typeList:
        if type == "scopus":
            print("******  " + type + " ******")
            # pd.set_option('display.max_colwidth', -1)

            readData(type)
            # getTag(type)