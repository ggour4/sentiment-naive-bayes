import numpy as np

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler


# Οι 5 συνδυασμοί κατηγοριοποιητή + αναπαράστασης.
COMBOS = ['MNB_COUNTS', 'MNB_TFIDF', 'MNB_W2V', 'BNB_BIN', 'BOOL_MNB']


def identity(tokens):
    # Τα κείμενα είναι ήδη σε tokens, οπότε ο vectorizer τα παίρνει ως έχουν.
    # Χωρίς αυτό, το default token_pattern του sklearn θα έκοβε το don't σε don.
    return tokens


def build_vectorizer(combo, prune):
    """Φτιάχνει τον vectorizer του συνδυασμού."""
    if prune:
        min_df = 2
        max_df = 0.9
    else:
        min_df = 1
        max_df = 1.0

    if combo == 'MNB_COUNTS':
        return CountVectorizer(analyzer=identity, min_df=min_df, max_df=max_df)

    if combo == 'MNB_TFIDF':
        return TfidfVectorizer(analyzer=identity, min_df=min_df, max_df=max_df)

    if combo == 'BNB_BIN':
        return CountVectorizer(analyzer=identity, binary=True,
                               min_df=min_df, max_df=max_df)

    if combo == 'BOOL_MNB':
        return CountVectorizer(analyzer=identity, binary=True,
                               min_df=min_df, max_df=max_df)


def document_vector(tokens, w2v):
    """Μέσος όρος των διανυσμάτων των λέξεων του εγγράφου (mean pooling)."""
    vectors = []
    for token in tokens:
        # κρατάμε μόνο όσες λέξεις έμαθε το Word2Vec από τα train κείμενα
        if token in w2v.wv:
            vectors.append(w2v.wv[token])

    if len(vectors) == 0:
        # έγγραφο χωρίς γνωστές λέξεις, το αφήνουμε μηδενικό
        return np.zeros(w2v.vector_size)

    return np.mean(vectors, axis=0)


def word2vec_features(train_docs, test_docs, prune):
    """Word2Vec + mean pooling + MinMaxScaler.

    Το Word2Vec εκπαιδεύεται ΜΟΝΟ στα train κείμενα και ο scaler κάνει fit
    ΜΟΝΟ στα train διανύσματα, αλλιώς έχουμε διαρροή.
    """
    # Στο W2V δεν υπάρχει min_df/max_df, οπότε το pruning γίνεται min_count.
    if prune:
        min_count = 2
    else:
        min_count = 1

    # workers=1 για να βγαίνουν τα ίδια αποτελέσματα σε κάθε εκτέλεση
    w2v = Word2Vec(sentences=train_docs, vector_size=100, window=5,
                   min_count=min_count, epochs=10, seed=42, workers=1)

    train_vectors = []
    for tokens in train_docs:
        train_vectors.append(document_vector(tokens, w2v))

    test_vectors = []
    for tokens in test_docs:
        test_vectors.append(document_vector(tokens, w2v))

    # ο MultinomialNB δεν δέχεται αρνητικές τιμές, γι' αυτό MinMaxScaler.
    # Το clip=True κόβει τις τιμές του test που πέφτουν έξω από το [0, 1].
    scaler = MinMaxScaler(clip=True)
    X_train = scaler.fit_transform(train_vectors)
    X_test = scaler.transform(test_vectors)

    return X_train, X_test


def vectorize(combo, train_docs, test_docs, prune):
    """Μετατρέπει τα κείμενα σε πίνακες χαρακτηριστικών."""
    if combo == 'MNB_W2V':
        return word2vec_features(train_docs, test_docs, prune)

    vec = build_vectorizer(combo, prune)
    X_train = vec.fit_transform(train_docs)
    X_test = vec.transform(test_docs)
    return X_train, X_test


def build_classifier(combo):
    """Ο κατηγοριοποιητής του συνδυασμού."""
    if combo == 'BNB_BIN':
        return BernoulliNB()

    return MultinomialNB()
