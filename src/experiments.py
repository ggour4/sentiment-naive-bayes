import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from models import vectorize, build_classifier


def run_cv(docs, labels, combo, prune):
    """5-fold stratified cross validation για έναν συνδυασμό.

    Τα docs είναι ήδη προεπεξεργασμένα (λίστες tokens). Η προεπεξεργασία δεν
    μαθαίνει τίποτα από το corpus, οπότε γίνεται έξω από το loop.
    Επιστρέφει μία γραμμή μετρικών ανά fold και το αθροισμένο confusion matrix.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    total_cm = np.zeros((2, 2), dtype=int)
    fold = 0

    for train_idx, test_idx in skf.split(docs, labels):
        fold = fold + 1

        # φτιάχνουμε τις λίστες train/test με τα indexes του fold
        train_docs = []
        train_labels = []
        for i in train_idx:
            train_docs.append(docs[i])
            train_labels.append(labels[i])

        test_docs = []
        test_labels = []
        for i in test_idx:
            test_docs.append(docs[i])
            test_labels.append(labels[i])

        # fit μόνο στα train, transform στα test
        X_train, X_test = vectorize(combo, train_docs, test_docs, prune)

        clf = build_classifier(combo)
        clf.fit(X_train, train_labels)
        predictions = clf.predict(X_test)

        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, predictions, average='macro', zero_division=0)

        cm = confusion_matrix(test_labels, predictions, labels=[0, 1])

        # κρατάμε το confusion matrix και ανά fold, για να μείνει στο csv
        rows.append({
            'combo': combo,
            'fold': fold,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'tn': cm[0][0],
            'fp': cm[0][1],
            'fn': cm[1][0],
            'tp': cm[1][1],
        })

        total_cm = total_cm + cm

    return rows, total_cm


def average(rows, metric):
    """Μέσος όρος μιας μετρικής στα folds."""
    values = []
    for row in rows:
        values.append(row[metric])
    return np.mean(values)


def stdev(rows, metric):
    """Τυπική απόκλιση μιας μετρικής στα folds."""
    values = []
    for row in rows:
        values.append(row[metric])
    return np.std(values)


def print_summary(name, rows, total_cm):
    """Τυπώνει μέσο όρο ± τυπική απόκλιση και το confusion matrix."""
    print(name,
          '| accuracy:', round(average(rows, 'accuracy'), 4),
          '+-', round(stdev(rows, 'accuracy'), 4),
          '| macro-F1:', round(average(rows, 'f1_macro'), 4),
          '+-', round(stdev(rows, 'f1_macro'), 4))
    print('   confusion matrix (άθροισμα folds):',
          'TN =', total_cm[0][0], 'FP =', total_cm[0][1],
          'FN =', total_cm[1][0], 'TP =', total_cm[1][1])
