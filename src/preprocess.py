import os
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Οι λέξεις άρνησης. Χρησιμεύουν σε δύο σημεία: για να τις βγάλουμε από τη
# stoplist (keep_negations) και για να ξεκινάει από αυτές η άρνηση του Βα.
NEGATION_WORDS = [
    'no', 'not', 'never', 'none', 'nothing', 'nobody', 'neither', 'nor',
    'cannot', "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't",
    "weren't", "won't", "wouldn't", "can't", "couldn't", "shouldn't",
    "hasn't", "haven't", "hadn't",
    # Το n't δεν εμφανίζεται ποτέ σε αυτό το dataset (οι συναιρέσεις είναι
    # ολόκληρες), το κρατάμε για την περίπτωση αλλιώς tokenized κειμένου.
    "n't",
]

# Τα φτιάχνουμε μία φορά, γιατί η preprocess καλείται για κάθε πρόταση.
STOPWORDS = set(stopwords.words('english'))
STOPWORDS_KEEPNEG = STOPWORDS - set(NEGATION_WORDS)
NEGATION_SET = set(NEGATION_WORDS)
STEMMER = PorterStemmer()


def is_punctuation(token):
    """True αν το token αποτελείται μόνο από σημεία στίξης."""
    for char in token:
        if char not in string.punctuation:
            return False
    return True


def apply_negation(tokens):
    """Βα: βάζει πρόθεμα NOT_ στα tokens μετά από λέξη άρνησης.

    Η εμβέλεια σταματάει στο επόμενο σημείο στίξης. Η ίδια η λέξη άρνησης δεν
    παίρνει πρόθεμα. Η λίστα που γυρίζει έχει το ίδιο μήκος με την αρχική,
    ώστε να μένουν ευθυγραμμισμένα τα βάρη.
    """
    result = []
    active = False

    for token in tokens:
        if token in NEGATION_SET:
            # η λέξη άρνησης μένει ως έχει και ανοίγει την εμβέλεια
            result.append(token)
            active = True
        elif is_punctuation(token):
            # η στίξη κλείνει την εμβέλεια, γι' αυτό δεν την έχουμε πετάξει ακόμα
            result.append(token)
            active = False
        elif active:
            result.append('NOT_' + token)
        else:
            result.append(token)

    return result


def preprocess(sentences, remove_punct=True, remove_stop=False,
               keep_negations=False, stem=False,
               negation=False, pos_weight=False, position_weight=False):
    """Εφαρμόζει προεπεξεργασία σε ένα έγγραφο και επιστρέφει λίστα tokens.

    Το έγγραφο μπαίνει ως λίστα προτάσεων και βγαίνει ως μία ενιαία λίστα
    tokens, γιατί αυτό θέλουν οι vectorizers. Ενδιάμεσα κάθε token κρατιέται
    ως ζευγάρι (token, βάρος), ώστε όταν πετάμε στίξη ή stopwords να φεύγει
    και το βάρος μαζί του χωρίς να χρειάζεται συγχρονισμός.

    Η σειρά των βημάτων δεν αλλάζει: θέση -> POS -> άρνηση -> στίξη ->
    stopwords -> stemming -> βάρη.
    """
    pairs = []
    last = len(sentences) - 1

    for i in range(len(sentences)):
        sentence = sentences[i]

        # 1. σήμανση θέσης: 1η και τελευταία πρόταση του εγγράφου
        weights = []
        for token in sentence:
            if position_weight and (i == 0 or i == last):
                weights.append(2)
            else:
                weights.append(1)

        # 2. POS tagging στα ΑΚΑΤΕΡΓΑΣΤΑ tokens, πριν από άρνηση και stemming
        if pos_weight:
            tagged = nltk.pos_tag(sentence)
            for j in range(len(sentence)):
                tag = tagged[j][1]
                # επίθετα (JJ) και επιρρήματα (RB) κουβαλάνε το συναίσθημα
                if tag.startswith('JJ') or tag.startswith('RB'):
                    # στο B5_all τα βάρη πολλαπλασιάζονται, γι' αυτό * 2
                    weights[j] = weights[j] * 2

        # 3. διαχείριση άρνησης, όσο υπάρχει ακόμα η στίξη
        tokens = sentence
        if negation:
            tokens = apply_negation(sentence)

        for j in range(len(tokens)):
            pairs.append((tokens[j], weights[j]))

    # 4. αφαίρεση στίξης
    if remove_punct:
        kept = []
        for token, weight in pairs:
            if not is_punctuation(token):
                kept.append((token, weight))
        pairs = kept

    # 5. αφαίρεση stopwords
    if remove_stop:
        # με keep_negations κρατάμε τα no/not/nor κτλ, τα θέλει το Βα
        if keep_negations:
            stoplist = STOPWORDS_KEEPNEG
        else:
            stoplist = STOPWORDS

        kept = []
        for token, weight in pairs:
            # τα NOT_ tokens δεν είναι ποτέ στη stoplist, οπότε επιβιώνουν
            if token not in stoplist:
                kept.append((token, weight))
        pairs = kept

    # 6. stemming
    if stem:
        stemmed = []
        for token, weight in pairs:
            stemmed.append((STEMMER.stem(token), weight))
        pairs = stemmed

    # 7. εφαρμογή βαρών: βάρος 2 σημαίνει ότι το token μπαίνει δύο φορές
    tokens = []
    for token, weight in pairs:
        for k in range(weight):
            tokens.append(token)

    return tokens


# Οι 8 διαμορφώσεις του μέρους Α. Το pruning δεν είναι προεπεξεργασία, είναι
# παράμετρος του vectorizer (ή min_count στο Word2Vec), γι' αυτό μπαίνει χωριστά.
PART_A_CONFIGS = [
    ('A0_raw', {'remove_punct': False}, False),
    ('A1_nopunct', {}, False),
    ('A2_stop', {'remove_stop': True}, False),
    ('A3_stop_keepneg', {'remove_stop': True, 'keep_negations': True}, False),
    ('A4_stem', {'stem': True}, False),
    ('A5_stop_stem', {'remove_stop': True, 'keep_negations': True, 'stem': True}, False),
    ('A6_full', {'remove_stop': True, 'keep_negations': True, 'stem': True}, True),
    ('A7_prune', {}, True),
]


def split_sentences(lines):
    """Ενώνει τις γραμμές που έχουν κοπεί στη μέση.

    Στο dataset υπάρχει μία πρόταση ανά γραμμή, αλλά ~3000 γραμμές κόβονται
    από αποσιωπητικά. Οπότε όποια γραμμή δεν τελειώνει σε . ! ? την κολλάμε
    με την επόμενη.
    """
    sentences = []
    current = []

    for line in lines:
        tokens = line.split()
        if len(tokens) == 0:
            continue

        # μαζεύουμε τα tokens της τρέχουσας πρότασης
        for token in tokens:
            current.append(token)

        # τελειώνει η πρόταση μόνο αν το τελευταίο token είναι τελικό σημείο στίξης
        if tokens[-1] in ('.', '!', '?'):
            sentences.append(current)
            current = []

    # αν το αρχείο τελειώνει χωρίς στίξη, κρατάμε ό,τι έμεινε
    if len(current) > 0:
        sentences.append(current)

    return sentences


def load_file(path):
    """Διαβάζει ένα αρχείο και το επιστρέφει ως λίστα προτάσεων."""
    f = open(path, encoding='utf-8')
    lines = f.readlines()
    f.close()
    return split_sentences(lines)


def load_data(folder='data/txt_sentoken', limit=None):
    """Φορτώνει το dataset.

    Επιστρέφει docs (λίστα εγγράφων, κάθε έγγραφο λίστα προτάσεων, κάθε
    πρόταση λίστα tokens) και labels (1 = pos, 0 = neg).
    Το limit είναι για γρήγορες δοκιμές σε λίγα έγγραφα.
    """
    docs = []
    labels = []

    for name, label in [('pos', 1), ('neg', 0)]:
        path = os.path.join(folder, name)
        # ταξινομημένα ονόματα για να βγαίνουν πάντα τα ίδια αποτελέσματα
        files = sorted(os.listdir(path))

        if limit is not None:
            files = files[:limit]

        for filename in files:
            sentences = load_file(os.path.join(path, filename))
            docs.append(sentences)
            labels.append(label)

        print('Φορτώθηκαν', len(files), 'αρχεία από', name)

    return docs, labels


if __name__ == '__main__':
    # δοκιμή σε 100 έγγραφα (50 pos + 50 neg) πριν τρέξουμε όλο το dataset
    docs, labels = load_data(limit=50)

    print()
    print('Έγγραφα:', len(docs), '| θετικά:', labels.count(1), '| αρνητικά:', labels.count(0))
    print()

    for name, flags, prune in PART_A_CONFIGS:
        total_tokens = 0
        vocabulary = set()

        for doc in docs:
            tokens = preprocess(doc, **flags)
            total_tokens = total_tokens + len(tokens)
            for token in tokens:
                vocabulary.add(token)

        print(name, '-> tokens:', total_tokens, '| λεξιλόγιο:', len(vocabulary),
              '| pruning:', prune)
