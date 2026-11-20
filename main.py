import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
# Agg γιατί τα γραφήματα σώζονται σε αρχείο, δεν ανοίγει παράθυρο.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from preprocess import load_data, preprocess, PART_A_CONFIGS
from models import COMBOS, identity, build_classifier
from experiments import run_cv, average, print_summary


CSV_COLUMNS = ['part', 'config', 'combo', 'fold', 'accuracy',
               'precision_macro', 'recall_macro', 'f1_macro',
               'tn', 'fp', 'fn', 'tp']


# Τα 5 σενάρια του μέρους Β. Το B1 δεν προσθέτει τίποτα, είναι το baseline.
PART_B_SCENARIOS = [
    ('B1_baseline', {}),
    ('B2_negation', {'negation': True}),
    ('B3_pos', {'pos_weight': True}),
    ('B4_position', {'position_weight': True}),
    ('B5_all', {'negation': True, 'pos_weight': True, 'position_weight': True}),
]


def write_csv(path, rows):
    """Γράφει τις γραμμές των πειραμάτων στο csv."""
    f = open(path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    f.close()
    print('Γράφτηκε το', path, 'με', len(rows), 'γραμμές')


def elapsed(start):
    """Πόση ώρα πέρασε από το start, σε λεπτά και δευτερόλεπτα."""
    seconds = time.time() - start
    minutes = int(seconds / 60)
    return str(minutes) + 'λ ' + str(int(seconds - minutes * 60)) + 'δ'


def run_part_a(docs, labels, start_all):
    """Τρέχει τις 8 διαμορφώσεις για τους 5 συνδυασμούς (40 πειράματα)."""
    all_rows = []
    # scores[combo][config] = ο μέσος macro-F1, για να βρούμε τα baselines
    scores = {}
    for combo in COMBOS:
        scores[combo] = {}

    total = len(PART_A_CONFIGS) * len(COMBOS)
    done = 0
    config_number = 0

    for config_name, flags, prune in PART_A_CONFIGS:
        config_number = config_number + 1

        print()
        print('=== ΜΕΡΟΣ Α | διαμόρφωση', config_number, 'από',
              len(PART_A_CONFIGS), ':', config_name, '===')
        print('    flags:', flags, '| pruning:', prune)

        # η προεπεξεργασία γίνεται μία φορά, έξω από το loop των folds
        print('    προεπεξεργασία', len(docs), 'εγγράφων...')
        start = time.time()
        processed = []
        for doc in docs:
            processed.append(preprocess(doc, **flags))
        print('    ολοκληρώθηκε σε', round(time.time() - start, 1), 'δευτερόλεπτα')

        for combo in COMBOS:
            done = done + 1
            print('  [' + str(done) + '/' + str(total) + ']', combo, '...')

            start = time.time()
            rows, total_cm = run_cv(processed, labels, combo, prune)

            for row in rows:
                row['part'] = 'A'
                row['config'] = config_name
                all_rows.append(row)

            scores[combo][config_name] = average(rows, 'f1_macro')

            print_summary('  ' + combo, rows, total_cm)
            print('   χρόνος:', round(time.time() - start, 1), 'δευτερόλεπτα',
                  '| από την αρχή:', elapsed(start_all))

    return all_rows, scores


def read_f1_from_csv(path):
    """Διαβάζει ένα csv πειραμάτων και μαζεύει τα macro-F1 των folds.

    Επιστρέφει values[config][combo] = λίστα με τα 5 macro-F1.
    """
    values = {}

    f = open(path, newline='', encoding='utf-8')
    reader = csv.DictReader(f)
    for row in reader:
        config = row['config']
        combo = row['combo']

        if config not in values:
            values[config] = {}
        if combo not in values[config]:
            values[config][combo] = []

        values[config][combo].append(float(row['f1_macro']))
    f.close()

    return values


def make_figure(values, config_names, title, out_path):
    """Ομαδοποιημένο ραβδόγραμμα macro-F1: διαμορφώσεις στον x, συνδυασμοί ως ομάδες."""
    width = 0.15
    combo_number = 0

    plt.figure(figsize=(12, 6))

    for combo in COMBOS:
        means = []
        stds = []
        positions = []
        config_number = 0

        for config in config_names:
            folds = values[config][combo]
            means.append(np.mean(folds))
            # η μπάρα σφάλματος είναι η τυπική απόκλιση στα 5 folds
            stds.append(np.std(folds))
            # κάθε συνδυασμός μετατοπίζεται λίγο δεξιά μέσα στην ομάδα
            positions.append(config_number + combo_number * width)
            config_number = config_number + 1

        plt.bar(positions, means, width, yerr=stds, capsize=3, label=combo)
        combo_number = combo_number + 1

    # τα ticks στο κέντρο της ομάδας, δηλαδή στην 3η από τις 5 μπάρες
    ticks = []
    for i in range(len(config_names)):
        ticks.append(i + 2 * width)

    plt.xticks(ticks, config_names, rotation=20)
    # στενό εύρος στον y για να ξεχωρίζουν οι διαφορές
    plt.ylim(0.5, 0.9)
    plt.ylabel('macro-F1')
    plt.title(title)
    # η λεζάντα έξω από το γράφημα, αλλιώς κρύβει τις τελευταίες μπάρες
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()

    print('Γράφτηκε το', out_path)


def read_confusion_from_csv(path, config_name):
    """Αθροίζει τα confusion matrices των folds, ανά συνδυασμό, για ένα σενάριο."""
    # matrices[combo] = [tn, fp, fn, tp]
    matrices = {}
    for combo in COMBOS:
        matrices[combo] = [0, 0, 0, 0]

    f = open(path, newline='', encoding='utf-8')
    reader = csv.DictReader(f)
    for row in reader:
        if row['config'] == config_name:
            combo = row['combo']
            matrices[combo][0] = matrices[combo][0] + int(row['tn'])
            matrices[combo][1] = matrices[combo][1] + int(row['fp'])
            matrices[combo][2] = matrices[combo][2] + int(row['fn'])
            matrices[combo][3] = matrices[combo][3] + int(row['tp'])
    f.close()

    return matrices


def make_confusion_figure(matrices, config_name, out_path):
    """(α) Heatmaps των αθροιστικών confusion matrices των 5 συνδυασμών."""
    fig, axes = plt.subplots(1, len(COMBOS), figsize=(16, 4))
    combo_number = 0

    for combo in COMBOS:
        tn, fp, fn, tp = matrices[combo]
        # σειρές = πραγματική κλάση, στήλες = πρόβλεψη
        cm = np.array([[tn, fp], [fn, tp]])

        ax = axes[combo_number]
        # vmin=0 ώστε το χρώμα να ξεκινάει από το μηδέν και να ταιριάζει με
        # το όριο που διαλέγει παρακάτω άσπρα ή μαύρα γράμματα
        ax.imshow(cm, cmap='Blues', vmin=0)

        # ο αριθμός μέσα σε κάθε κελί, άσπρος στα σκούρα κελιά για να διαβάζεται
        for i in range(2):
            for j in range(2):
                if cm[i][j] > cm.max() / 2:
                    color = 'white'
                else:
                    color = 'black'
                ax.text(j, i, cm[i][j], ha='center', va='center', color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['πρόβλ. neg', 'πρόβλ. pos'], rotation=20)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['neg', 'pos'])
        ax.set_title(combo, fontsize=10)

        combo_number = combo_number + 1

    fig.suptitle('Αθροιστικά confusion matrices στα 5 folds — ' + config_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print('Γράφτηκε το', out_path)


def make_difference_figure(values, scenario_names, out_path):
    """(β) Διαφορά κάθε σεναρίου από το B1_baseline, σε μονάδες macro-F1."""
    width = 0.15
    combo_number = 0

    plt.figure(figsize=(11, 6))

    for combo in COMBOS:
        baseline = np.mean(values['B1_baseline'][combo])

        diffs = []
        positions = []
        scenario_number = 0

        for scenario in scenario_names:
            diffs.append(np.mean(values[scenario][combo]) - baseline)
            positions.append(scenario_number + combo_number * width)
            scenario_number = scenario_number + 1

        plt.bar(positions, diffs, width, label=combo)
        combo_number = combo_number + 1

    ticks = []
    for i in range(len(scenario_names)):
        ticks.append(i + 2 * width)

    # η γραμμή στο μηδέν δείχνει πού είναι το baseline
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(ticks, scenario_names)
    plt.ylabel('διαφορά macro-F1 από το B1_baseline')
    plt.title('Μέρος Β: κέρδος ή ζημιά κάθε σεναρίου σε σχέση με το baseline')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()

    print('Γράφτηκε το', out_path)


def make_boxplot_figure(values, config_names, combo, out_path):
    """(γ) Κατανομή του macro-F1 στα 5 folds ενός συνδυασμού, ανά διαμόρφωση."""
    data = []
    for config in config_names:
        data.append(values[config][combo])

    plt.figure(figsize=(11, 6))
    plt.boxplot(data)

    # τα ticks του boxplot ξεκινούν από το 1, όχι από το 0
    positions = []
    for i in range(len(config_names)):
        positions.append(i + 1)

    plt.xticks(positions, config_names, rotation=20)
    plt.ylabel('macro-F1')
    plt.title('Μέρος Α: κατανομή macro-F1 στα 5 folds — ' + combo)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()

    print('Γράφτηκε το', out_path)


def make_top_words_figure(docs, labels, baselines, min_df, out_path):
    """(δ) Οι 15 λέξεις με τη μεγαλύτερη διαφορά log-πιθανότητας ανά κλάση.

    Εκπαιδεύει ένα MultinomialNB με counts σε ΟΛΟ το dataset, με την
    προεπεξεργασία του σεναρίου B2_negation. Δεν είναι αποτίμηση, είναι μόνο
    για να δούμε τι έμαθε το μοντέλο, οπότε δεν χρειάζεται train/test split.

    Το min_df κόβει τις σπάνιες λέξεις. Με min_df=1 βγαίνουν σχεδόν μόνο
    τίτλοι ταινιών και ονόματα ηθοποιών: εμφανίζονται σε ελάχιστες κριτικές
    μίας μόνο κλάσης, οπότε παίρνουν τον μεγαλύτερο λόγο πιθανοτήτων χωρίς
    να λένε κάτι για το συναίσθημα.
    """
    config_name = baselines['MNB_COUNTS']['config']
    # το pruning του baseline δεν μας ενδιαφέρει εδώ, το min_df το δίνουμε εμείς
    base_flags, prune = find_config(config_name)

    flags = {}
    for key in base_flags:
        flags[key] = base_flags[key]
    flags['negation'] = True

    print('   εκπαίδευση MultinomialNB σε όλο το dataset (' + config_name +
          ' + άρνηση, min_df=' + str(min_df) + ')...')

    processed = []
    for doc in docs:
        processed.append(preprocess(doc, **flags))

    vec = CountVectorizer(analyzer=identity, min_df=min_df)
    X = vec.fit_transform(processed)

    clf = build_classifier('MNB_COUNTS')
    clf.fit(X, labels)

    words = vec.get_feature_names_out()
    # classes_ = [0, 1], δηλαδή [neg, pos]. Θετική διαφορά = λέξη υπέρ του pos.
    diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    order = np.argsort(diff)
    chosen = list(order[:15]) + list(order[-15:])

    chosen_words = []
    chosen_values = []
    colors = []
    for index in chosen:
        chosen_words.append(words[index])
        chosen_values.append(diff[index])
        if diff[index] > 0:
            colors.append('tab:blue')
        else:
            colors.append('tab:red')

    positions = []
    for i in range(len(chosen_words)):
        positions.append(i)

    plt.figure(figsize=(9, 9))
    plt.barh(positions, chosen_values, color=colors)
    plt.yticks(positions, chosen_words)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('log P(λέξη | pos) - log P(λέξη | neg)')
    plt.title('B2_negation, min_df=' + str(min_df) +
              ': 15 λέξεις υπέρ neg (κόκκινο) και 15 υπέρ pos (μπλε)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()

    print('Γράφτηκε το', out_path)


def make_plots():
    """Φτιάχνει τα δύο γραφήματα διαβάζοντας τα csv που έχουν ήδη γραφτεί."""
    if not os.path.exists('results/figures'):
        os.mkdir('results/figures')

    config_names = []
    for name, flags, prune in PART_A_CONFIGS:
        config_names.append(name)

    values_a = read_f1_from_csv('results/part_a.csv')
    make_figure(values_a, config_names,
                'Μέρος Α: macro-F1 ανά διαμόρφωση προεπεξεργασίας',
                'results/figures/part_a.png')

    scenario_names = []
    for name, flags in PART_B_SCENARIOS:
        scenario_names.append(name)

    values_b = read_f1_from_csv('results/part_b.csv')
    make_figure(values_b, scenario_names,
                'Μέρος Β: macro-F1 ανά σενάριο',
                'results/figures/part_b.png')

    matrices = read_confusion_from_csv('results/part_b.csv', 'B5_all')
    make_confusion_figure(matrices, 'B5_all',
                          'results/figures/confusion_b5_all.png')

    # η σύγκριση γίνεται με το B1_baseline, οπότε αυτό δεν μπαίνει στον άξονα
    make_difference_figure(values_b, scenario_names[1:],
                           'results/figures/part_b_diff.png')

    make_boxplot_figure(values_a, config_names, 'MNB_TFIDF',
                        'results/figures/folds_mnb_tfidf.png')


def find_config(config_name):
    """Βρίσκει τα flags και το pruning μιας διαμόρφωσης του μέρους Α."""
    for name, flags, prune in PART_A_CONFIGS:
        if name == config_name:
            return flags, prune


def run_part_b(docs, labels, baselines, start_all):
    """Τρέχει τα 5 σενάρια, το καθένα πάνω στο baseline κάθε συνδυασμού."""
    all_rows = []

    total = len(PART_B_SCENARIOS) * len(COMBOS)
    done = 0
    scenario_number = 0

    for scenario_name, scenario_flags in PART_B_SCENARIOS:
        scenario_number = scenario_number + 1

        print()
        print('=== ΜΕΡΟΣ Β | σενάριο', scenario_number, 'από',
              len(PART_B_SCENARIOS), ':', scenario_name, '===')
        print('    τεχνικές:', scenario_flags)

        for combo in COMBOS:
            done = done + 1

            # κάθε συνδυασμός κρατάει τη δική του καλύτερη διαμόρφωση από το Α
            config_name = baselines[combo]['config']
            base_flags, prune = find_config(config_name)

            # τα flags του baseline μαζί με τα flags του σεναρίου
            flags = {}
            for key in base_flags:
                flags[key] = base_flags[key]
            for key in scenario_flags:
                flags[key] = scenario_flags[key]

            print('  [' + str(done) + '/' + str(total) + ']', combo,
                  'πάνω στο', config_name, '...')

            start = time.time()

            # εδώ η προεπεξεργασία γίνεται ανά συνδυασμό, γιατί κάθε
            # συνδυασμός έχει διαφορετικό baseline από το μέρος Α
            processed = []
            for doc in docs:
                processed.append(preprocess(doc, **flags))

            rows, total_cm = run_cv(processed, labels, combo, prune)

            for row in rows:
                row['part'] = 'B'
                row['config'] = scenario_name
                all_rows.append(row)

            print_summary('  ' + combo + ' (' + config_name + ')', rows, total_cm)
            print('   χρόνος:', round(time.time() - start, 1), 'δευτερόλεπτα',
                  '| από την αρχή:', elapsed(start_all))

    return all_rows


def pick_baselines(scores):
    """Διαλέγει τη διαμόρφωση με τον μεγαλύτερο macro-F1 για κάθε συνδυασμό."""
    baselines = {}

    for combo in COMBOS:
        best_config = None
        best_score = -1

        for config_name, flags, prune in PART_A_CONFIGS:
            score = scores[combo][config_name]
            if score > best_score:
                best_score = score
                best_config = config_name

        baselines[combo] = {'config': best_config, 'f1_macro': best_score}

    return baselines


if __name__ == '__main__':
    start_all = time.time()

    print('Φόρτωση δεδομένων...')
    docs, labels = load_data()
    print('Σύνολο:', len(docs), 'έγγραφα |', labels.count(1), 'θετικά |',
          labels.count(0), 'αρνητικά')

    if not os.path.exists('results'):
        os.mkdir('results')

    print()
    print('##### ΜΕΡΟΣ Α: 8 διαμορφώσεις x 5 συνδυασμοί = 40 πειράματα #####')
    rows_a, scores = run_part_a(docs, labels, start_all)

    print()
    write_csv('results/part_a.csv', rows_a)

    baselines = pick_baselines(scores)
    f = open('results/baseline_choices.json', 'w', encoding='utf-8')
    json.dump(baselines, f, indent=2, ensure_ascii=False)
    f.close()
    print('Γράφτηκε το results/baseline_choices.json')

    print()
    print('Baselines για το μέρος Β:')
    for combo in COMBOS:
        print('  ', combo, '->', baselines[combo]['config'],
              '(macro-F1', round(baselines[combo]['f1_macro'], 4), ')')

    print()
    print('##### ΜΕΡΟΣ Β: 5 σενάρια x 5 συνδυασμοί = 25 πειράματα #####')
    rows_b = run_part_b(docs, labels, baselines, start_all)

    print()
    write_csv('results/part_b.csv', rows_b)

    print()
    print('Γραφήματα...')
    make_plots()
    # δύο εκδοχές: με φίλτρο σπάνιων λέξεων και χωρίς, για σύγκριση
    make_top_words_figure(docs, labels, baselines, 20,
                          'results/figures/top_words.png')
    make_top_words_figure(docs, labels, baselines, 1,
                          'results/figures/top_words_nofilter.png')

    print()
    print('Τέλος. Συνολικός χρόνος:', elapsed(start_all))
