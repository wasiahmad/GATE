"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import numpy as np
from collections import Counter
from clie.inputters import constant

NO_RELATION = constant.NEGATIVE_LABEL


def score(results, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    confusion_stat = dict()
    unique_labels = set()
    for row in range(len(results)):
        gold = results[row]['gold']
        guess = results[row]['pred']
        unique_labels.update([gold, guess])

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

        if gold not in confusion_stat:
            confusion_stat[gold] = dict()
        if guess not in confusion_stat[gold]:
            confusion_stat[gold][guess] = 0
        confusion_stat[gold][guess] += 1

    unique_labels = sorted(list(unique_labels))
    num_labels = len(unique_labels)
    confusion_matrix = np.zeros((num_labels, num_labels))
    for i, gold in enumerate(unique_labels):
        if gold in confusion_stat:
            for j, guess in enumerate(unique_labels):
                if guess in confusion_stat[gold]:
                    confusion_matrix[i, j] = confusion_stat[gold][guess]

    # Print verbose information
    verbose_out = ""
    if verbose:
        verbose_out = "Per-relation statistics:\n"
        relations = gold_by_relation.keys()
        longest_relation = 0
        # sort of a list of strings and find the maximum length (used for pretty printing)
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            verbose_out += ("{:<" + str(longest_relation) + "}").format(relation)
            verbose_out += "  P: "
            if prec < 0.1: verbose_out += ' '
            if prec < 1.0: verbose_out += ' '
            verbose_out += "{:.2%}".format(prec)
            verbose_out += "  R: "
            if recall < 0.1: verbose_out += ' '
            if recall < 1.0: verbose_out += ' '
            verbose_out += "{:.2%}".format(recall)
            verbose_out += "  F1: "
            if f1 < 0.1: verbose_out += ' '
            if f1 < 1.0: verbose_out += ' '
            verbose_out += "{:.2%}".format(f1)
            verbose_out += "  #: %d" % gold
            verbose_out += '\n'

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    return {
        'precision': prec_micro,
        'recall': recall_micro,
        'f1': f1_micro,
        'verbose_out': verbose_out,
        'confusion_matrix': confusion_matrix,
        'labels': unique_labels
    }
