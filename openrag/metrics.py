import numpy as np
import string
import re
from collections import Counter
import re


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))


def f1(decoded_preds, decoded_labels):
    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt) for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    return 100 * np.mean(f1_all)


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def hotpot_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def hotpot_exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_entity_tags(sentence):
    entity_regex = r"(.+?)(?=\s<|$)"
    tag_regex = r"<(.+?)>"
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)

    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results


def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0


import unicodedata

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')



def match_qnorm(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = [normalize_answer(gt) for gt in ground_truth]
    match = 0
    for gt in ground_truth:
        if gt in prediction:
            match = 1
    
    if type(ground_truth) == list:
        if len(ground_truth) == 0:
            f1 = 0
        else:
            

            f1 = np.max([qa_f1_score(prediction, gt) for gt in ground_truth])
    else:
        f1 = qa_f1_score(prediction, ground_truth)
    
    if type(ground_truth) == list:
        if len(ground_truth) == 0:
            em = 0
        else:
            em = np.max([int(prediction == gt) for gt in ground_truth])
    else:
        em = int(prediction == gt)

    return match, f1, em


def match_qnorm_hotpot(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = [normalize_answer(gt) for gt in ground_truth]
    match = 0
    for gt in ground_truth:
        if gt in prediction:
            match = 1
    
    if type(ground_truth) == list:
        if len(ground_truth) == 0:
            f1 = 0
        else:
            f1 = np.max([hotpot_f1_score(prediction, gt)[0] for gt in ground_truth])
    else:
        f1 = hotpot_f1_score(prediction, ground_truth)[0]
    
    if type(ground_truth) == list:
        if len(ground_truth) == 0:
            em = 0
        else:
            em = np.max([int(prediction == gt) for gt in ground_truth])
    else:
        em = int(prediction == gt)

    return match, f1, em