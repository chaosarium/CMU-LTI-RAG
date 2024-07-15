from collections import Counter
import pandas as pd
import pickle

def load_annotated_qas(qa_jsons_paths: list[str], is_augmented):
    if is_augmented:
        qa_jsons_paths = list(map(lambda s: s.replace(".json", "_aug.json"), qa_jsons_paths))
    df_acc = []
    for filepath in qa_jsons_paths:
        df = pd.read_json(filepath, orient='records')
        df['filepath'] = filepath
        df_acc.append(df)
    
    dfs = pd.concat(df_acc).reset_index()
    return dfs.to_dict(orient='records')

# The below evaluation metrics are based on ATLAS (https://github.com/facebookresearch/atlas/blob/main/src/evaluation.py). ATLAS is under CC BY-NC 4.0
from lib.reference_evaluation import ATLASEvaluation

def eval_exact_match(prediction: str, ground_truths: list[str]) -> float:
    return ATLASEvaluation.exact_match_score(prediction, ground_truths, ATLASEvaluation.normalize_answer)

# supposed to be macro-averaged F1
def eval_f1(prediction: str, ground_truths: list[str]) -> float:
    return ATLASEvaluation.f1_score(prediction, ground_truths, ATLASEvaluation.normalize_answer)

def eval_rouge(prediction: str, ground_truths: list[str]) -> tuple[float, float, float]:
    return ATLASEvaluation.rouge_score(prediction, ground_truths)

# based on ATLAS's F1 implementation
def eval_precision_recall(prediction, ground_truth) -> tuple[float, float]:
    prediction_tokens = ATLASEvaluation.normalize_answer(prediction).split()
    ground_truth_tokens = ATLASEvaluation.normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return precision, recall

def eval_recall(prediction, ground_truths) -> float:
    return max([eval_precision_recall(prediction, gt)[1] for gt in ground_truths])

def eval_precision(prediction, ground_truths) -> float:
    return max([eval_precision_recall(prediction, gt)[0] for gt in ground_truths])