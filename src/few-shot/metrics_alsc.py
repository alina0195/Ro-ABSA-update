import evaluate
from sklearn.metrics import f1_score
from typing import List

def evaluate_predictions(pred_labels_list: List[str]
		       , true_labels_list: List[str]) -> dict:


	exact_match = evaluate.load("exact_match")

	f1_scores = []
	em_scores = []
	
	for pred, true in zip(pred_labels_list, true_labels_list):
		f1_val = f1_score([true], [pred], average='weighted')
		em_val = exact_match.compute(predictions=[pred], references=[true])['exact_match']
  		f1_scores.append(f1_val)
  		em_scores.append(em)

	return {
		'f1_scores': f1_scores,
		'em_scores': em_scores,
		'average_f1': sum(f1_scores) / len(f1_scores),
		'average_em': sum(em_scores) / len(em_scores)
		}