from sklearn.metrics import f1_score
import evaluate


def compute_bleu(prediction, reference):
    bleu_metric = evaluate.load("bleu")
    result_bleu = bleu_metric.compute(predictions=prediction, references=[[ref] for ref in reference])
    print("BLEU:", result_bleu['bleu'])
    print("BLEU precisions:", result_bleu['precisions'])
    
    return result_bleu


def compute_rouge(prediction, reference):
    rouge_metric = evaluate.load("rouge")
    result_rouge = rouge_metric.compute(predictions=prediction, references=reference)
    print('Rouge R1': result_rouge['rouge1'])
    print('Rouge R2': result_rouge['rouge2'])
    print('Rouge L': result_rouge['rougeL'])
    return result_rouge


def f1(pred, target):
      return f1_score(target, pred, average='weighted')

def recall(pred, target):
    # pred = staff is negative; product is positive
    # how many pairs "CATEGORY is SENTIMENT" from gold label are generated as prediction
    
    pred = pred.split(';')
    target = target.split(';')
    
    pred = [p.strip() for p in pred]
    target = [p.strip() for p in target]

    sum = 0
    already_seen = []
    for p in pred:
        if p in target and p not in already_seen:
            sum += 1
            already_seen.append(p)
    sum=sum/(len(target))
    return sum


def precision(pred, target):
    # pred = staff is negative; product is positive
    # how many predicted pairs "CATEGORY is SENTIMENT" are correct 
    pred = pred.split(';')  
    target = target.split(';')
    
    pred = [p.strip() for p in pred]
    target = [p.strip() for p in target]

    correct = 0
    already_seen = []
    for p in pred:
        if p in target and p not in already_seen:
            correct += 1
            already_seen.append(p)
    
    return correct / len(pred) if len(pred) > 0 else 0


def label_f1(precision, recall):
    # F1 instance level
    if precision + recall == 0:
        return 0.0  
    return 2 * (precision * recall) / (precision + recall)



"""
    wandb.log({
        'Test Rouge R1': result_rouge['rouge1'],
        'Test Rouge R2': result_rouge['rouge2'],
        'Test Rouge L': result_rouge['rougeL'],
        'Test Bleu': result_bleu['bleu'],
        'Test Bleu precisions': np.mean(result_bleu['precisions']), 
        'Test F1': result_f1,
        'Test Recall': result_recall,
        'Test Precision': result_precision,
        'Test F1 instance level': result_f1_instance_level
    })
"""