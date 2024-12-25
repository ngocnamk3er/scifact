import argparse
import jsonlines

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
label_prediction = jsonlines.open(args.label_prediction)

pred_labels = []
true_labels = []

LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

for data, prediction in zip(dataset, label_prediction):
    assert data['id'] == prediction['claim_id']

    if args.filter:
        prediction['labels'] = {doc_id: pred for doc_id, pred in prediction['labels'].items()
                                if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
    if not prediction['labels']:
        continue

    claim_id = data['id']
    for doc_id, pred in prediction['labels'].items():
        pred_label = pred['label']
        true_label = {es['label'] for es in data['evidence'].get(doc_id) or []}
        assert len(true_label) <= 1, 'Currently support only one label per doc'
        true_label = next(iter(true_label)) if true_label else 'NOT_ENOUGH_INFO'
        pred_labels.append(LABELS[pred_label])
        true_labels.append(LABELS[true_label])

accuracy = round(sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels), 4)
macro_f1 = round(f1_score(true_labels, pred_labels, average="macro"), 4)
macro_f1_wo_nei = round(f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]), 4)

print(f'Accuracy           {accuracy}')
print(f'Macro F1:          {macro_f1}')
print(f'Macro F1 w/o NEI:  {macro_f1_wo_nei}')
print()

# F1, Precision, and Recall for each class
f1_scores = f1_score(true_labels, pred_labels, average=None)
precision_scores = precision_score(true_labels, pred_labels, average=None)
recall_scores = recall_score(true_labels, pred_labels, average=None)

print(f'F1:                {", ".join([str(round(x, 4)) for x in f1_scores])}')
print(f'Precision:         {", ".join([str(round(x, 4)) for x in precision_scores])}')
print(f'Recall:            {", ".join([str(round(x, 4)) for x in recall_scores])}')
print()

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
print('Confusion Matrix:')
print(conf_matrix)
