import pickle
import glob
import numpy as np
import copy 

from factory.evaluate.metrics import avg_recall

pickles = np.sort(glob.glob('../cv-predictions/ensemble_007/*/*.pkl'))
#pickles = glob.glob('../cv-predictions/experiment063/*.pkl')
# pickles6 = np.sort(glob.glob('../cv-predictions/ensemble_006/*/*.pkl'))[2:]
# pickles = np.concatenate((pickles, pickles6))
pickles = list(pickles)
#pickles.append('../cv-predictions/experiment053/predictions.pkl')

predictions = []
for p in pickles:
    with open(p, 'rb') as f:
        predictions.append(pickle.load(f))

avg_recall_list = []
for p in predictions:
    avg_recall_list.append(avg_recall(p['y_true'], p['y_pred']))

for ind, p in enumerate(pickles):
    print('{} : {:.4f} {:.4f} {:.4f} {:.4f}'.format(p.split('/')[-2], 
          avg_recall_list[ind]['avg_recall'],
          avg_recall_list[ind]['gra_recall'],
          avg_recall_list[ind]['con_recall'],
          avg_recall_list[ind]['vow_recall']))

def average_predictions(predictions, weights=None):
    if weights is None:
        weights = [1.] * len(predictions)
    weights = np.asarray(weights)
    weights /= weights.sum()
    averaged = copy.deepcopy(predictions[0])
    for k,v in averaged['y_pred'].items():
        averaged['y_pred'][k] = averaged['y_pred'][k]*weights[0]
    for ind, p in enumerate(predictions[1:]):
        for k,v in p['y_pred'].items():
            averaged['y_pred'][k] += v * weights[ind+1]
    # for k,v in averaged['y_pred'].items():
    #     averaged['y_pred'][k] /= len(predictions)
    return averaged

all_ensemble = average_predictions(predictions)
print ('ENSEMBLE [ALL]: {:.4f}'.format(avg_recall(all_ensemble['y_true'], all_ensemble['y_pred'])['avg_recall']))

effb4_ensemble = average_predictions(np.asarray(predictions)[[-2,-1]])
effb4_recalls = avg_recall(effb4_ensemble['y_true'], effb4_ensemble['y_pred'])
print ('ENSEMBLE [EFFB4]: {:.4f} {:.4f} {:.4f} {:.4f}'.format(
    effb4_recalls['avg_recall'], effb4_recalls['gra_recall'],
    effb4_recalls['con_recall'], effb4_recalls['vow_recall']
))

avg_recall(predictions[0]['y_true'], predictions[0]['y_pred'])['avg_recall']


# Analyze grapheme_root more closely
from sklearn.metrics import recall_score

x = predictions[4]['y_pred']['grapheme_root']
y = predictions[4]['y_true']['grapheme_root']

scores = []
for i in range(x.shape[-1]):
    scores.append(recall_score((y==i).astype('float'), (np.argmax(x, axis=1)==i).astype('float')))

sorted_worst = np.argsort(scores)

worst = x[x['']]

# Experiment with Level 2 classifier ...
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 

x = np.concatenate((predictions[4]['y_pred']['grapheme_root'], 
                    predictions[4]['y_pred']['vowel_diacritic'], 
                    predictions[4]['y_pred']['consonant_diacritic']), axis=1)
y = predictions[4]['y_true']['grapheme_root']

train_indices = np.random.choice(range(len(x)), 9000, replace=False)
test_indices  = np.asarray(list(set(range(len(x))) - set(train_indices)))

x_train, y_train = x[train_indices], y[train_indices]
x_test,  y_test  = x[test_indices],  y[test_indices]

recall_score(y_test, np.argmax(x_test[:,:168], axis=1), average='macro')

class Model(nn.Module):
    #
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(186, 186)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(186, 168) 
    #
    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x)))

class SimpleDataset(torch.utils.data.Dataset):
    #
    def __init__(self, x,y):
        super(SimpleDataset, self).__init__()
        self.x = x
        self.y = y
    #
    def __len__(self):
        return len(self.x)
    #
    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]
        return x, y


model = Model()
model.train()
_, counts = np.unique(y_train, return_counts=True)
freqs = counts/counts.sum()
invfreqs = np.max(freqs)/freqs
dataset = SimpleDataset(x=x_train, y=y_train)
loader = torch.utils.data.DataLoader(dataset, **{'batch_size': 1024, 'shuffle': True, 'drop_last': False})
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(invfreqs).float())
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
EPOCHS = 100
for e in range(EPOCHS):
    for _x, _y in loader:
        optimizer.zero_grad()
        out = model(_x)
        loss = criterion(out, _y)
        loss.backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        model.eval()
        test_preds = model(torch.from_numpy(x_test).float())
    model.train()
    test_preds = test_preds.cpu().numpy()
    print(recall_score(y_test, np.argmax(test_preds, axis=1), average='macro'))









