import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, accuracy_score, recall_score, matthews_corrcoef
from pytorch_metric_learning import losses, distances, reducers, testers
from tqdm import tqdm
from model import MRGNN
import torch.nn as nn
from utils import *

class EntityClassify(MRGNN):
    def build_input_layer(self):
        return nn.ModuleList([ MLP(in_dim, self.h_dim, self.hm_dim, 1, self.dropout, 'layer', init_activate=False) for in_dim in self.in_dims])

def model_train(train_idx):
    model.train()
    optimizer.zero_grad()
    out, emb = model(data.x,  num_nodes, num_relations)
    loss = LOSS(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def model_val(val_idx, plus_test=False):
    model.eval()
    out, emb = model(data.x, num_nodes, num_relations)
    out_val = out.exp()[val_idx]
    pred_val = out_val.argmax(dim=-1)
    auroc_val = roc_auc_score(data.y[val_idx].cpu(), out_val[:, 1].cpu())
    
    if plus_test == False:
        acc_val = accuracy_score(data.y[val_idx].cpu(), pred_val.cpu())
        pr, re, thresholds2 = precision_recall_curve(data.y[val_idx].cpu(), out_val[:, 1].cpu())
        auprc_val = auc(re, pr)
        sens_val = recall_score(data.y[val_idx].cpu(), pred_val.cpu())
        spec_val = recall_score(data.y[val_idx].cpu(), pred_val.cpu(), pos_label=0)
        mcc_val = matthews_corrcoef(data.y[val_idx].cpu(), pred_val.cpu())
        return acc_val, sens_val, spec_val, mcc_val, auroc_val, auprc_val
    
    else:
        out_test = out.exp()[test_index]
        pred_test = out_test.argmax(dim=-1)
        acc_test = accuracy_score(data.y[test_index].cpu(), pred_test.cpu())
        auroc_test = roc_auc_score(data.y[test_index].cpu(), out_test[:, 1].cpu())
        pr, re, thresholds2 = precision_recall_curve(data.y[test_index].cpu(), out_test[:, 1].cpu())
        auprc_test = auc(re, pr)
        sens_test = recall_score(data.y[test_index].cpu(), pred_test.cpu())
        spec_test = recall_score(data.y[test_index].cpu(), pred_test.cpu(), pos_label=0)
        mcc_test = matthews_corrcoef(data.y[test_index].cpu(), pred_test.cpu())
        return auroc_val, acc_test, sens_test, spec_test, mcc_test, auroc_test, auprc_test

data = torch.load('./data/graph_drugsim.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=661)
train_index, test_index = next(sss.split(data.x.cpu(), data.y.cpu()))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=154)
train_slice, val_slice = next(sss.split(data.x[train_index].cpu(), data.y[train_index].cpu()))
val_index = train_index[val_slice]
train_index_ = train_index[train_slice]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=18)

f2 = torch.from_numpy(np.loadtxt('./data/drug similarity matrix 1.txt',dtype=np.float32))
data.x= torch.from_numpy(np.concatenate((data.x.cpu(), f2), axis=1)).to(device)
f3 = torch.from_numpy(np.loadtxt('./data/drug similarity matrix 2.txt',dtype=np.float32))
data.x= torch.from_numpy(np.concatenate((data.x.cpu(), f3), axis=1)).to(device)

# Hyperparameters
triplets_per_anchor = 60
low = 0.274
margin = 0.274

in_dims = [7922, 4264]
n_hidden = 64
num_classes = 2
dropout = 0.5

alpha = 10
beta = 1
transf_type = 0
threshold = 1/28
num_nodes = 8225
n_mlp_layer = 2
threshold_c = 0.03
hm_dim = 32
num_relations = 5
lr = 1e-2
l2norm = 1
weight_decay = 0.00738


# cross-validation
LOSS = F.cross_entropy

edge_index = torch.transpose(data.edge_index, 0, 1)
edge_type = torch.unsqueeze(data.edge_type, dim=1)
triples = torch.cat((edge_index, edge_type), dim=1)
triples = triples[:, [0, 2, 1]]

print('Hold-out validation progressing...')
model = EntityClassify(in_dims,
                           n_hidden,
                           num_classes,
                           triples=triples,
                           num_classes=num_classes,
                           dropout=dropout,
                           alpha=alpha,
                           beta=beta,
                           threshold=threshold,
                           num_nodes=num_nodes,
                           n_mlp_layer=n_mlp_layer,
                           threshold_c=threshold_c,
                           hm_dim=hm_dim).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2norm)
auroc_val_max = 0
epoch_count = 0
for epoch in range(500):
    epoch_count += 1
    model_train(train_index)
    auroc_val, acc, sens, spec, mcc, auroc, auprc = model_val(val_index, plus_test=True)
    if auroc_val > auroc_val_max:
        epoch_count = 0
        auroc_val_max = auroc_val
        acc_max = acc
        auroc_max = auroc
        auprc_max = auprc
        sens_max = sens
        spec_max = spec
        mcc_max = mcc
        epoch_max = epoch
    if epoch_count == 60:
        break
        
test_result = pd.Series([acc_max, sens_max, spec_max, mcc_max, auroc_max, auprc_max],
                        index=['Acc', 'Sens', 'Spec', 'MCC', 'AUC', 'AUPRC'])
print('Hold-out validation results:')
print(test_result)