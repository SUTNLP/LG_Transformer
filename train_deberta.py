import numpy as np
import torch 
import json
import pickle
import unicodedata
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, PolynomialLR
import torch.nn.functional as F
import torch 
import json
import pickle
import unicodedata
from tqdm import tqdm
from copy import deepcopy
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DebertaTokenizer, DebertaModel, RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
import os
from collections import defaultdict, namedtuple, OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, RMSprop
from copy import deepcopy
from typing import Union, Callable
from sklearn.utils import shuffle
import random
# from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DebertaTokenizer, DebertaModel, RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import pandas as pd
from itertools import count
import os
from collections import defaultdict, namedtuple, OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, RMSprop
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import optuna
import random
# from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

seeds = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
seed_idx = 0
seed = seeds[seed_idx]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms

model_name = 'deberta'

if model_name == 'bert':
  model_path_or_name = 'bert-large-uncased'
  tokenizer = BertTokenizer.from_pretrained(model_path_or_name)
  model = BertModel.from_pretrained(model_path_or_name)
elif model_name == 'roberta':
  model_path_or_name = 'roberta-large'
  tokenizer = RobertaTokenizer.from_pretrained(model_path_or_name)
  model = RobertaModel.from_pretrained(model_path_or_name)
elif model_name == 'deberta':
  model_path_or_name = 'microsoft/deberta-base'
  tokenizer = DebertaTokenizer.from_pretrained(model_path_or_name)
  model = DebertaModel.from_pretrained(model_path_or_name)


train_data = pd.read_csv('train.tsv', sep='\t')
eval_data = pd.read_csv('eval.tsv', sep='\t')
test_data = pd.read_csv('test.tsv', sep='\t')
nahj_test = pd.read_csv('nagj_test.tsv', sep='\t')
train_label = pd.read_csv('train_label.tsv', sep='\t')
eval_label = pd.read_csv('eval_label.tsv', sep='\t')

labels_name = list(train_label.columns)
labels_name.remove('Argument ID')

id_to_label = dict()
for label_id, label_name in enumerate(labels_name):
  id_to_label[label_id] = label_name

train_dataframe = train_data.merge(train_label, on='Argument ID')
eval_dataframe = eval_data.merge(eval_label, on='Argument ID')
test_dataframe = test_data

def tolist(tensor):
  return tensor.detach().cpu().tolist()


class_weights = list()
for label_name in labels_name:
  ys = train_dataframe[label_name].values.tolist()
  class_weight_ = compute_class_weight(class_weight='balanced', classes=np.unique(ys), y=ys)
  class_weights.append(class_weight_)


adj_mat = [[0 for j in range(len(labels_name))] for i in range(len(labels_name))]
all_mat = [[0 for j in range(len(labels_name))] for i in range(len(labels_name))]
for data in train_dataframe.iterrows():
  for i, label_1 in enumerate(labels_name):
    for j, label_2 in enumerate(labels_name):
      if data[1][label_1] and data[1][label_2]:
        adj_mat[i][j] += 1
      if (data[1][label_1] or data[1][label_2]):
        all_mat[i][j] += 1


adj_mat = np.array(adj_mat)
all_mat = np.array(all_mat)
edge_attr_mat = np.divide(adj_mat, all_mat)

adj_mat_thresh = deepcopy(adj_mat)
thresh = 0
adj_mat_thresh[adj_mat_thresh >= thresh] = 1

edges = list()
edges_attr = list()
for i in range(len(labels_name)):
  for j in range(len(labels_name)):
    if adj_mat_thresh[i, j] == 1:
      edges.append([i, j])
      edges_attr.append([edge_attr_mat[i][j]])
edges = torch.LongTensor(edges)
edges_attr = torch.tensor(edges_attr)


class ValueDataset(Dataset):

  def __init__(self, dataframe, tokenizer, max_length, is_test=False):
    self.dataframe = dataframe
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.is_test = is_test

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    sample = self.dataframe.loc[idx]
    labels = ' , '.join(labels_name)
    tokenized_text = tokenizer(
          sample['Premise'],
          labels,
          max_length=self.max_length,
          padding='max_length',
          truncation='only_first',
          return_tensors='pt')
    
    input_ids = tokenized_text['input_ids']
    labels_start = (input_ids == tokenizer.sep_token_id).nonzero().contiguous().view(-1).tolist()[0] + 1

    labels_tokens = []
    for label_name in labels_name:
      label_tokens = tokenizer(label_name, add_special_tokens=False)
      labels_tokens.append(label_tokens['input_ids'])

    labels_tokens_span = []
    c_token = labels_start
    for label_tokens in labels_tokens:
      labels_tokens_span.append([c_token, c_token + len(label_tokens) - 1])
      c_token += len(label_tokens) + 1

    tokenized_text['labels_tokens_span'] = torch.tensor(labels_tokens_span)

    if not self.is_test:
      labels = torch.LongTensor([sample[label_name] for label_name in labels_name])
      tokenized_text['labels'] = labels
    return tokenized_text





class PGD():

    def __init__(self, model,emb_name,epsilon=1.,alpha=0.3):
        # The emb_name parameter should be replaced with the parameter name of the embedding in your model
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    # adversarial training : attack to change embedding abit with regards projected gradiant descent
    def attack(self,first_strike=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if first_strike:
                    # print('tt', param.data)
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    # Compute new params
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    # Restore to the back-up embeddings
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    # Project Gradiant Descent
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    # Back-up parameters
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler' not in name:
                self.grad_backup[name] = param.grad.clone()

    # Restore grad parameters
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'pooler' not in name:
                param.grad = self.grad_backup[name]


class ValueModel(nn.Module):

  def __init__(self, transformer, hidden_size):
    super(ValueModel, self).__init__()
    self.transformer = transformer
    # gcn layers
    self.conv1 = GCNConv(transformer.config.hidden_size, hidden_size)
    self.dropout1 = nn.Dropout(p=.3)
    self.conv2 = GCNConv(hidden_size, hidden_size)
    self.dropout2 = nn.Dropout(p=.3)
    self.conv3 = GCNConv(hidden_size, hidden_size)
    self.dropout3 = nn.Dropout(p=.3)
    # head layer
    self.head_layers = nn.ModuleList([nn.Linear(hidden_size, 2) for i in range(len(labels_name))])
    # params

  def integrate(self, batch_output, batch_labels_tokens_span):
    batch_size = batch_output.shape[0]
    integrated_batch = []
    self.batch_output = batch_output
    for i in range(batch_size):
      integrated_sample_labels = []
      output = batch_output[i]
      self.output = output
      labels_tokens_span = batch_labels_tokens_span[i]
      self.T = labels_tokens_span
      for label_tokens_span in labels_tokens_span:
        self.A = label_tokens_span
        integrated_label = output[label_tokens_span[0].item(): label_tokens_span[1].item() + 1].sum(0).view(-1)
        self.Y = integrated_label
        assert integrated_label.shape[0] == self.transformer.config.hidden_size
        integrated_sample_labels.append(integrated_label)
      integrated_sample_labels = torch.stack(integrated_sample_labels)
      integrated_batch.append(integrated_sample_labels)
    integrated_batch = torch.stack(integrated_batch)
    return integrated_batch

  def forward(self, data, edge_index, edge_attr, batch_labels_tokens_span):
    # transformer layers
    batch_size = data['input_ids'].shape[0]
    x = self.transformer(input_ids=data['input_ids'], attention_mask=data['attention_mask']).last_hidden_state
    x = self.integrate(x, batch_labels_tokens_span)
    # graph layers

    x = self.conv1(x, edge_index=edge_index)
    x = F.relu(x)
    x = self.dropout1(x)

    x = self.conv2(x, edge_index=edge_index)
    x = F.relu(x)
    x = self.dropout2(x)

    x = self.conv3(x, edge_index=edge_index)
    x = F.relu(x)
    x = self.dropout3(x)

    hidden = x 

    # linear layers
    out_list = list()
    for label_id, head_layer in enumerate(self.head_layers):
      out_list.append(head_layer(hidden[:, label_id, :]))
    out_logits = torch.stack(out_list).transpose(1, 0).contiguous()
    return out_logits, hidden.view(batch_size, -1), out_list, None


def train(params, epoch, model, train_dataloader, optimizer, scheduler, device,
          use_adv=False, use_vadv=False, vat_weight=.5, adv_reg=False, adv_use_every_layer=True):
  model.train()
  named_weights = [n for n, _ in model.named_parameters() if 'dense.weight' in n and 'pooler' not in n] + ["word_embeddings."]

  rdrop_ratio = 0
  loss_collection = [[] for _ in range(5)]
  loss_fns = [nn.CrossEntropyLoss(torch.tensor(class_weight_).to(device).float()) for class_weight_ in class_weights]
  for step, data in enumerate(train_dataloader):
    if adv_use_every_layer:
      rand_layer = random.sample(named_weights, 1)[0] 
      adv_layer = rand_layer
    else:
      adv_rand = random.uniform(0, 1) 
      if adv_rand > .5:
        adv_layer = "word_embeddings."
      else:
        rand_layer = random.sample(named_weights, 1)[0] 
        adv_layer = rand_layer
    pgd = PGD(
      model=model,
      emb_name=adv_layer
    )

    c_batch_size = data['input_ids'].shape[0]
    batch_labels_tokens_span = data.pop('labels_tokens_span')
    labels = data.pop('labels').to(device).t().split(1)
    for key in data:
      if 'joint' not in key:
        data[key] = data[key].to(device).view(c_batch_size, -1)
      else:
        data[key] = data[key].to(device)

    logits_1, _, out_list_1, _ = model(data, edges.to(device).t(), edges_attr.to(device), batch_labels_tokens_span)
    loss_list = list()
    for list_1, label, loss_fn in zip(out_list_1, labels, loss_fns):
      loss_list.append(loss_fn(list_1, label.view(-1)))
    main_loss = torch.stack(loss_list).mean()
    loss = main_loss 
    loss.backward()
    loss_collection[0].append(main_loss.item())



    if use_adv:
      # PGD Start
        pgd.backup_grad()
        attack_times = 1
        for attack_time in range(attack_times):
            # Add adversarial perturbation to the embedding, backup param.data during the first attack
            pgd.attack(first_strike=(attack_time==0))
            if attack_time != attack_times-1:
              model.zero_grad()
            else:
              pgd.restore_grad()
            logits_1, _, out_list_1, _ = model(data, edges.to(device).t(), edges_attr.to(device), batch_labels_tokens_span)
            loss_list = list()
            for list_1, label, loss_fn in zip(out_list_1, labels, loss_fns):
              loss_list.append(loss_fn(list_1, label.view(-1)))
            main_loss = torch.stack(loss_list).mean()
            loss_adv = main_loss
            loss_collection[2].append(loss_adv.item())
            # Backpropagation, and on the basis of the normal grad, accumulate the gradient of the adversarial training
            if not adv_reg or attack_time != attack_times-1:
              loss_adv.backward()
        if adv_reg:
          reg_logits = torch.cat([hidden_1_detached, hidden_2], dim=0)
          reg_labels = torch.cat([torch.arange(logits.shape[0]), torch.arange(logits.shape[0])], dim=0)
          reg_loss = ntxent(reg_logits, reg_labels)
          reg_loss = reg_loss * .3
          loss_adv_ = loss_adv + reg_loss
          loss_adv_.backward()
          loss_collection[3].append(reg_loss.item())

        # Restore embedding parameters
        pgd.restore() 



    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if len(loss_collection[0]) % log_step == 0:
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | Main Loss {round(sum(loss_collection[0]) / (len(loss_collection[0]) + 1e-8), 4)}')
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | Vat Loss {round(sum(loss_collection[1]) / (len(loss_collection[1]) + 1e-8), 4)}')
      print(f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step + 1}/{len(train_dataloader)}] | Adv Loss {round(sum(loss_collection[2]) / (len(loss_collection[2]) + 1e-8), 4)}')
      print('------------------------------------------------')
      loss_collection = [[] for _ in range(5)]

def eval(params, epoch, model, eval_dataloader, device):
  with torch.no_grad():
    model.eval()
    all_labels = OrderedDict()
    all_preds = OrderedDict()
    for label_name in labels_name:
      all_labels[label_name] = list()
      all_preds[label_name] = list()

    for data in eval_dataloader:
      c_batch_size = data['input_ids'].shape[0]
      batch_labels_tokens_span = data.pop('labels_tokens_span').to(device)

      for key in data:
        if 'joint' not in key:
          data[key] = data[key].to(device).view(c_batch_size, -1)
        else:
          data[key] = data[key].to(device)

      labels = data.pop('labels').view(-1)
      logits, _, _, _ = model(data, edges.to(device).t(), edges_attr.to(device).float(), batch_labels_tokens_span)

      preds = tolist(logits.argmax(2).view(c_batch_size, -1))
      labels = tolist(labels.view(c_batch_size, -1))

      for all_pred, all_label in zip(preds, labels):
        for id, (pred, label) in enumerate(zip(all_pred, all_label)):
          
          all_labels[id_to_label[id]].append(label)
          all_preds[id_to_label[id]].append(pred)

    f1_list = list()
    for label_name in labels_name:
      # print(len(all_labels[label_name]), len(all_preds[label_name]))
      f1 = f1_score(all_labels[label_name], all_preds[label_name])
      f1_list.append(f1)

    # for f1, label_name in zip(f1_list, labels_name):
    #   print(f'{label_name} -> {round(f1, 4)}')
    f1_mean = sum(f1_list) / len(f1_list)
    return f1_mean

def test(params, epoch, model, eval_dataloader, device):
  with torch.no_grad():
    model.eval()
    all_preds = OrderedDict()
    for label_name in labels_name:
      all_preds[label_name] = list()
    for data in eval_dataloader:
      c_batch_size = data['input_ids'].shape[0]
      batch_labels_tokens_span = data.pop('labels_tokens_span').to(device)
      for key in data:
        if 'joint' not in key:
          data[key] = data[key].to(device).view(c_batch_size, -1)
        else:
          data[key] = data[key].to(device)

      logits, _, _, _ = model(data, edges.to(device).t(), edges_attr.to(device).float(), batch_labels_tokens_span)

      preds = tolist(logits.argmax(2).view(c_batch_size, -1))

      for all_pred in preds:
        for id, pred in enumerate(all_pred):
          all_preds[id_to_label[id]].append(pred)
    return all_preds



def save_model(epoch, model, optimizer, scheduler, f1_list, c_f1):
  filename = 'best_ch.pt'
  torch.save(
      {'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(), 
       'f1_list': f1_list + [c_f1]},
        filename)

def load_model():
  filename = 'best_ch.pt'
  if os.path.exists(filename):
    saved_dict = torch.load(filename)
    return True, saved_dict
  else:
    return False, None
    



transformers.logging.set_verbosity_error()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
hidden_size = 400
max_length = 150
batch_size = 2
params = {
    'epochs': 30,
    'learning_rate': 2e-5
}
log_step = 300
epochs = params['epochs']
weight_decay = 1e-3
# define model
value_model = ValueModel(deepcopy(model), 400).to(device)
# train dataloader
train_dataset = ValueDataset(train_dataframe, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# eval dataloader
eval_dataset = ValueDataset(eval_dataframe, tokenizer, max_length)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
# test dataloader
test_dataset = ValueDataset(test_dataframe, tokenizer, max_length, is_test=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# test dataloader
nahj_test_dataset = ValueDataset(nahj_test, tokenizer, max_length, is_test=True)
nahj_test_dataloader = DataLoader(nahj_test_dataset, batch_size=batch_size, shuffle=False)
# optimizations
optimization_steps = params['epochs'] * len(train_dataloader)
# warmup_ratio = .1
# warmup_steps = int(warmup_ratio * optimization_steps)
optimizer = AdamW(value_model.parameters(), lr=params['learning_rate'])
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio, num_training_steps=optimization_steps)
scheduler = PolynomialLR(optimizer, optimization_steps)
best_f1 = 0.
all_f1 = list()
start_epoch = 0
patience = 10


checkpoint_avl, saved_dict = load_model()
print(checkpoint_avl)
if checkpoint_avl:
  start_epoch = saved_dict['epoch']
  model_state_dict = saved_dict['model_state_dict']
  optimizer_state_dict = saved_dict['optimizer_state_dict']
  scheduler_state_dict = saved_dict['scheduler_state_dict']
  all_f1 = saved_dict['f1_list']
  best_f1 = max(all_f1)

  value_model.load_state_dict(model_state_dict)
  optimizer.load_state_dict(optimizer_state_dict)
  scheduler.load_state_dict(scheduler_state_dict)
  


for epoch in range(start_epoch, params['epochs']):
  train(params, epoch, value_model, train_dataloader, optimizer, scheduler, device)
  c_f1 = eval(params, epoch, value_model, eval_dataloader, device)
  if c_f1 > best_f1:
    best_f1 = c_f1
    save_model(epoch + 1, value_model, optimizer, scheduler, all_f1, c_f1)

  print(f'EPOCH {epoch + 1} EVAL -------> ', round(100 * c_f1, 2))
  print(f'EPOCH {epoch + 1} BEST -------> ', round(100 * best_f1, 2))


  all_f1.append(c_f1)

_, saved_dict = load_model()
value_model.load_state_dict(saved_dict['model_state_dict'])
preds = test(params, epoch, value_model, test_dataloader, device)
nahj_preds = test(params, epoch, value_model, nahj_test_dataloader, device)

test_out = deepcopy(test_dataframe)
nahj_test_out = deepcopy(nahj_test)
for key in preds.keys():
  test_out[key] = preds[key]
  nahj_test_out[key] = nahj_preds[key]
test_out.to_csv('test_preds.tsv', sep='\t', index=False)
nahj_test_out.to_csv('nahj_test_preds.tsv', sep='\t', index=False)

