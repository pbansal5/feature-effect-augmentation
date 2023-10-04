import torch
import numpy as np
import tqdm as tqdm_
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import DebertaTokenizer, DebertaModel
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
import transformers
import copy
import os
import argparse
import wandb
from custom_functions import *
from custom_dataloader import *
from custom_evaluator import *
from custom_utils import *
import argparse
from torch.utils.data import DataLoader
import configparser
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression




## Loading Arguments

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0,help='random seed')
parser.add_argument('--method', type=str, default='std') # mnli, yelp
parser.add_argument('--model', type=str, default='distil') # distil,deberta,roberta
parser.add_argument('--dataset', type=str, default='cc')
parser.add_argument('--datadir', type=str, default='/data/pbansal')
parser.add_argument('--local', action='store_true')
parser.add_argument('--eval_easy', action='store_true')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_predictions', action='store_true')
parser.add_argument('--wandb_name', type=str, default='debugv2')
parser.add_argument('--prepend', type=str, default='')
parser.add_argument('--target', type=float, default=0)
parser.add_argument('--propen_weight', type=float, default=0)
parser.add_argument('--reisz_weight', type=float, default=0)
parser.add_argument('--gamma', type=float, default=1.0)

params = parser.parse_args()
np.random.seed(params.seed)
method = params.method
valid_methods = [
    'reisz_estimator','propensity_estimator',
    'std', 'removetoken', 'regtocustom',
    'subsample','groupdro',
    'inlp','inlp_rlace',
    'bias_dfl','success_dfl','bias_demo_dfl',
    'bias_poe','bias_demo_poe'
    ]

assert method in valid_methods
dataset = params.dataset
datadir = params.datadir
model_name = params.model
assert model_name in ['distil','deberta-b','deberta-l','bert-b','bert-l','roberta-b','roberta-l','tinybert','gpt2','gpt2-large']

config = configparser.ConfigParser()
config.read('custom_configs.ini')
config_lists = config.sections()

config_method = method
if not (method in config_lists): config_method = 'DEFAULT'
config_weights = dict({key : float(value) for key,value in config[config_method].items()})
if (params.propen_weight>0 and config_weights['propen_weight'] > 0.0):
    config_weights['propen_weight'] = params.propen_weight
if (params.reisz_weight>0 and config_weights['reisz_weight'] > 0.0):
    config_weights['reisz_weight'] = params.reisz_weight

if ('-l' in model_name):
    batch_size = 16
else : 
    batch_size = 32
useScheduler = True
linear_weight_decay,bias_linear_weight_decay,reisz_linear_weight_decay = 1e-2,1e-2,1e-2
bert_weight_decay,bias_bert_weight_decay = 1e-2,1e-2
bert_lr,bias_bert_lr = 1e-5,1e-5
linear_lr,bias_linear_lr = 1e-4,1e-4
model_out_dim,bias_model_out_dim = 768,128
max_length = 128
load_saved_dicts = False
cer_target = params.target if config_weights['cer_weight']>0 else 0.0

load_style = 'std'
if (method in ['groupdro','subsample']):
    load_style = method
# load_style = 'subsample'




## Loading Dataset
# if ('imdb' in dataset): max_length = 64
train_dataset,test_dataset,train_size,val_size,num_epochs = get_dataset(dataset,method,load_style,params)

eval_every = int(train_size/batch_size)
iterations = eval_every*num_epochs+1

train_dataloader = DataLoader(train_dataset,collate_fn=my_collate,batch_size=batch_size,shuffle=True)
print ("Train Summary : ")
train_dataset.print_summary()
print ("Test Summary : ")
test_dataset.print_summary()
# if (train_dataset.get_treated()<0.1):
#     config_weights['reisz_weight'] = 0.1*config_weights['reisz_weight']




## Initialising Model
large_small = 'large' if '-l' in model_name else 'base'
if (model_name == 'gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').cuda()
    tokenizer.pad_token = tokenizer.eos_token
if (model_name == 'gpt2-large'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2Model.from_pretrained('gpt2-large').cuda()
    tokenizer.pad_token = tokenizer.eos_token
elif (model_name == 'distil'):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-%s-uncased"%large_small)
    model = DistilBertModel.from_pretrained("distilbert-%s-uncased"%large_small).cuda()
elif ('roberta' in model_name): 
    tokenizer = RobertaTokenizer.from_pretrained("roberta-%s"%large_small)
    model = RobertaModel.from_pretrained("roberta-%s"%large_small).cuda()
elif ('deberta' in model_name): 
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-%s"%large_small)
    model = DebertaModel.from_pretrained("microsoft/deberta-%s"%large_small).cuda()
elif ('tiny' in model_name): 
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny").cuda()
    model_out_dim = 128
elif ('bert' in model_name):
    tokenizer = BertTokenizer.from_pretrained("bert-%s-uncased"%large_small)
    model = BertModel.from_pretrained("bert-%s-uncased"%large_small).cuda()
else : 
    exit()

if (('dfl' in method) or ('poe' in method)):
    use_probabilities = True
else : use_probabilities = False
use_probabilities = True
reg_head = custom_reg_head(model_out_dim,use_probabilities)
reisz_head = custom_reisz_head(model_out_dim)
propen_head = custom_propen_head(model_out_dim)

bert_pipe = make_forward(model,tokenizer,dataset,model_out_dim,max_length).forward
bias_bert_pipe,bias_reg_head,bias_model = None,None,None

if ('bias' in params.method or 'success' in params.method):
    if ('bias' in params.method) :
        bias_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        bias_model = AutoModel.from_pretrained("prajjwal1/bert-tiny").cuda()
    else : 
        bias_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-%s-uncased"%'base')
        bias_model = DistilBertModel.from_pretrained("distilbert-%s-uncased"%'base').cuda()
    bias_reg_head = custom_reg_head(bias_model_out_dim,True)
    bias_bert_pipe = make_forward(bias_model,bias_tokenizer,dataset,bias_model_out_dim,max_length).forward






## Loading Wandb

if (params.method == 'reisz_estimator'):
    this_weight = config_weights['reisz_weight']
elif (params.method == 'propensity_estimator'):
    this_weight = config_weights['propen_weight']
else : 
    this_weight = config_weights['cer_weight']

wandb.login(key="e45d2f6c4df62f742cc5974e9865de8bfeaa")
wandb.init(project=params.wandb_name, entity="pbansal")
wandb.run.name = '%s%0.3frs_weight_%s_%s_%0.2fcer_target_%s_model_%sseed_%0.1fgamma'%(params.prepend,this_weight,dataset,method,cer_target,model_name,params.seed,params.gamma)





## Loading State Dicts

if (load_saved_dicts):
# if (load_saved_dicts or method == 'inlp'):
    name_ = method if method != 'inlp' else 'std'
    name_ = '%s_%s'%(dataset,'%s_%0.2fcer_weight_%0.2fcer_target'%(name_,config_weights['cer_weight'],cer_target))
    saved_dicts = torch.load('%s/saved_models/%s.pt'%(datadir,name_))
    load_dicts(saved_dicts,model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head)

## Initialising Optimizers 

tmle_beta = torch.zeros(1,requires_grad=True, device="cuda")

parameters_list = [{'params':list(filter(lambda p: p.requires_grad,model.parameters())),'weight_decay':bert_weight_decay,'lr':bert_lr},
                {'params':list(filter(lambda p: p.requires_grad,reg_head.parameters())),'weight_decay':linear_weight_decay,'lr':linear_lr},
                {'params':list(filter(lambda p: p.requires_grad,reisz_head.linear.parameters())),'weight_decay':reisz_linear_weight_decay,'lr':linear_lr},
                {'params':list(filter(lambda p: p.requires_grad,propen_head.linear.parameters())),'weight_decay':reisz_linear_weight_decay,'lr':linear_lr},
                {'params':[tmle_beta],'weight_decay':0,'lr':1e-4},]

if ('bias' in params.method or 'success' in params.method):
    parameters_list = parameters_list + [
        {'params':list(filter(lambda p: p.requires_grad,bias_model.parameters())),'weight_decay':bias_bert_weight_decay,'lr':bias_bert_lr},
        {'params':list(filter(lambda p: p.requires_grad,bias_reg_head.parameters())),'weight_decay':bias_linear_weight_decay,'lr':bias_linear_lr}]

optimizer = torch.optim.Adam(parameters_list)

if (useScheduler):
    sched = transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=10,num_training_steps=iterations)





## Training Setup and Loop

best_loss = float('inf')
thres,counter = 30,0
bce_loss_fn = torch.nn.BCELoss()
my_evaluator = evaluator(test_dataset,bert_pipe,bias_bert_pipe,reg_head,bias_reg_head,reisz_head,propen_head,config_weights,params)

group_weights = torch.FloatTensor([1,1,1,1]).cuda()
group_weights = group_weights/group_weights.sum()
group_weights_step_size = 0.01
running_average = 0.0

pshift10,pshift01 = train_dataset.get_bias_numbers()
pshift10,pshift01 = cer_target/pshift10,cer_target/pshift01

for i in tqdm_.trange(iterations):
    # if (method == 'inlp') : break

    this_text, this_label, this_treatment, this_group, this_conditionalprobs = next(iter(train_dataloader))

    # Wipe Treatment Information for "Remove Token from Input"
    if (method == 'removetoken'):
        this_treatment = torch.zeros(this_treatment.shape).cuda()

    bert_rep = bert_pipe(this_text)
    reg_pred = reg_head(bert_rep,this_treatment)[:,0]

    # BIAS Computation
    bias_reg_loss,bias_reg_pred =  0,None
    if ('bias' in method):
        bias_bert_rep = bias_bert_pipe(this_text)
        bias_reg_pred = bias_reg_head(bias_bert_rep,this_treatment)[:,0]
        if ('demo' in method):
            bias_reg_loss = bce_loss_fn(bias_reg_pred,this_treatment.float())
        else : 
            bias_reg_loss = bce_loss_fn(bias_reg_pred,this_label)
    elif ('success' in method):
        bias_bert_rep = bias_bert_pipe(this_text)
        bias_reg_pred = bias_reg_head(bias_bert_rep,this_treatment)[:,0]
        correct = ((reg_pred>0.5).int() == this_label).float()
        bias_reg_loss = bce_loss_fn(bias_reg_pred,correct)

    # REGRESSION Loss Computation
    reg_loss, group_weights = reg_loss_computation(method,reg_pred,this_label,this_treatment,this_group,bias_reg_pred,params.gamma,group_weights,group_weights_step_size)

    # Reisz Loss Computation
    reisz_loss,propen_loss,tmle_loss = 0,0,0
    if ('estimator' in method ):
        propen_pred = propen_head(bert_rep)
        reisz_pred = reisz_head(bert_rep)
        reisz_loss = (-2*(reisz_pred[:,1]-reisz_pred[:,0])+(torch.gather(reisz_pred,1,this_treatment[:,None])[:,0]**2)).mean()
        propen_loss = bce_loss_fn(propen_pred[:,0],this_treatment.float())
        tmle_loss = ((this_label-(reg_pred+tmle_beta*torch.gather(reisz_pred,1,this_treatment[:,None])[:,0]))**2).mean()
        
    # Regularizer Loss Computation
    ate_loss = 0
    if (config_weights['cer_weight']>0): #and cc_treatment_train[batch_indices].sum()>5):
        # direct_est = (reg_head(bert_rep,torch.ones(this_treatment.shape))[:,0]-
        #                 reg_head(bert_rep,torch.zeros(this_treatment.shape))[:,0]).mean()
        # ate_loss = ((direct_est-cer_target)**2).mean()#

        this_treatment_aug = 1-this_treatment
        reg_pred_aug = reg_head(bert_rep,this_treatment_aug)[:,0]
        this_label_aug = copy.deepcopy(this_label)
        indices01 = torch.where(torch.logical_and(this_label_aug==0,this_treatment_aug==1))[0]
        indices10 = torch.where(torch.logical_and(this_label_aug==1,this_treatment_aug==0))[0]
        indices10 = indices10[torch.randperm(len(indices10))][:int(len(indices10)*pshift10)]
        indices01 = indices01[torch.randperm(len(indices01))][:int(len(indices01)*pshift01)]
        this_label_aug[indices10] = 0
        this_label_aug[indices01] = 1
        ate_loss = -(this_label_aug*torch.log(reg_pred_aug) + (1-this_label_aug)*torch.log(1-reg_pred_aug)).mean()

        # direct_est = (reg_head(bert_rep,torch.ones(this_treatment.shape))[:,0]-
        #                 reg_head(bert_rep,torch.zeros(this_treatment.shape))[:,0]).mean()
        # ate_loss = ((direct_est-cer_target)**2).mean()#

    # FINAL Loss Computation
    loss = config_weights['reisz_weight']*reisz_loss+config_weights['reg_weight']*reg_loss+\
        config_weights['propen_weight']*propen_loss+config_weights['cer_weight']*ate_loss+\
        config_weights['tmle_weight']*tmle_loss+config_weights['bias_reg_weight']*bias_reg_loss

    wandb.log({'train/loss':loss,'train/reisz_loss':reisz_loss,'train/reg_loss':reg_loss,'train/bias_reg_loss':bias_reg_loss,'train/cer_loss':ate_loss,'train/propen_loss':propen_loss,'train/tmle_loss':tmle_loss,'train/tmle_beta':tmle_beta,'train/cer_avg':running_average})

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (useScheduler): sched.step()

    if (i%eval_every == 0):
        with torch.no_grad():
            this_loss = my_evaluator.evaluate(group_weights)
            if (this_loss < best_loss):
                counter = 0
                best_loss = this_loss
                best_dicts = save_dicts(model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head)
            else : 
                counter += 1
            if (counter >= thres):
                break



## INLP Computation

if (method == 'inlp'):

    # with torch.no_grad():
    #     this_loss = my_evaluator.evaluate(group_weights)
    load_dicts(best_dicts,model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head)
    
    representations,labels = get_predictions(train_dataset,bert_pipe,reg_head,reisz_head)
    X = representations.cpu().numpy()
    Y = labels[:,2].cpu().numpy()

    if ('rlace' in method):
        optimizer_class = torch.optim.SGD
        optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
        optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
        epsilon = 0.010 # stop 0.1% from majority acc
        batch_size = 512
        train_indices = np.random.choice(len(X),size=(int(0.8*len(X)),),replace=False)
        dev_indices = np.setdiff1d(np.arange(len(X)),train_indices)

        output = solve_adv_game(X[train_indices], Y[train_indices], X[dev_indices], Y[dev_indices], rank=1, device="cuda", out_iters=50000, optimizer_class=optimizer_class, optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size)
        P = output["P"]
    else : 
        P,_,_ = get_debiasing_projection(SGDClassifier, {}, 100, representations.shape[1], True, 0.50, X, Y, X, Y, by_class = False)

    reg_head.protection_matrix = torch.from_numpy(P).cuda().float()
    with torch.no_grad():
        this_loss = my_evaluator.evaluate(group_weights)

    best_dicts = save_dicts(model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head)






## Evaluating Best Model

with torch.no_grad():
    load_dicts(best_dicts,model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head)
    best_loss = my_evaluator.evaluate(group_weights,prepend='best_')





## Saving Model and Predictions

if (params.save_predictions):
    representations,scores = get_predictions(train_dataset,bert_pipe,reg_head,reisz_head)
    torch.save(representations,'%s/saved_representations/%s.pt'%(datadir,wandb.run.name))
    torch.save(scores,'%s/saved_scores/%s.pt'%(datadir,wandb.run.name))

    representations,scores = get_predictions(test_dataset,bert_pipe,reg_head,reisz_head)
    torch.save(representations,'%s/saved_representations/test_%s.pt'%(datadir,wandb.run.name))
    torch.save(scores,'%s/saved_scores/test_%s.pt'%(datadir,wandb.run.name))

    if ('bias' in params.method or 'success' in params.method):
        representations,scores = get_predictions(train_dataset,bias_bert_pipe,bias_reg_head)
        torch.save(representations,'%s/saved_representations/bias_%s.pt'%(datadir,wandb.run.name))
        torch.save(scores,'%s/saved_scores/bias_%s.pt'%(datadir,wandb.run.name))

        representations,scores = get_predictions(test_dataset,bias_bert_pipe,bias_reg_head)
        torch.save(representations,'%s/saved_representations/test_bias_%s.pt'%(datadir,wandb.run.name))
        torch.save(scores,'%s/saved_scores/test_bias_%s.pt'%(datadir,wandb.run.name))
    
if (params.save_model):
    torch.save(best_dicts,'%s/saved_models/%s.pt'%(datadir,wandb.run.name))
