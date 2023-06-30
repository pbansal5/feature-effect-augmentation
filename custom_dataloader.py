import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, load_metric
from tqdm import tqdm
import tqdm as tqdm_
from transformers import DistilBertTokenizer, DistilBertModel
import transformers
import torch
import copy
import os
import argparse
import wandb
import torch
from torch.utils.data import Dataset

def get_dataset (dataset,method,load_style,params):    
    if ('toyccless50' in dataset or 'toyccmore950' in dataset):
        num_epochs = 4
        train_size,val_size= 19727,3297
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_val.json'%dataset,method,load_style,params=params)
    elif ('killv4' in dataset or 'killv5' in dataset or 'killv6' in dataset or 'christv2' in dataset):
        # num_epochs = 10
        num_epochs = 5
        train_size,val_size= 4000,1000
        # train_size,val_size = 32612,5403
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_val.json'%dataset,method,load_style,params=params)
    elif ('black' in dataset or 'white' in dataset or 'racist' in dataset or 'hate' in dataset or \
         'gay' in dataset or 'stupid' in dataset or 'ignorant' in dataset or 'kill' in dataset or \
            'president' in dataset or 'police' in dataset or 'donald' in dataset or 'trump' in dataset or \
                 'country' in dataset or 'your' in dataset or 'folks' in dataset or 'guys' in dataset):
        num_epochs = 5
        train_size,val_size= 5000,1000
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_val.json'%dataset,method,load_style,params=params)
    elif ('equaltoycc' in dataset):
        num_epochs = 10
        # num_epochs = 5
        train_size,val_size= 7112,5880
        # train_size,val_size = 32612,5403
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_val.json'%dataset,method,load_style,params=params)
    elif ('toycc' in dataset):
        num_epochs = 5
        train_size,val_size= 269038,43046
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_val.json'%dataset,method,load_style,params=params)
    elif ('cebab' in dataset):
        num_epochs = 10
        train_size,val_size= 1060,222
        train_dataset,test_dataset = load_datasets('%s_train_inclusive.json'%dataset,'%s_validation.json'%dataset,method,load_style,params=params)
    elif ('yelp' in dataset):
        num_epochs = 4
        train_size,val_size= 131784,17186
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_test.json'%dataset,method,load_style,params=params) 
    elif ('imdb_highnatural_moretreatment' in dataset or 'imdb_bothnatural' in dataset):
        train_size,val_size= 1354,1328
        # org_config 
        num_epochs = 3*5
        train_size= train_size*2
        # gpt2_config 
        # num_epochs = 3*3
        # train_size= train_size
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_test.json'%dataset,method,load_style,params=params) 
    elif ('imdb' in dataset):
        num_epochs = 5
        train_size,val_size= 25000,2500
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_test.json'%dataset,method,load_style,params=params) 
    elif ('mnli' in dataset):
        num_epochs = 10
        # train_size,val_size= 261802,6692
        train_size,val_size= 50000,6692
        train_dataset,test_dataset = load_datasets('%s_train.json'%dataset,'%s_validation_matched.json'%dataset,method,load_style,params=params) 
    return train_dataset,test_dataset,train_size,val_size,num_epochs

class CustomDataset(Dataset):
    def __init__(self, train_file,ordering=None,subset=None, load_style='std',params=None):
        if (params.local):
            data_files_ = {"train":"../datasets/%s"%train_file}
        else : 
            data_files_ = {"train":"/data/pbansal/datasets/%s"%train_file}
            # if ('yelp' in train_file):
            #     data_files_ = {"train":"/data/pbansal/datasets/%s"%train_file}
            # else :
            #     data_files_ = {"train":"./%s"%train_file}
        self.train_file = train_file
        dataset_ = load_dataset("json", data_files=data_files_)['train']
        self.cc_text = np.array([x['sentence'] for x in dataset_])
        if ('cebab' in train_file):
            self.cc_text_cf = np.array([x['sentence_cf'] for x in dataset_])
        self.cc_easy = np.array([x['easy'] for x in dataset_])
        self.cc_label = np.array([x['label'] for x in dataset_])
        self.cc_label_clean = np.array([x['label_clean'] for x in dataset_])
        self.cc_treatment = np.array([x['treatment'] for x in dataset_])
        
        # if (dataset_[0].get('trigger') is None):
        if True:
            self.cc_trigger = np.array([x['label_clean'] for x in dataset_])
        else : 
            self.cc_trigger = np.array([x['trigger'] for x in dataset_])
        self.params = params

        if type(ordering) is np.ndarray:
            if type(subset) is str : 
                if (subset == 'first'):
                    ordering_new = ordering[:int(0.8*len(ordering))]
                elif (subset == 'last') : 
                    ordering_new = ordering[int(0.8*len(ordering)):]
                else : exit()
            self.cc_text,self.cc_label,self.cc_label_clean,self.cc_treatment,self.cc_trigger = self.cc_text[ordering_new],self.cc_label[ordering_new],self.cc_label_clean[ordering_new],self.cc_treatment[ordering_new],self.cc_trigger[ordering_new]

        assert len(np.unique(self.cc_label)) <= 2
        assert len(np.unique(self.cc_treatment)) <= 2

        self.load_style = load_style
        assert load_style in ['std','label_wise','groupdro','subsample']

        self.label0_indices = np.where(self.cc_label == 0)[0]
        self.label1_indices = np.where(self.cc_label == 1)[0]

        self.group_indices_dict = dict({})
        self.group_ids = np.zeros(len(self.cc_label))-1
        for i in range(1,5):
            this_indices = np.where(np.logical_and(self.cc_label_clean == int((i-1)/2),self.cc_treatment == int((i-1)%2)))[0]
            self.group_indices_dict['group%d_indices'%i] = this_indices
            self.group_ids[this_indices] = i

        self.cov_group_indices_dict = dict({})
        self.cov_group_ids = np.zeros(len(self.cc_label))-1
        for i in range(1,5):
            this_indices = np.where(np.logical_and(self.cc_trigger == int((i-1)/2),self.cc_treatment == int((i-1)%2)))[0]
            self.cov_group_indices_dict['group%d_indices'%i] = this_indices
            self.cov_group_ids[this_indices] = i

        self.label_indices_dict = dict({})
        for i in range(2):
            self.label_indices_dict['label%d_indices'%i] = np.where(self.cc_label_clean == i)[0]

        self.ProbLabel1 = np.mean(self.cc_label)
        self.ProbTreatment1 = np.mean(self.cc_treatment)

        self.GroupwiseConditionalLabelTreatment = [np.logical_and(self.cc_label_clean == int((i-1)/2),self.cc_treatment == int((i-1)%2)).astype(np.int32).sum()/ \
            ((self.cc_treatment == int((i-1)%2)).astype(np.int32).sum()+1) for i in range(1,5)]
        
    def print_summary(self):
        print ("Ratio treated : %0.2f, Label Mean : %0.2f"%(self.cc_treatment.mean(),self.cc_label.mean()))
        print ("Treated Label Mean : %0.2f, Untreated Label Mean : %0.2f"%(self.cc_label[self.cc_treatment==1].mean(),self.cc_label[self.cc_treatment==0].mean()))
        
    def get_treated(self):
        return self.cc_treatment.mean()
        
    def get_bias_numbers(self):
        return self.cc_label[self.cc_treatment==1].mean(),1-self.cc_label[self.cc_treatment==0].mean()

    def __len__(self):
        return len(self.cc_label)
        
    def __getitem__(self, idx):
        if (self.load_style != 'std'):
            if (self.load_style == 'label_wise'):
                dict_interest = self.label_indices_dict
                selected_list = np.random.choice(np.array([x for x in dict_interest.values()],dtype=object))
            elif (self.load_style == 'groupdro'):
                dict_interest = self.group_indices_dict
                selected_list = np.random.choice(np.array([x for x in dict_interest.values()],dtype=object))
            elif (self.load_style == 'subsample'):
                label_selected = int(np.random.uniform()<self.ProbLabel1)
                treatment_selected = int(np.random.uniform()<self.ProbTreatment1)
                selected_list = self.group_indices_dict['group%d_indices'%int(2*label_selected + treatment_selected + 1)]
            idx = np.random.choice(selected_list)
        document = self.cc_text[idx]
        if ('cebab' in self.train_file):
            if (self.cc_treatment[idx] == 1):
                document = (self.cc_text_cf[idx],self.cc_text[idx])
            else : 
                document = (self.cc_text[idx],self.cc_text_cf[idx])
        return document,self.cc_label[idx],self.cc_treatment[idx],self.group_ids[idx],self.GroupwiseConditionalLabelTreatment[int(self.group_ids[idx])-1]

    def __getall__(self):
        if (self.params.eval_easy):
            return self.cc_text[self.cc_easy==1],self.cc_label[self.cc_easy==1],self.cc_treatment[self.cc_easy==1],self.group_ids[self.cc_easy==1],self.cov_group_ids[self.cc_easy==1]
        else:
            documents = self.cc_text
            if ('cebab' in self.train_file):
                documents = ([x if z==0 else y for (x,y,z) in zip(self.cc_text,self.cc_text_cf,self.cc_treatment)],
                            [y if z==0 else x for (x,y,z) in zip(self.cc_text,self.cc_text_cf,self.cc_treatment)])
            return documents,self.cc_label,self.cc_treatment,self.group_ids,self.cov_group_ids

def my_collate(batch):
    text = [item[0] for item in batch]
    label = torch.FloatTensor([item[1] for item in batch]).cuda()
    treatment = torch.LongTensor([item[2] for item in batch]).cuda()
    group = torch.LongTensor([item[3] for item in batch]).cuda()
    conditionalprobs = torch.LongTensor([item[4] for item in batch]).cuda()
    return [text, label, treatment,group,conditionalprobs]

def load_datasets(train_file,test_file,method,load_style,params):
    # if (not ('estimator' in method)):
    #     test_dataset = CustomDataset(test_file, load_style='std',params=params)
    #     train_dataset = CustomDataset(train_file, load_style=load_style,params=params) 
    # else : 
    #     train_dataset = CustomDataset(train_file, load_style='std',params=params) 
    #     ordering = np.arange(len(train_dataset))
    #     np.random.shuffle(ordering)
    #     # np.save('ordering_mnli.npy',ordering)
    #     # exit()
    #     train_dataset = CustomDataset(train_file, load_style=load_style,ordering=ordering,subset='first',params=params) 
    #     test_dataset = CustomDataset(train_file, load_style=load_style,ordering=ordering,subset='last',params=params) 

    test_dataset = CustomDataset(test_file, load_style='std',params=params)
    train_dataset = CustomDataset(train_file, load_style=load_style,params=params) 
    return train_dataset,test_dataset