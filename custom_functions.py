import torch
import numpy as np
import transformers

def save_dicts(model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head):
    final_dict = dict({})
    model.cpu(),reg_head.cpu(),reisz_head.cpu(),propen_head.cpu()
    final_dict['model'] = model.state_dict()
    final_dict['reg_head'] = reg_head.state_dict()
    final_dict['reisz_head'] = reisz_head.state_dict()
    final_dict['propen_head'] = propen_head.state_dict()
    model.cuda(),reg_head.cuda(),reisz_head.cuda(),propen_head.cuda()
    if (type(bias_model) == transformers.models.bert.modeling_bert.BertModel):
        bias_model.cpu(),bias_reg_head.cpu()
        final_dict['bias_model'] = bias_model.state_dict()
        final_dict['bias_reg_head'] = bias_reg_head.state_dict()
        bias_model.cuda(),bias_reg_head.cuda()
    return final_dict

def load_dicts(saved_dicts,model,reg_head,reisz_head,propen_head,bias_model,bias_reg_head):
    model.cpu().load_state_dict(saved_dicts['model'])
    reg_head.cpu().load_state_dict(saved_dicts['reg_head'])
    reisz_head.cpu().load_state_dict(saved_dicts['reisz_head'])
    propen_head.cpu().load_state_dict(saved_dicts['propen_head'])
    model.cuda(),reg_head.cuda(),reisz_head.cuda(),propen_head.cuda()
    model.requires_grad_(False),reg_head.requires_grad_(False),reisz_head.requires_grad_(False),propen_head.requires_grad_(False)

    if (type(bias_model) == transformers.models.bert.modeling_bert.BertModel):
        bias_model.cpu().load_state_dict(saved_dicts['bias_model'])
        bias_reg_head.cpu().load_state_dict(saved_dicts['bias_reg_head'])
        bias_model.cuda(),bias_reg_head.cuda()
        bias_model.requires_grad_(False),bias_reg_head.requires_grad_(False)



def get_predictions(train_dataset,bert_pipe,reg_head,reisz_head=None):
    with torch.no_grad():
        texts,labels,treatments,_,cov_group_ids = train_dataset.__getall__()
        treatments = torch.LongTensor(treatments).cuda()
        cov_group_ids = torch.LongTensor(cov_group_ids).cuda()
        labels = torch.LongTensor(labels).cuda()
        bert_rep = bert_pipe(texts).cuda()
        reg_pred = reg_head(bert_rep,treatments)[:,0]
        if (reisz_head):
            reisz_pred = reisz_head(bert_rep)
            bert_rep = get_rep(bert_rep,treatments)
            return bert_rep.cpu(),torch.stack([reg_pred,labels,treatments,cov_group_ids,reisz_pred[:,0],reisz_pred[:,1]],dim=1).cpu()
        else : 
            bert_rep = get_rep(bert_rep,treatments)
            return bert_rep.cpu(),torch.stack([reg_pred,labels,treatments,cov_group_ids],dim=1).cpu()

def get_rep(out,t):
    model_out_dim = int(out.shape[1]/2)
    out_treated = out[:,model_out_dim:]
    out_untreated = out[:,:model_out_dim]
    out_new = torch.stack([out_treated[i] if t[i] == 1 else out_untreated[i] for i in range(len(out_untreated))],dim=0)
    return out_new

def reg_loss_computation(method,reg_pred,this_label,this_treatment,this_group,bias_reg_pred,dfl_gamma,group_weights,group_weights_step_size):
    # REGRESSION Loss Computation
    eval_ = (reg_pred.shape[0]>256)
    if (method == 'groupdro'):
        reg_losses = -(this_label*torch.log(reg_pred) + (1-this_label)*torch.log(1-reg_pred))
        group_losses = [reg_losses[this_group==j].mean() if reg_losses[this_group==j].shape[0]>0 else torch.zeros(1).cuda() for j in range(1,5)]
        group_weights = group_weights*torch.exp(group_weights_step_size*torch.FloatTensor([x.data for x in group_losses])).cuda()
        group_weights = group_weights/group_weights.sum()
        reg_loss = [x*y for x,y in zip(group_losses,group_weights)]
        reg_loss = sum(reg_loss)
    elif ('dfl' in method):
        weights = torch.pow(torch.abs(this_label-bias_reg_pred),dfl_gamma)
        reg_loss = -(this_label*torch.log(reg_pred) + (1-this_label)*torch.log(1-reg_pred))
        reg_loss = reg_loss*weights.data
        reg_loss = reg_loss.mean()
    elif ('poe' in method):
        org_reg_loss = -(this_label*torch.log(reg_pred) + (1-this_label)*torch.log(1-reg_pred)).mean()
        pos_probs = reg_pred*bias_reg_pred.data
        neg_probs = (1-reg_pred)*(1-bias_reg_pred.data)
        combined_probs = pos_probs/(pos_probs+neg_probs)
        reg_loss = -(this_label*torch.log(combined_probs) + (1-this_label)*torch.log(1-combined_probs))
        reg_loss = reg_loss.mean() + org_reg_loss
    else : 
        if (eval_ and 'subsample' in method):
            reg_losses = -(this_label*torch.log(reg_pred) + (1-this_label)*torch.log(1-reg_pred))
            group_losses = [reg_losses[this_group==j].mean() if reg_losses[this_group==j].shape[0]>0 else torch.zeros(1).cuda() for j in range(1,5)]
            group_weights = [(this_label==int(i/2)).float().mean()*(this_treatment==int(i%2)).float().mean() for i in range(4)]
            reg_loss = [x*y for x,y in zip(group_losses,group_weights)]
            reg_loss = sum(reg_loss)
        else : 
            reg_loss = -(this_label*torch.log(reg_pred) + (1-this_label)*torch.log(1-reg_pred)).mean()
        # reg_loss = ((reg_pred-this_label)**2).mean()

    return reg_loss, group_weights


class make_forward:
    def __init__(self,model,tokenizer,dataset,model_out_dim,max_length):
        self.model = model
        self.augment = False
        self.tokenizer = tokenizer
        self.model_out_dim = model_out_dim
        self.max_length = max_length
        self.dataset = dataset
        self.armodel =  ('gpt2' in str(type(model)))

    def random_mask(self,inputs):
        dropout_probs = torch.bernoulli(torch.ones(inputs['input_ids'].shape)*0.85).bool().numpy()

        inputs['input_ids'] = np.where(dropout_probs,inputs['input_ids'],103)
        inputs['attention_mask'] = np.where(dropout_probs,inputs['attention_mask'],0)
        
        return inputs

    def forward(self,documents):
        ourModel = self.model
        logits_our = []
        if ('imdb' in self.dataset):
            documents = ['perceived rating 1/10. '+x for x in documents]+['perceived rating 9/10. '+x for x in documents]
        elif ('christ' in self.dataset):
            documents = [x for x in documents]+['kill. '+x for x in documents]
        elif ('kill' in self.dataset):
            documents = [x for x in documents]+['kill. '+x for x in documents]
        elif ('cebab' in self.dataset):
            documents = [x[0] for x in documents]+[x[1] for x in documents]
        else : 
            documents = ['untreated '+x for x in documents]+['treated '+x for x in documents]
        # documents = [x.replace(' great ', ' bad ') for x in documents]+[x.replace(' bad ', ' great ') for x in documents]
        doc_tokens = self.tokenizer(documents,max_length=self.max_length,
                            add_special_tokens = True,truncation=True,return_tensors = 'np',
                            return_attention_mask = True,padding = 'longest')
        if (self.augment):
            doc_tokens = self.random_mask(doc_tokens)
        
        if not torch.is_grad_enabled():
            ourModel.eval()
            val_batch_size = 100
            with torch.no_grad():
                for i in range(0,len(documents),val_batch_size):
                    if (self.armodel):
                        index_ = doc_tokens['attention_mask'][i:i+val_batch_size].sum(axis=1)-1
                    else : 
                        index_ = np.zeros(doc_tokens['attention_mask'][i:i+val_batch_size].shape[0])
                    temp = ourModel(**{'input_ids':torch.LongTensor(doc_tokens['input_ids'][i:i+val_batch_size]).cuda(),
                                        'attention_mask':torch.LongTensor(doc_tokens['attention_mask'][i:i+val_batch_size]).cuda()}).last_hidden_state
                    temp = temp[np.arange(temp.shape[0]),index_]
                    temp = temp.view(-1,temp.shape[-1])
                    xembs_our = temp.cpu()
                    logits_our.append(xembs_our)
                logits_our = torch.cat(logits_our,dim=0)
            ourModel.train()
        else : 
            if (self.armodel):
                index_ = doc_tokens['attention_mask'].sum(axis=1)-1
            else : 
                index_ = np.zeros(doc_tokens['attention_mask'].shape[0])
            temp = ourModel(**{'input_ids':torch.LongTensor(doc_tokens['input_ids']).cuda(),
                                'attention_mask':torch.LongTensor(doc_tokens['attention_mask']).cuda()}).last_hidden_state
            temp = temp[np.arange(temp.shape[0]),index_]
            temp = temp.view(-1,temp.shape[-1])
            xembs_our = temp
            logits_our.append(xembs_our)
            logits_our = torch.cat(logits_our,dim=0)

        logits_our = logits_our[:,:self.model_out_dim]
        # logits_our = torch.nn.functional.normalize(logits_our[:,:self.model_out_dim],dim=-1)
        bs = int(logits_our.shape[0]/2)
        logits_our = torch.cat([logits_our[:bs],logits_our[bs:]],dim=1)

        return logits_our

class custom_reisz_head(torch.nn.Module):
    def __init__(self,model_out_dim):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,1)).cuda()
        self.model_out_dim= model_out_dim
    def forward(self, out):
        if not torch.is_grad_enabled():
            self.linear.eval()
        # out = torch.cat([self.linear(out[:,:self.model_out_dim].detach()),self.linear(out[:,self.model_out_dim:].detach())],dim=1)
        out = torch.cat([self.linear(out[:,:self.model_out_dim]),self.linear(out[:,self.model_out_dim:])],dim=1)
        if not torch.is_grad_enabled():
            self.linear.train()
        return out

class custom_propen_head(torch.nn.Module):
    def __init__(self,model_out_dim):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,1)).cuda()
        self.model_out_dim= model_out_dim
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, out):
        if not torch.is_grad_enabled():
            self.linear.eval()
        # out = self.linear(out[:,:self.model_out_dim].detach())
        out = self.linear(out[:,:self.model_out_dim])
        out = self.sigmoid(out)
        if not torch.is_grad_enabled():
            self.linear.train()
        return out

class custom_reg_head(torch.nn.Module):
    def __init__(self,model_out_dim,probs=False):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,1)).cuda()
        # self.linear = torch.nn.Sequential(torch.nn.Linear(model_out_dim,1024),torch.nn.ReLU(),
        #                                   torch.nn.Linear(1024,1)).cuda()
        self.model_out_dim= model_out_dim
        self.sigmoid = torch.nn.Sigmoid()
        self.probs = probs
        self.protection_matrix = torch.eye(model_out_dim).cuda()

    def forward(self,out,t):
        if not torch.is_grad_enabled():
            self.linear.eval()
        out_treated = out[:,self.model_out_dim:]
        out_untreated = out[:,:self.model_out_dim]
        out_new = torch.stack([out_treated[i] if t[i] == 1 else out_untreated[i] for i in range(len(out_untreated))],dim=0)
        out = self.linear(out_new@self.protection_matrix)
        if (self.probs):
            out = self.sigmoid(out)
        if not torch.is_grad_enabled():
            self.linear.train()
        return out