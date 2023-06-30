import torch
import wandb
from custom_functions import *

class evaluator:
    def __init__(self,test_dataset,bert_pipe,bias_bert_pipe,reg_head,bias_reg_head,reisz_head,propen_head,config_weights,params):
        self.test_dataset = test_dataset
        self.bert_pipe = bert_pipe
        self.bias_bert_pipe = bias_bert_pipe
        self.reg_head = reg_head
        self.bias_reg_head = bias_reg_head
        self.reisz_head = reisz_head
        self.propen_head = propen_head
        self.config_weights = config_weights
        self.params = params
        self.bce_loss_fn = torch.nn.BCELoss()
        self.best_reg_preds = 0
        self.best_reisz_preds = 0
        self.best_propen_preds = 0
        self.best_reg_loss = float('inf')
        self.best_reisz_loss = float('inf')
        self.best_propen_loss = float('inf')

    def get_best_reisz(self):
        _,labels,_,_,_ = self.test_dataset.__getall__()
        direct_est = self.best_direct_est
        reg_pred = self.best_reg_preds
        dr_terms = (self.best_reisz_preds*(labels-reg_pred))
        dr_terms = dr_terms.mean()
        dr_reisz = (direct_est + dr_terms.mean().cpu())
        return dr_reisz

    def evaluate(self,group_weights,prepend=''):
        bert_pipe = self.bert_pipe
        texts,labels,treatments,groups,cov_groups = self.test_dataset.__getall__()
        
        labels = torch.FloatTensor(labels).cuda()
        treatments = torch.LongTensor(treatments).cuda()
        groups = torch.LongTensor(groups).cuda()
        cov_groups = torch.LongTensor(cov_groups).cuda()

        bert_rep = bert_pipe(texts).cuda()
        # Wipe Treatment Information for "Remove Token from Input"
        if (self.params.method == 'removetoken'):
            reg_pred = self.reg_head(bert_rep,torch.zeros(treatments.shape).cuda())[:,0]
        else : 
            reg_pred = self.reg_head(bert_rep,treatments)[:,0]

        
        dict_to_log = dict({})

        bias_reg_pred = None
        if ('bias' in self.params.method or 'success' in self.params.method):
            bias_bert_rep = self.bias_bert_pipe(texts).cuda()
            bias_reg_pred = self.bias_reg_head(bias_bert_rep,treatments)[:,0]

            isCorrect = (labels == (bias_reg_pred>0.5).float()).float()            
            accuracies_group = [isCorrect[torch.where(cov_groups==j)].mean() for j in range(1,5)]
            accuracies_label = [isCorrect[torch.where(labels==j)].mean() for j in range(2)]
            accuracies_treatment = [isCorrect[torch.where(treatments==j)].mean() for j in range(2)]

            print ("Bias Accuracy Total: %0.2f"%(isCorrect.mean()))

            for j in range(1,5):
                dict_to_log['%sbias_accuracies/group%d'%(prepend,j)] = accuracies_group[j-1]
            for j in range(2):
                dict_to_log['%sbias_accuracies/label%d'%(prepend,j)] = accuracies_label[j]
                dict_to_log['%sbias_accuracies/treatment%d'%(prepend,j)] = accuracies_treatment[j]
    
            dict_to_log['%sbias_accuracies/total'%prepend] = isCorrect.mean()
            
        reisz_pred = self.reisz_head(bert_rep)
        propen_pred = self.propen_head(bert_rep)

        isCorrect = (labels == (reg_pred>0.5).float()).float()            
        accuracies_group = [isCorrect[torch.where(cov_groups==j)].mean() for j in range(1,5)]
        accuracies_label = [isCorrect[torch.where(labels==j)].mean() for j in range(2)]
        accuracies_treatment = [isCorrect[torch.where(treatments==j)].mean() for j in range(2)]

        print ("Accuracies Groups: %0.2f, %0.2f, %0.2f, %0.2f"%(accuracies_group[0],accuracies_group[1],accuracies_group[2],accuracies_group[3]))
        print ("Accuracies Labels: %0.2f, %0.2f"%(accuracies_label[0],accuracies_label[1]))
        print ("Accuracies Treatments: %0.2f, %0.2f"%(accuracies_treatment[0],accuracies_treatment[1]))
        print ("Accuracy Total: %0.2f"%(isCorrect.mean()))

        for j in range(1,5):
            dict_to_log['%snum_eval_samples/num_group%d'%(prepend,j)] = torch.where(cov_groups==j)[0].shape[0]
            dict_to_log['%saccuracies/group%d'%(prepend,j)] = accuracies_group[j-1]
        for j in range(2):
            dict_to_log['%snum_eval_samples/num_label%d'%(prepend,j)] = torch.where(labels==j)[0].shape[0]
            dict_to_log['%saccuracies/label%d'%(prepend,j)] = accuracies_label[j]
            dict_to_log['%snum_eval_samples/num_treatment%d'%(prepend,j)] = torch.where(treatments==j)[0].shape[0]
            dict_to_log['%saccuracies/treatment%d'%(prepend,j)] = accuracies_treatment[j]

        dict_to_log['%saccuracies/total'%prepend] = isCorrect.mean()
        
        propen_loss = self.bce_loss_fn(propen_pred[:,0],treatments.float())

        reisz_loss_indices = (-2*(reisz_pred[:,1]-reisz_pred[:,0])+(torch.gather(reisz_pred,1,treatments[:,None])[:,0]**2))
        reisz_loss_treated = reisz_loss_indices[treatments==1].mean()
        reisz_loss_untreated = reisz_loss_indices[treatments==0].mean()
        print ('reisz treated loss : %0.3f,reisz untreated loss : %0.3f'%(reisz_loss_treated,reisz_loss_untreated))

        reisz_loss = reisz_loss_indices.mean()
        reg_loss, _ = reg_loss_computation(self.params.method,reg_pred,labels,treatments,groups,bias_reg_pred,self.params.gamma,group_weights,0)

        # BIAS Computation
        bias_reg_loss =  0
        if ('bias' in self.params.method):
            if ('demo' in self.params.method):
                bias_reg_loss = self.bce_loss_fn(bias_reg_pred,treatments.float())
            else : 
                bias_reg_loss = self.bce_loss_fn(bias_reg_pred,labels)
        elif ('success' in self.params.method):
            correct = ((reg_pred>0.5).int() == labels).float()
            bias_reg_loss = self.bce_loss_fn(bias_reg_pred,correct)


        direct_est = (self.reg_head(bert_rep,torch.ones(bert_rep.shape[0]))[:,0]-
                    self.reg_head(bert_rep,torch.zeros(bert_rep.shape[0]))[:,0])
        ate_loss = ((direct_est.mean()-self.params.target)**2).mean()

        loss = self.config_weights['reisz_weight']*reisz_loss+self.config_weights['reg_weight']*reg_loss+self.config_weights['propen_weight']*propen_loss # \
            # + self.config_weights['cer_weight']*ate_loss+self.config_weights['bias_reg_weight']*bias_reg_loss

        print ('reg loss %0.3f,reisz loss %0.3f,propen loss %0.3f'%(reg_loss,reisz_loss,propen_loss))
        
        dict_to_log.update({
            '%seval/reg_loss'%prepend:reg_loss,
            '%seval/reisz_loss'%prepend:reisz_loss,
            '%seval/propen_loss'%prepend:propen_loss,
            '%seval/cer_loss'%prepend:ate_loss,
            '%seval/bias_reg_loss'%prepend:bias_reg_loss,
            '%seval/loss'%prepend:loss,
            '%seval/reisz_loss_treated'%prepend:reisz_loss_treated,
            '%seval/reisz_loss_untreated'%prepend:reisz_loss_untreated
        })

        ## Storing Best Reisz and Reg Preds and Propens

        if (reisz_loss<self.best_reisz_loss):
            self.best_reisz_loss = float(reisz_loss)
            self.best_reisz_preds = torch.gather(reisz_pred,1,treatments[:,None])[:,0]

        if (propen_loss<self.best_propen_loss):
            self.best_propen_loss = float(propen_loss)
            self.best_propen_preds = propen_pred

        if (reg_loss<self.best_reg_loss):
            self.best_reg_loss = float(reg_loss)
            self.best_reg_preds = reg_pred
            self.best_direct_est = direct_est.mean().cpu()

        # dataset_name_ = 'christv3'
        # torch.save(cov_groups,'../saved_reisz/groups_%s.pt'%dataset_name_)
        # torch.save(labels,'../saved_reisz/labels_%s.pt'%dataset_name_)
        # torch.save(self.best_reisz_preds,'../saved_reisz/reisz_pred_%s.pt'%dataset_name_)
        # torch.save(self.best_reg_preds,'../saved_reisz/reg_pred_%s.pt'%dataset_name_)
        # torch.save(self.best_direct_est,'../saved_reisz/direct_pred_%s.pt'%dataset_name_)

        # print ("Untreated Covariate samples after being Treated %0.3f"%self.reg_head(bert_rep,torch.ones(bert_rep.shape[0]))[:,0][treatments==0].mean())
        # print ("Untreated Covariate samples after being Untreated %0.3f"%self.reg_head(bert_rep,torch.zeros(bert_rep.shape[0]))[:,0][treatments==0].mean())
        # print ("Treated Covariate samples after being Treated %0.3f"%self.reg_head(bert_rep,torch.ones(bert_rep.shape[0]))[:,0][treatments==1].mean())
        # print ("Treated Covariate samples after being Untreated %0.3f"%self.reg_head(bert_rep,torch.zeros(bert_rep.shape[0]))[:,0][treatments==1].mean())

        direct_est = self.best_direct_est

        ips_reisz = (self.best_reisz_preds*labels).mean()
        dr_terms = (self.best_reisz_preds*(labels-self.best_reg_preds))
        dr_terms = dr_terms.mean()

        dr_reisz = (direct_est + dr_terms.mean().cpu())

        print ('Direct predicted effect : ','%0.3f'%(direct_est))
        print ('Reisz ips predicted effect : ','%0.3f'%ips_reisz)    
        print ('Reisz dr predicted effect : ','%0.3f'%dr_reisz)
        
        dict_to_log['%seffect/dr'%prepend] = dr_reisz
        dict_to_log['%seffect/direct'%prepend] = direct_est
        dict_to_log['%seffect/ips'%prepend] = ips_reisz

        
        reisz_pred_ = torch.gather(reisz_pred,1,treatments[:,None])[:,0]
        reiszes = [reisz_pred_[torch.where(cov_groups==j)].mean() for j in range(1,5)]
        reiszes_square = [(reisz_pred_[torch.where(cov_groups==j)]**2).mean() for j in range(1,5)]
        print ('Reisz %0.3f, %0.3f, %0.3f, %0.3f'%(reiszes[0],reiszes[1],reiszes[2],reiszes[3]))
        for i in range(1,5):
            dict_to_log['%sreisz/group%d'%(prepend,i)]=reiszes[i-1]
            dict_to_log['%sreisz_square/group%d'%(prepend,i)]=reiszes_square[i-1]

        errors_complete = labels-reg_pred

        for i in range(1,5):
            dict_to_log['%serrors/group%d_factual'%(prepend,i)]=(errors_complete[torch.where(cov_groups==i)]**2).mean()
        dict_to_log['%serrors/total_factual'%prepend]=(errors_complete**2).mean()

        for i in range(1,5):
            dict_to_log['%serrors(avg)/group%d_factual'%(prepend,i)]=(errors_complete[torch.where(cov_groups==i)]).mean()
        dict_to_log['%serrors(avg)/total_factual'%prepend]=(errors_complete).mean()






        for i in range(1,5):
            dict_to_log['%spropensities/probs_group%d'%(prepend,i)]=propen_pred[:,0][torch.where(cov_groups==i)].mean()
            
        clipped_treatment_pred_multiplicatives = (treatments*(1/self.best_propen_preds[:,0]) - (1-treatments)*(1/(1-self.best_propen_preds[:,0]))).clamp(min=-10).clamp(max=10)
        clipped_ips_treatment = (clipped_treatment_pred_multiplicatives*labels).mean().cpu()
        clipped_dr_treatment = (direct_est + (clipped_treatment_pred_multiplicatives*(labels-self.best_reg_preds)).mean().cpu())

        unclipped_treatment_pred_multiplicatives = (treatments*(1/self.best_propen_preds[:,0]) - (1-treatments)*(1/(1-self.best_propen_preds[:,0])))
        unclipped_ips_treatment = (unclipped_treatment_pred_multiplicatives*labels).mean().cpu()
        unclipped_dr_treatment = (direct_est + (unclipped_treatment_pred_multiplicatives*(labels-self.best_reg_preds)).mean().cpu())

        print ('Propensities clipped,unclipped ips predicted effect : ','%0.3f'%clipped_ips_treatment,'%0.3f'%unclipped_ips_treatment)
        print ('Propensities clipped,unclipped dr predicted effect : ','%0.3f'%clipped_dr_treatment,'%0.3f'%unclipped_dr_treatment)

        dict_to_log.update({
            '%spropensities/unclipped_ips'%prepend: unclipped_ips_treatment,
            '%spropensities/unclipped_dr'%prepend: unclipped_dr_treatment,
            '%spropensities/clipped_ips'%prepend: clipped_ips_treatment,
            '%spropensities/clipped_dr'%prepend: clipped_dr_treatment,
        })

        wandb.log(dict_to_log)

        return float(loss.cpu().numpy())
