import os
from unittest.util import _MAX_LENGTH
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import torch.nn.functional as F
import copy

from tools import prob2pr, plot_single_rel_graph, get_best_f, get_accuracy
from tools import AverageMeter, accuracy, temp_anneal_binary_k

from sklearn.metrics import auc

from torch.distributions import Categorical

import matplotlib.pyplot as plt

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def model_performance(model, dataloader):
    criterion = nn.BCELoss().to(device)
    
    probs = []
    labels = []

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # Compute prediction and loss
            mask = data.op_mask
            pred = model(data.inp, data.mask, data.prob)

            tgt = data.tgt
            pred = torch.masked_select(pred, mask)
            tgt = torch.masked_select(tgt, mask)
            
            if pred.size() != tgt.size():
                print('prediction size unequal to target size !')
          
            probs.extend(pred.tolist())
            labels.extend(tgt.tolist())
    
    cal_p1, cal_loss1, temp1 = temp_anneal_binary_k(probs, labels,temp = np.linspace(1.05,1.55, 50))
    cal_p, cal_loss, temp = temp_anneal_binary_k(cal_p1, labels, temp = np.linspace(temp1-0.01, temp1+0.01, 50))
    
    print(f'calibrated loss:{cal_loss}, optimal temperature:{temp}')

    p,r,t = prob2pr(probs, labels)
    auc1 = auc(r,p)
    acc, f = get_accuracy(probs, labels)
    loss = criterion(torch.tensor(probs), torch.tensor(labels))
    
    ece = plot_single_rel_graph(probs, labels)
    best_f, best_acc, _ = get_best_f(p, r, t)
    print(f'ece:{ece},auc:{auc1}, acc:{acc}, f"{f}, loss:{loss}, best f:{best_f}, best_acc:{best_acc}')
        
    return auc1, loss, best_f, cal_loss

class SpeechDataGenerator(Dataset):
    def __init__(self, datapath:list):
        # read from .npy file and get the list of dictionary
        datalist = []
        for path in datapath:
            data = np.load(path, allow_pickle=True).tolist()
            datalist.extend(data)
        self.data = datalist
        self.max_length = 128
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sentence = self.data[idx]

        v_all = sentence['embedding']
        v_all = torch.from_numpy(np.asarray(v_all, dtype=np.float32)).detach().to(device)
        sentence_len = v_all.shape[0]
            
        target = sentence['op_tgts']
        target = torch.from_numpy(np.asarray(target, dtype=np.float32)).detach().to(device)
        # target = torch.cat((torch.unsqueeze(target,1), torch.unsqueeze(target,1)), -1)
        
        mask = sentence['mask']
        mask = torch.from_numpy(np.asarray(mask, dtype='b')).detach().to(device)
        
        prob = np.asarray(sentence['prob'], dtype=np.float32)
        prob = torch.from_numpy(prob).detach().to(device)
        # print(mask.type())
        # print(mask.size())
        # mask = torch.cat((torch.unsqueeze(mask,1), torch.unsqueeze(mask,1)), -1)
        # print(f'mask:{mask.size()}')
        # print(f'mask:{mask}')

        return [v_all, target, mask, prob]
    
class ConfidenceModelAttn(nn.Module):
    # applies attention to logits
    def __init__(self, max_len = 128, label_len = 768, qkv_len = 768, dropout=0.2):  # embedding length uncertain
        super().__init__()
        self.max_len = max_len # maximum sequence length
        self.label_len = label_len # number of label classes
        self.qkv_len = qkv_len # query, key and value length
        self.num_heads = 1
        
        # self.value = nn.Sequential(
        #     nn.Linear(self.label_len, self.qkv_len),
        #     nn.ReLU(),
        # )

        # self.dropout=nn.Dropout(p=self.dp_p)
        self.attn = nn.MultiheadAttention(embed_dim=self.qkv_len, num_heads= self.num_heads, batch_first=True)
        
        self.attn_combine = nn.Sequential(
                                nn.Linear(self.label_len + self.qkv_len+1, 1),
                                # nn.ReLU(),
                                nn.Sigmoid(),
                                )
        
        self.fc_layer = nn.Sequential(
                                nn.Linear(self.label_len,1), # todo: input size uncertain
                                # nn.ReLU(),
                                # nn.Linear(5002,200),
                                # nn.ReLU(),
                                # nn.Linear(200,1),
                                nn.Sigmoid()
                                # nn.Softmax()
                                )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                # nn.init.constant_(m.weight, 0.0)
                m.bias.data.normal_(std=0.1)
                # m.bias.data.fill_(0.0)
                m.weight.data[0][-1] = 1.0
                
        self.attn_combine.apply(init_weights)
                                     
    def forward(self, batch, key_mask, prob):
        all_v = batch
        # query = self.query(all_v)
        # key = self.key(all_v)
        value = all_v
        # value = self.value(all_v)
        
        S = key_mask.size(dim=1) # sequence length
        # construct attention mask
        attn_mask1 = key_mask.unsqueeze(2).repeat(1,1,S)
        attn_mask2 = key_mask.unsqueeze(1).repeat(1,S,1)
        attn_mask = attn_mask1*attn_mask2
        
        # convert mask (float tensor) to bool tensor
        key_mask = key_mask <0.5
        attn_mask = attn_mask <0.5
        
        # multihead_attn = nn.MultiheadAttention(self.qkv_len, self.num_heads)
        # attn_output, attn_output_weights = self.attn(query, key, value, key_padding_mask = key_mask) # all attention output for the sequence
        attn_output, attn_output_weights = self.attn(value, value, value, attn_mask = attn_mask)
        
        attn_combined = self.attn_combine(torch.cat((all_v, attn_output, prob.unsqueeze(-1)), -1))
        
        out = attn_combined.squeeze()
        # out = self.fc_layer(attn_combined).squeeze()

        return out

class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        max_length = max(stc.size()[0] for stc in transposed_data[0])
        
        batch_padded_inputs = []
        batch_attn_masks = []
        batch_padded_tgts = []
        batch_op_masks = []
        batch_prob = []

        for sen, tgt, mask, prob in zip(transposed_data[0], transposed_data[1], transposed_data[2], transposed_data[3]):
            # How many pad tokens do we need to add?
            num_pads = max_length - sen.size()[0]

            # Add `num_pads` padding tokens to the end of the sequence.
            padded_input = F.pad(sen,(0,0,0,num_pads))
            padded_tgt = F.pad(input = tgt,pad = (0,num_pads), value = 0)
            prob = F.pad(input = prob,pad = (0,num_pads), value = 0)

            # Define the attention mask--it's just a `1` for every real token
            # and a `0` for every padding token.
            attn_mask = [1] * len(sen) + [0] * num_pads
            attn_mask = torch.from_numpy(np.asarray(attn_mask, dtype=np.float32)).detach().to(device)
            
            output_mask = F.pad(input = mask,pad = (0,num_pads), value = 0)
            output_mask = output_mask >0
            
            # Add the padded results to the batch.
            batch_padded_inputs.append(padded_input)
            batch_attn_masks.append(attn_mask)
            batch_padded_tgts.append(padded_tgt)
            batch_op_masks.append(output_mask)
            batch_prob.append(prob)
            
        self.inp = torch.stack(batch_padded_inputs, 0)
        self.tgt = torch.stack(batch_padded_tgts, 0)
        self.mask = torch.stack(batch_attn_masks)
        self.op_mask = torch.stack(batch_op_masks)
        self.prob = torch.stack(batch_prob)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        self.mask = self.mask.pin_memory()
        self.op_mask = self.mask.pin_memory()
        self.prob = self.prob.pin_memory()
        return self
    
def collate_wrapper(batch):
    return CustomBatch(batch)

def train_loop(dataloader, model, loss_fn, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()

    tot_1s = 0
    tot_0s = 0
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(data.inp, data.mask, data.prob)
        
        pred = torch.masked_select(pred, data.op_mask)
        tgt = torch.masked_select(data.tgt, data.op_mask)  
        tot_1s+=torch.sum((tgt == True).int())
        tot_0s+=torch.sum((tgt == False).int())
        # import pdb; pdb.set_trace()
        
        try:
            loss = loss_fn(pred, tgt)
        except:
            # except only one value batch, make sure tgt and pred have the same sizes
            loss = loss_fn(torch.squeeze(tgt), torch.squeeze(pred))
            
        prob = torch.masked_select(data.prob, data.op_mask)
        
        # Backpropagation
        optimizer.zero_grad()
        import pdb;pdb.set_trace()
        loss.backward()
        optimizer.step()
        
        acc = accuracy(pred,tgt)
        
        accs.update(acc,torch.sum((data.op_mask == True).int()))
        losses.update(loss, torch.sum((data.op_mask == True).int()))

    print(f'Train loss {losses.avg}, accuracy {accs.avg}')
    print(f'1s:{tot_1s}, 0s: {tot_0s}')
    return losses.avg

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    tot_1s = 0
    tot_0s = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(data.inp, data.mask, data.prob)
            # pred = pred*data.op_mask
            pred = torch.masked_select(pred, data.op_mask)
            tgt = torch.masked_select(data.tgt, data.op_mask)
            
            tot_1s+=torch.sum((tgt == True).int())
            tot_0s+=torch.sum((tgt == False).int()) 
            try:
                loss = loss_fn(pred, tgt)
                # print(f'pred:{pred}')
            except:
                # except only one value batch, make sure tgt and pred have the same sizes
                loss = loss_fn(torch.squeeze(tgt), torch.squeeze(pred))
                
            acc = accuracy(pred,tgt)
            accs.update(acc,torch.sum((data.op_mask == True).int()))
            losses.update(loss, torch.sum((data.op_mask == True).int()))
           
        print(f'Test loss {losses.avg}, accuracy {accs.avg}')
    return losses.avg



def main(args):
    idx = args.train_file_idx
    # idx=[0,3]

    train_file = [ '/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train_45_2.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train_45_1.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train_50_2.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train_8_.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train_10_.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/train_selfAttn.npy',
                  '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/train_selfAttn2.npy']
    
    train_file = [train_file[i] for i in idx]
    print(train_file)
    
    # test_file = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/self_attn_test.npy',]
    test_file=['/home/alta/BLTSpeaking/exp-zl437/demo/gector/CEM/data/self_attn/test.npy']
                #  '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_selfAttn.npy']
    # test_file = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_selfAttn.npy',]
    print(test_file)

    batch_size = 120

    train_dataset = SpeechDataGenerator(test_file) 
    test_dataset = SpeechDataGenerator(test_file)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
    
    confidence_model = ConfidenceModelAttn(qkv_len=args.num_para).to(device)

    criterion = nn.BCELoss().to(device) # lr = 0.3e-5
    optimizer = optim.Adam(confidence_model.parameters(), lr=0.5e-5, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    
    train_loss = []
    test_loss = []
    aucs = []
    best_fs = []
    models = []
    cal_losses = []
    
    if args.load_pre_trained==1:
        checkpoint = torch.load(args.model_path, map_location=device)
        confidence_model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

        test_loss.append(loss)
        aucs.append(checkpoint['auc'])
        best_fs.append(checkpoint['fscore'])
        
        models.append(checkpoint['model_state_dict'])
    epochs = 100 # 170 
    # import pdb; pdb.set_trace()
    # return  sacct -j 15125420 --format=Elapsed

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_ls = train_loop(train_dataloader, confidence_model, criterion, optimizer)
        tst_ls = test_loop(test_dataloader, confidence_model, criterion)
        auc, loss, best_f, cal_loss = model_performance(confidence_model, test_dataloader) # print model performance
        
        model_copy = copy.deepcopy(confidence_model)
        aucs.append(auc)
        best_fs.append(best_f)
        train_loss.append(tr_ls)
        test_loss.append(tst_ls)
        cal_losses.append(cal_loss)
        models.append(model_copy.state_dict())
    print("Done!")
    
    auc_array = np.array(aucs)
    idx = np.argmax(auc_array)
    
    print(f'best auc:{auc_array[idx]}, loss:{test_loss[idx]}, best f:{best_fs[idx]}')
    
    if args.model_path:
        torch.save({
                'model_state_dict': models[idx],
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss[idx],
                'auc':auc_array[idx],
                'fscore':best_fs[idx],
                }, args.model_path)
    
    plt.plot(range(len(test_loss)), test_loss)
    plt.plot(range(len(aucs)), aucs)
    plt.plot(range(len(best_fs)), best_fs)
    plt.plot(range(len(cal_losses)), cal_losses)
    
    plt.xlabel('epochs')

    plt.legend(['test loss', 'AUC', 'F 0.5', 'cal loss'])
    plt.savefig('/home/zl437/rds/hpc-work/gector/CEM/loss_epoch.png')
    
    # torch.save(confidence_model, '/home/zl437/rds/hpc-work/gector/CEM/model/cm_self_attn_fce.pt')

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_idx',
                        help='idx of path to the train file.', nargs='+',
                        type=int,
                        default=[0]
                        )
    
    parser.add_argument('--model_path',
                        help='Path to save output model.',
                        default=None
                        )
    
    parser.add_argument('--num_para',
                        help='Number of parameters in the first linear layer.',
                        type=int,
                        default=768
                        )
    
    parser.add_argument('--load_pre_trained',
                        type=int,
                        default=0
                        )
    
    args = parser.parse_args()
    main(args)