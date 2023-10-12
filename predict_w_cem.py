from tqdm import tqdm
import argparse

from utils.preprocess_data import align_sequences, read_parallel_lines
from utils.helpers import read_one_file_lines
from gector.gec_model import GecBERTModel
import numpy as np
from numpy.random import default_rng

from CEM.cem_model import ConfidenceModel
from CEM.cem_w_self_a import ConfidenceModelAttn
from gen_cem_data import my_equal_edits
import torch
import os

import torch.nn.functional as F

feature_length=768
device = "cuda" if torch.cuda.is_available() else "cpu"

def my_predict_sing_stc(stc, model, masking, masking_idxs=None):
    '''
    predict single sentence, return output sentence, probability lists and index list
    '''
    batch = [stc.split()]
    sequences = model.preprocess(batch)

    probabilities, idxs, error_probs, all_error_probs, logits_labels, embeddings = model.predict(sequences, masking, masking_idxs)
    
    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs, error_probs)

    result_line = [" ".join(x) for x in pred_batch]

    return result_line[0] ,probabilities[0], idxs[0], all_error_probs[0], logits_labels[0], embeddings[0]

def predict_single_edit_cem(embeddings, prob, threshold = 0.5, model_path_folder='/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce'):
    '''input embedding and prob for single edit, return True if correct/ false if not'''
    files = os.listdir(model_path_folder)
    model_lists = []
    
    for f in files:
        model_path = model_path_folder+'/'+f
        
        model = ConfidenceModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_lists.append(model)
    
    with torch.no_grad():
        # Compute prediction
        features = np.asarray(embeddings, dtype=np.float32)
        prob = np.asarray(prob, dtype=np.float32)
        
        features = torch.from_numpy(features).detach().to(device)
        prob = torch.from_numpy(prob).detach().to(device)
        features = torch.cat((features, prob.unsqueeze(dim=0)))
        
        preds = []
        for m in model_lists:
            pred = m(features)
            preds.append(pred.unsqueeze(dim=0))
            
        # average over all preds
        preds = torch.cat(preds, dim=0)
        pred = torch.mean(preds, dim=0)
    
    if pred>threshold:
        return True
    else:
        return False

def predict_stc_attn_cem(embeddings, mask, probs, model_path_folder, threshold):
    files = os.listdir(model_path_folder)
    model_lists = []
    
    for f in files:
        model_path = model_path_folder+'/'+f
        
        model = ConfidenceModelAttn().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_lists.append(model)
        
    with torch.no_grad():
        v_all = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).detach().to(device)
        mask = torch.from_numpy(np.asarray(mask, dtype='b')).detach().to(device)
        probs = torch.from_numpy(np.asarray(probs, dtype=np.float32)).detach().to(device)
        
        attn_mask = [1] * len(v_all)
        attn_mask = torch.from_numpy(np.asarray(attn_mask, dtype=np.float32)).detach().to(device)
        
        import pdb;pdb.set_trace()
        pred = []
        for m in model_lists:
            cem_p = m(v_all, attn_mask, probs)
            pred.append(cem_p.unsqueeze(dim=0))

        # average over all preds
        preds = torch.cat(preds, dim=0)
        pred = torch.mean(preds, dim=0)
        
        pred = pred[mask>0.5]
        pred = pred < threshold # position with True should be set to be the keep tag

    return pred

       
def get_single_stc_attn(source_sent, model, cem_folder, masking, t = 0.0, masking_idxs=None):
    '''predict with CEM'''
    # predict source sentence
    batch = [source_sent.split()]
    sequences = model.preprocess(batch)
    probabilities, idxs_, error_probs, all_error_probs, logits_labels, embeddings = model.predict(sequences, masking, masking_idxs)
    # import pdb;pdb.set_trace()
    
    probs = probabilities[0]
    idxs = idxs_[0]
    embeddings = embeddings[0]

    vocab = model.vocab
    # pick only the probabilities and embeddings at edits indices
    edits_pred = [[(i-1,i),vocab.get_token_from_index(idxs[i], namespace = 'labels')] for i in range(len(idxs)) if (idxs[i] != 0)] #  and idxs[i] != 5000
    edits_idx = [i for i in range(len(idxs)) if idxs[i] != 0]
    # edit_logits_labels = [logits_labels[i] for i in edits_idx]
    
    # for cme training
    mask_for_cme = [0]*len(idxs) # mask to show where is editted
    
    for pred_e in edits_pred:
        # print(pred_e)
        position = pred_e[0][1]
        mask_for_cme[position]=1 # ?

    predict_stc_attn_cem(embeddings, mask_for_cme, probs)


    for pred_e, idx in zip(edits_pred, edits_idx):
        # if low confidence score, set the tag to keep
        if not predict_single_edit_cem(embeddings[idx], probs[idx], threshold = t, model_path_folder=cem_folder):
            idxs[idx]=0

    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs_, error_probs)

    result_line = [" ".join(x) for x in pred_batch]
    
    return result_line[0]

def get_single_stc_logits_only(source_sent, model, cem_folder, masking, t = 0.0, masking_idxs=None):
    '''predict with CEM'''
    # predict source sentence
    batch = [source_sent.split()]
    sequences = model.preprocess(batch)
    probabilities, idxs_, error_probs, all_error_probs, logits_labels, embeddings = model.predict(sequences, masking, masking_idxs)
    
    probs = probabilities[0]
    idxs = idxs_[0]
    embeddings = embeddings[0]

    vocab = model.vocab
    # pick only the probabilities and embeddings at edits indices
    edits_pred = [[(i-1,i),vocab.get_token_from_index(idxs[i], namespace = 'labels')] for i in range(len(idxs)) if (idxs[i] != 0)] #  and idxs[i] != 5000
    edits_idx = [i for i in range(len(idxs)) if idxs[i] != 0]
    # edit_logits_labels = [logits_labels[i] for i in edits_idx]

    for pred_e, idx in zip(edits_pred, edits_idx):
        # if low confidence score, set the tag to keep
        if not predict_single_edit_cem(embeddings[idx], probs[idx], threshold = t, model_path_folder=cem_folder):
            idxs[idx]=0

    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs_, error_probs)

    result_line = [" ".join(x) for x in pred_batch]
    
    return result_line[0]

def get_single_stc_prob_only(source_sent, model, masking, t = 0.0, masking_idxs=None):
    '''predict with probability'''
    # predict source sentence
    batch = [source_sent.split()]
    sequences = model.preprocess(batch)
    probabilities, idxs_, error_probs, all_error_probs, logits_labels, embeddings = model.predict(sequences, masking, masking_idxs)
    
    probs = probabilities[0]
    idxs = idxs_[0]
    embeddings = embeddings[0]
    
    logits_labels = torch.tensor(logits_labels)
    
    probs_cal = F.softmax(logits_labels/1.325,dim=-1, dtype=torch.float64)
    max_vals, idx = torch.max(probs_cal, dim=-1)
    # import pdb;pdb.set_trace()

    vocab = model.vocab
    # pick only the probabilities and embeddings at edits indices
    edits_pred = [[(i-1,i),vocab.get_token_from_index(idxs[i], namespace = 'labels')] for i in range(len(idxs)) if (idxs[i] != 0)] #  and idxs[i] != 5000
    edits_idx = [i for i in range(len(idxs)) if idxs[i] != 0]
    # edit_logits_labels = [logits_labels[i] for i in edits_idx]
    

    for pred_e, idx in zip(edits_pred, edits_idx):
        # if low confidence score, set the tag to keep
        if not max_vals[0,idx]>=t:
            idxs[idx]=0

    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs_, error_probs)

    result_line = [" ".join(x) for x in pred_batch]
    
    return result_line[0]

def predict_w_cem(source_file, model, cem_folder, masking = None, t=0.0, interval=[0.0,1.0], ite = 1):
    '''
    input: source file and target file path, model
    output: edits indices, all embeddings, all logits for labels, edits logits, target
    
    masking: percent of masking in embedding space, masking_idx: position of masking, optional
    '''
    source_data = read_one_file_lines(source_file)
    source_data = source_data[int(interval[0]*len(source_data)):int(interval[1]*len(source_data))]

    result_lines = []
    compare_lines = []
    
    for j in range(len(source_data)):
        source_sent = source_data[j]
        if j%50==0:
            print(j)
        
        prob_source = source_sent
        logits_source = source_sent
        for i in range(ite):
            result_line=get_single_stc_logits_only(
                    logits_source, model, cem_folder,
                    masking, t=t)
            result_line0=get_single_stc_prob_only(prob_source, model, masking, t)
            
            logits_source = result_line
            prob_source = result_line0
            
        result_lines.append(result_line)
        compare_lines.append(result_line0)

    return result_lines, compare_lines

def predict_w_sf_probs(source_file, model, t=0.0, interval=[0.0,1.0]):
    source_data = read_one_file_lines(source_file)
    source_data = source_data[int(interval[0]*len(source_data)):int(interval[1]*len(source_data))]

    compare_lines = []
    
    for j in range(len(source_data)):
        source_sent = source_data[j]
        if j%50==0:
            print(j)

        result_line0=get_single_stc_prob_only(source_sent, model, None, t)
        compare_lines.append(result_line0)

    return compare_lines


def predict_for_file(source_file, target_file, model, output_path, masking = None):
    source_data, target_data = read_parallel_lines(source_file, target_file)

    data = []
    for source_sent in source_data:
        
        pred_sent, probs, idxs, _, logits_labels, all_embeddings = my_predict_sing_stc(source_sent, model, masking) # make predictions
        
        data.append(pred_sent)
        # import pdb; pdb.set_trace()
    
    with open(output_path, 'w') as f:
        f.write("\n".join(data) + '\n')
        
    return data

def write_to_file(result_lines, outputpath):
    with open(outputpath, 'w') as f:
        f.write("\n".join(result_lines) + '\n')
    return

def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights,)
    
    print(args.masking, args.output_txt_file, args.interval)
    # result_lines, compare_lines = predict_w_cem(
    #     args.input_file, model, 
    #     cem_folder='/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce_masked',
    #     masking=args.masking, t=args.prob_threshold,
    #     interval=args.interval, ite=args.iterations)
    compare_lines = predict_w_sf_probs(
        args.input_file, model,t=args.prob_threshold,interval=args.interval,
    )
    if args.output_txt_file:
        # write_to_file(result_lines, '/home/zl437/rds/hpc-work/gector/conll14st-test-data/preds_w_mask/predictions_cem/'+args.output_txt_file)
        write_to_file(compare_lines, '/home/zl437/rds/hpc-work/gector/bea-data/test_w_mask/predictions_cal/'+args.output_txt_file)

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default=['/home/zl437/rds/hpc-work/gector/data/roberta_1_gectorv2.th']
                        )
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='/home/zl437/rds/hpc-work/gector/data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        # default='/home/zl437/rds/hpc-work/gector/fce-data/source-test.txt',
                        # default='/home/zl437/rds/hpc-work/gector/conll14st-test-data/conlltest.txt'
                        default='/home/zl437/rds/hpc-work/gector/bea-data/test/bea2019test.txt'
                        # default='/home/zl437/rds/hpc-work/gector/bea-data/nucle/source-test.txt'
                        )
    parser.add_argument('--prob_threshold',
                        help='When predicint with cem, the threshold to cut of edits',
                        type=float,
                        default=0.0
                        )
    parser.add_argument('--masking', 
                        type=float,
                        help='masking percent, default: None',
                        default=None)
    parser.add_argument('--iterations', 
                        type=int,
                        help='number of iterations',
                        default=1)
    parser.add_argument('--masking_ite', 
                        type=int,
                        help='masking iterations',
                        default=2)
    parser.add_argument('--interval', 
                        nargs='+', type=float,
                        help='interval of dataset',
                        default=[0.0,1.0])
    parser.add_argument('--output_txt_file',
                        help='Name of the output data file',
                        default='threshold0.txt'
                        )
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=1)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    args = parser.parse_args()
    
    main(args)
    
    # python predict_w_cem.py --prob_threshold 0.7 --output_txt_file threshold70.txt