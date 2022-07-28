import argparse

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel

from utils.preprocess_data import convert_data_from_raw_files


def predict_for_file(input_file, output_file, model, batch_size=32, to_normalize=False):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    with open(output_file, 'w') as f:
        f.write("\n".join(result_lines) + '\n')
    return cnt_corrections

def predict_sing_stc(stc, model):
    
    batch = [stc.split()]
    sequences = model.preprocess(batch)

    probabilities, idxs, error_probs = model.predict(sequences)

    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs, error_probs)

    return pred_batch


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
                         weigths=args.weights)

    # cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
    #                                    batch_size=args.batch_size, 
    #                                    to_normalize=args.normalize)

    # evaluate with m2 or ERRANT
    # print(f"Produced overall corrections: {cnt_corrections}")
    stc = 'This are grammatical sentence .'
    # preds = predict_sing_stc(stc, model)

    target_path =  '/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep-v2/test.tgt'
    pred_path = '/data/test_pred/test_pred.txt'

    convert_data_from_raw_files(source_file=args.input_file,target_file=target_path,output_file=pred_path,chunk_size=50)




if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default=['/home/alta/BLTSpeaking/exp-vr313/GEC/gector/trained_models/roberta_1_gectorv2.th']
                        )
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='/home/alta/BLTSpeaking/exp-zl437/demo/gector/data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        default='/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep-v2/test.src'
                        )
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default='data/test_pred/test_pred.txt')
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
                        default=5)
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
                        default=0.0)
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