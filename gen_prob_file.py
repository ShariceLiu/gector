import argparse

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel

from utils.preprocess_data import convert_data_from_raw_files, my_get_prob_edits

def predict_sing_stc(stc, model):
    
    batch = [stc.split()]
    sequences = model.preprocess(batch)

    probabilities, idxs, error_probs = model.predict(sequences)

    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs, error_probs)

    result_line = [" ".join(x) for x in pred_batch]

    return result_line ,probabilities, idxs


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


    stc = 'Finally, about the free three hours, my personal prefer to go shopping .'

    target_path =  '/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep-v2/test.tgt'
    source_path = args.input_file
    output_path = args.output_file

    edits, probs, labels = my_get_prob_edits(target_file=target_path, source_file=source_path, model=model)

    # print(edits, probs, labels)

    lines = []
    for i in range(len(edits)):
        line = "{} {} {}\n".format(probs[i], labels[i], edits[i])
        lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines)

    # output, probabilities, idxs = predict_sing_stc(stc, model)
    # print(output, probabilities, idxs)

    # convert_data_from_raw_files(source_file=args.input_file,target_file=target_path,output_file="",chunk_size=50)

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