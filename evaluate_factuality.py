import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from predict_factuality import predict_factuality


def main(args):
    tokenizer = BertTokenizerFast.from_pretrained(args.model_string)
    print(f"Loading pre-trained model: {args.model_string}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_string, return_dict=True).to(args.device)
    if args.model_path is not None:
        if os.path.isdir(args.model_path):
            for _, _, files in os.walk(args.model_path):
                for fname in files:
                    if fname.endswith('.ckpt'):
                        args.model_path = os.path.join(args.model_path, fname)
                        break
        ckpt = torch.load(args.model_path)
        try:
            model.load_state_dict(ckpt['state_dict'])
        except:
            state_dict = {}
            for key in ckpt['state_dict'].keys():
                assert key.startswith('model.')
                state_dict[key[6:]] = ckpt['state_dict'][key]
            model.load_state_dict(state_dict)
    model.eval()

    print(f"Loading pre-trained conditioning model: {args.attribute_model_string}...")
    conditioning_model = AutoModelForSequenceClassification.from_pretrained(args.attribute_model_string).to(args.device)
    conditioning_model.eval()
    if args.verbose:
        #checkpoint = torch.load(args.ckpt, map_location=args.device)
        #print(f"=> loaded checkpoint '{args.ckpt}' (epoch {checkpoint['epoch']})")
        print(f"model num params {num_params(model)}")
        print(f"conditioning_model num params {num_params(conditioning_model)}")

    inputs = []
    with open(args.in_file, 'r', encoding='utf-8') as rf:
        for line in rf:
            inputs.append(line.strip())
    
    for inp in tqdm(inputs, total=len(inputs)):
        results = predict_factuality(model,
                        tokenizer, 
                        conditioning_model, 
                        [inp],
                        precondition_topk=args.precondition_topk,
                        do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
        print(results[0])
        import pdb;pdb.set_trace()

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='patrickvonplaten/bert2bert_cnn_daily_mail')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--attribute_model_string', type=str, default='')

    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample or greedy; only greedy implemented')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)