#!/usr/bin/env python
# coding: utf-8


""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
from __future__ import absolute_import, division, print_function
import os
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--OS_ID", default=0, type=int, required=True,
                        help="")
parser.add_argument("--GAP", default=33000, type=int, required=True,
                        help="")
parser.add_argument("--NEG", action='store_true',
                        help="")
parser.add_argument("--para", action='store_true',
                        help="")
parser.add_argument("--ae_model_path", default=None, type=str, required=True,
                        help="")
parser.add_argument("--span", type=bool, default=False)


mata_args = parser.parse_args()
OS_ID = mata_args.OS_ID
GAP = mata_args.GAP

if mata_args.para:
    mata_args.NEG = False
else:
    mata_args.NEG = True
SPAN_LOSS = mata_args.span
NEG = mata_args.NEG
ae_model_path = mata_args.ae_model_path
print('args.OS_ID {} args.GAP {} args.NEG {} args.ae_model_path {} span_loss {}'.format(OS_ID, GAP, NEG, ae_model_path, SPAN_LOSS))
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % OS_ID

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor, SquadResult
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate

import argparse
import logging

import random
import glob
import timeit
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  RobertaForQuestionAnsweringClassify, RobertaTokenizer, RobertaConfig,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer,
                                  DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer,
                                  AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering, XLMTokenizer,
                                  )

from transformers import AdamW, get_linear_schedule_with_warmup, squad_convert_examples_to_features_cg




logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())                   for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForQuestionAnsweringClassify, RobertaTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_question_ids(input_ids, pad_id=1, max_query_length=64):
    input_ids = input_ids.cpu().data.numpy()
    question_masks = []
    question_ids = []
    question_targets = []
    for input_id in input_ids:
        input_id = input_id.tolist()
        idx = input_id.index(2)

        question_mask = [1,] * (idx) + [0,] * (max_query_length - idx)
        question_id = input_id[:idx] + [pad_id,] * (max_query_length - idx)
        question_target = input_id[1:idx + 1] + [pad_id,] * (max_query_length - idx)
        question_ids.append(question_id)
        question_masks.append(question_mask)
        question_targets.append(question_target)
    
    question_ids = torch.tensor([question_id for question_id in question_ids], dtype=torch.long).cuda()
    question_masks = torch.tensor([question_mask for question_mask in question_masks], dtype=torch.long).cuda()
    question_targets = torch.tensor([question_target for question_target in question_targets], dtype=torch.long).cuda()
        
    tgt_mask = (question_targets != pad_id).unsqueeze(-2)
    question_targets_masks = tgt_mask & Variable(
        subsequent_mask(question_targets.size(-1)).type_as(tgt_mask.data))
   
    return question_ids, question_masks, question_targets, question_targets_masks




def postprocess_doc(rd):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(rd))

def postprocess_raw(rq):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(rq[1: rq.index(1)-1])) + ' ?'

def postprocess_gen(gq):
    if 2 not in gq:
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(gq))
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(gq[: gq.index(2)-1])) +' ?'

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

 
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    for _ in train_iterator:
   
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            #for kk in range(10000):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            questions, question_masks, question_targets, question_targets_masks = get_question_ids(batch[0])
            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilbert'] else batch[2],
                'start_positions': batch[3],
                'end_positions':   batch[4],
                'labels': batch[7],
                'questions':questions,
                'question_masks':question_masks,
                'question_targets':question_targets,
                'question_targets_masks': question_targets_masks
            }


            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})


            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            prob = outputs[2]
            latent = outputs[1]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            

            tr_loss += loss.item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        _ = evaluate(args, model, tokenizer)
                        """
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        """
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    print('global_step', global_step)
                    print('loss', (tr_loss - logging_loss)/args.logging_steps)
                    logging_loss = tr_loss
                    print('\n')
                    
                    if args.n_gpu > 1:
                        generator_text = model.module.ae_model.greedy_decode(latent,
                                                    max_len=30,
                                                    start_id=0)
                    else:
                        generator_text = model.ae_model.greedy_decode(latent,
                                                    max_len=30,
                                                    start_id=0)
                    q = questions.cpu().detach().numpy()[0]

                    
                    print('questions[0]', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(q)))
                    g = generator_text.cpu().detach().numpy()[0]

                    print('generator_text[0]', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(g)))
                    
                    print('\n')
                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                   
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    

    return global_step, tr_loss / global_step



def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    raw_questions = []
    gen_questions = []
    

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        batch = tuple(t.to(args.device) for t in batch)
        questions, question_masks, question_targets, question_targets_masks = get_question_ids(batch[0])
        inputs = {
            'input_ids':       batch[0],
            'attention_mask':  batch[1],
            'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilbert'] else batch[2],
            'questions':questions,
            'question_masks':question_masks,
            'question_targets':question_targets,
            'question_targets_masks': question_targets_masks
        }


        if args.model_type in ['xlnet', 'xlm']:
            inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})


        outputs = model(**inputs)
        
        prob = outputs[2]
        latent = outputs[1]
        
        
           
        if args.n_gpu > 1:
            generator_text = model.module.ae_model.greedy_decode(latent,
                                        max_len=30,
                                        start_id=0)
        else:
            generator_text = model.ae_model.greedy_decode(latent,
                                        max_len=30,
                                        start_id=0)
        q = questions.cpu().detach().numpy().tolist()
        g = generator_text.cpu().detach().numpy().tolist()
        
        raw_questions += q
        gen_questions += g
        
    
   
    rouge_1_list = []
    rouge_2_list = []
    rouge_L_list = []
    bleu_list = []
    for i in range(len(raw_questions)):
        rq = raw_questions[i]
        gq = gen_questions[i]
        ground_truth = postprocess_raw(rq).lower()
        gen = postprocess_gen(gq).lower()


        rouge_1 = rouge.rouge_n(summary=gen, references=ground_truth, n=1)
        rouge_2 = rouge.rouge_n(summary=gen, references=ground_truth, n=2)
        rouge_l = rouge.rouge_l(summary=gen,references=ground_truth)
        bleu_score = bleu.bleu(summary=gen, references=ground_truth)
        
        rouge_1_list.append(rouge_1)
        rouge_2_list.append(rouge_2)
        rouge_L_list.append(rouge_l)
        bleu_list.append(bleu_score)
        
    print(len(bleu_list))
    print("ROUGE-1: {:.3f}, ROUGE-2: {:.3f}, ROUGE-L: {:.3f}, BLEU: {:.3f}".format(
        np.mean(rouge_1_list) * 100., np.mean(rouge_2_list) * 100., np.mean(rouge_L_list) * 100., np.mean(bleu_list)
    ).replace(", ", "\n"))
    return raw_questions, gen_questions

def test_mrc(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilbert'] else batch[2],
            }
            example_indices = batch[3]
            
            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})

            outputs = model.mrc_forward(**inputs, output_embedding=True)
            start_logits, end_logits, classification_logits, emb = outputs
            
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            
            output = [to_list(output[i]) for output in outputs]

            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id, start_logits, end_logits, 
                    start_top_index=start_top_index, 
                    end_top_index=end_top_index, 
                    cls_logits=cls_logits
                )

            else:
                start_logits, end_logits, _ = output
                result = SquadResult(
                    unique_id, start_logits, end_logits
                )

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ['xlnet', 'xlm']:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file,
                        start_n_top, end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        predictions = compute_predictions_logits(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold, tokenizer)

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length))
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features_cg( 
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset='pt',
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if output_examples:
        return dataset, examples, features
    return dataset





from config_args import *
args = Args()
args.model_name_or_path = '/data/squad/mrc_models'
args.eval_batch_size = 10




from dag_model import *

config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
print('self.model_name_or_path', args.model_name_or_path)
config = config_class.from_pretrained('roberta-large',
                                      cache_dir=args.cache_dir if args.cache_dir else None)


tokenizer = tokenizer_class.from_pretrained('roberta-large',
                                            do_lower_case=args.do_lower_case,
                                            cache_dir=args.cache_dir if args.cache_dir else None)


model = DAG.from_pretrained('roberta-large',
                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                    config=config,
                                    cache_dir=None,
                                   N=6,
                                   d_ff=4096)



args.device = torch.device("cuda")
path = mata_args.ae_model_path 
model_prefix = path.split('/')[-3] + '_' + path.split('/')[-2]

print('load path:', path)
model.load_state_dict(torch.load(path))

model.to(args.device)

output_dir = 'gen_res/' + model_prefix

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for p in model.roberta.embeddings.parameters():
    p.requires_grad= True


# Note that DistributedSampler samples randomly

features_and_dataset = torch.load('/data/cached_train_roberta-large_384')

if OS_ID == 3:
    dataset = torch.utils.data.Subset(features_and_dataset["dataset"], range(OS_ID * GAP, len(features_and_dataset["dataset"])))
else:
    dataset = torch.utils.data.Subset(features_and_dataset["dataset"], range(OS_ID * GAP, GAP * (OS_ID + 1)))

eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


print(len(dataset))






import pickle
def Jaccrad(s1, s2):
    terms_reference = [t.lower() for t in s2.split()]
    terms_model = [t.lower() for t in s1.split()]
    
    grams_reference = set(terms_reference)
    grams_model = set(terms_model)
    temp=0
    for i in grams_reference:
        if i in grams_model:
            temp=temp+1
    fenmu=len(grams_model)+len(grams_reference)-temp 
    jaccard_coefficient=float(temp/fenmu)
    return jaccard_coefficient



num_labels = 2
args.n_gpu = 1
c_loss_fct = CrossEntropyLoss()
epsilon_list = [2.0, 4.0, 6.0, 8.0]
if NEG:
    scaled = 500.
else:
    scaled = 10000.
max_query_length = 64
max_length = 384
JACARD_UPPER = 1.0
JACARD_LOWER = 0.3


gen_results = dict()

print('scaled', scaled)


uniq_id = GAP * OS_ID
iterate = 0
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    iterate += 1
    print(uniq_id)
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)    
    if SPAN_LOSS:
        inputs = {
        'input_ids':      batch[0],
        'attention_mask': batch[1],
        'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilbert'] else batch[2],
        'labels': batch[7],
        'start_positions': batch[3],
        'end_positions':   batch[4]
        }
    else:
        inputs = {
        'input_ids':      batch[0],
        'attention_mask': batch[1],
        'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilbert'] else batch[2],
        'labels': batch[7]
        }
    questions, question_masks, question_targets, question_targets_masks = get_question_ids(inputs['input_ids'])

    
    
    label_np = inputs['labels'].cpu().detach().numpy()
    if NEG:
        targets = 1 - inputs['labels']


    else:
        targets = inputs['labels']

    targets_np = targets.cpu().detach().numpy()
    for epsilon in epsilon_list:
        if args.n_gpu > 1:
            embedding_dag = model.module.roberta.embeddings(input_ids=questions)
            embedding_mrc_origin = model.module.roberta.embeddings(input_ids=inputs['input_ids'])
        else:
            embedding_dag = model.roberta.embeddings(input_ids=questions)
            embedding_mrc_origin = model.roberta.embeddings(input_ids=inputs['input_ids'])
        embedding_mrc = embedding_mrc_origin.clone()
        original_epsilon = epsilon

        for _ in range(5):
            embedding_mrc = to_var(embedding_mrc.clone())
            embedding_mrc.requires_grad = True
            if SPAN_LOSS:

                if args.n_gpu>1:
                    total_loss, start_logits, end_logits, classification_logits = model.module.mrc_forward(inputs_embeds=embedding_mrc, output_embedding=False, start_positions=inputs['start_positions'], end_positions=inputs['end_positions'])
                else:
                    total_loss, start_logits, end_logits, classification_logits = model.mrc_forward(inputs_embeds=embedding_mrc, output_embedding=False, start_positions=inputs['start_positions'], end_positions=inputs['end_positions'])
                print('total_loss', total_loss)
                model.zero_grad()
                total_loss.backward()
            else:
                if args.n_gpu>1:
                    start_logits, end_logits, classification_logits = model.module.mrc_forward(inputs_embeds=embedding_mrc, output_embedding=False)
                else:
                    start_logits, end_logits, classification_logits = model.mrc_forward(inputs_embeds=embedding_mrc, output_embedding=False)
                c_loss = c_loss_fct(classification_logits.view(-1, num_labels), targets.view(-1))
                model.zero_grad()
                c_loss.backward()
            embedding_mrc_grad = embedding_mrc.grad.data

            scores = classification_logits.cpu().detach().numpy()
            question_masks_expand = question_masks.unsqueeze(2).to(embedding_mrc_grad)
            document_mask_padding = torch.zeros(question_masks.shape[0], embedding_mrc.shape[1] - question_masks.shape[1]).to(question_masks)
            mrc_masks = torch.cat((question_masks, document_mask_padding), 1)
            mrc_masks_expand = mrc_masks.unsqueeze(2).to(embedding_mrc_grad)

            embedding_dag_grad = embedding_mrc_grad[:, :max_query_length, :] * question_masks_expand
            embedding_dag = embedding_dag - epsilon * embedding_dag_grad * scaled
            embedding_mrc = embedding_mrc - epsilon * embedding_mrc_grad * mrc_masks_expand * scaled

            if args.n_gpu > 1:
                latent, prob = model.module.ae_model(src_emb=embedding_dag, 
                                             src=questions, 
                                             tgt=questions, 
                                             src_mask=question_masks.unsqueeze(-2), 
                                             tgt_mask=question_targets_masks)
            else:
                latent, prob = model.ae_model(src_emb=embedding_dag, 
                                             src=questions, 
                                             tgt=questions, 
                                             src_mask=question_masks.unsqueeze(-2), 
                                             tgt_mask=question_targets_masks)


            if args.n_gpu > 1:
                generator_text = model.module.ae_model.greedy_decode(latent,
                                            max_len=30,
                                            start_id=0)
            else:
                generator_text = model.ae_model.greedy_decode(latent,
                                                max_len=30,
                                                    start_id=0)
            d = inputs['input_ids'].cpu().detach().numpy().tolist()
            q = questions.cpu().detach().numpy().tolist()

            g = generator_text.cpu().detach().numpy().tolist()
            for i in range(len(g)):
                if uniq_id + i not in gen_results:
                    gen_results[uniq_id + i] = dict()
                    gen_results[uniq_id + i]['question'] = postprocess_raw(q[i]) 
                    gen_results[uniq_id + i]['gen'] = []
                    gen_results[uniq_id + i]['label'] = label_np[i]
                else:
                    question = gen_results[uniq_id + i]['question']
                    gen_sample = postprocess_gen(g[i])
                    jacard_dist = Jaccrad(question, gen_sample)
                    if jacard_dist < JACARD_UPPER and jacard_dist > JACARD_LOWER:
                        score = scores[i][targets_np[i]].tolist()
                        if len(gen_results[uniq_id + i]['gen']) > 0 and gen_sample != gen_results[uniq_id + i]['gen'][-1][0]:
                            gen_results[uniq_id + i]['gen'].append((gen_sample, jacard_dist, original_epsilon, _, score))
                        elif len(gen_results[uniq_id + i]['gen']) == 0:
                            gen_results[uniq_id + i]['gen'].append((gen_sample, jacard_dist, original_epsilon, _, score))


            epsilon = epsilon * 0.9

    uniq_id += args.eval_batch_size
    if (iterate % 100) == 0:
        print('uniq_id', uniq_id)
        try:
            print(question, gen_sample)
        except UnicodeEncodeError:
            print('UnicodeEncodeError')

    if NEG:
        if SPAN_LOSS:
            pickle.dump(gen_results, open(os.path.join(output_dir, 'gen_results_%d_span.pkl' % OS_ID),'wb'))
        else:
            pickle.dump(gen_results, open(os.path.join(output_dir, 'gen_results_%d.pkl' % OS_ID),'wb'))
    else:
        if SPAN_LOSS:
            pickle.dump(gen_results, open(os.path.join(output_dir, 'para_gen_results_%d_span.pkl' % OS_ID),'wb'))
        else:
            pickle.dump(gen_results, open(os.path.join(output_dir, 'para_gen_results_%d.pkl' % OS_ID),'wb'))



if NEG:
    if SPAN_LOSS:
        pickle.dump(gen_results, open(os.path.join(output_dir, 'gen_results_%d_span.pkl' % OS_ID),'wb'))
    else:
        pickle.dump(gen_results, open(os.path.join(output_dir, 'gen_results_%d.pkl' % OS_ID),'wb'))
else:
    if SPAN_LOSS:
        pickle.dump(gen_results, open(os.path.join(output_dir, 'para_gen_results_%d_span.pkl' % OS_ID),'wb'))
    else:
        pickle.dump(gen_results, open(os.path.join(output_dir, 'para_gen_results_%d.pkl' % OS_ID),'wb'))






