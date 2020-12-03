#!/usr/bin/env python
# coding: utf-8

# In[2]:


# coding=utf-8

from __future__ import absolute_import, division, print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor, WikiProcessor, SquadResult
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate
from dag_model import *
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

from transformers import AdamW, get_linear_schedule_with_warmup, squad_convert_examples_to_features_cg, squad_convert_examples_to_features_wiki
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator

bleu = BLEUCalculator()
rouge = RougeCalculator(stopwords=False, lang="en")

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)), ())

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
        #if idx > max_query_length:
        question_mask = [1,] * (idx) + [0,] * (max_query_length - idx)
        question_id = input_id[:idx] + [pad_id,] * (max_query_length - idx)
        question_target = input_id[1:idx + 1] + [pad_id,] * (max_query_length - idx)
        question_ids.append(question_id)
        question_masks.append(question_mask)
        question_targets.append(question_target)
    
    #print(np.array(question_ids).shape)
    #for question_id in question_ids:
    #    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(question_id)))
    #    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(question_target)))
    #    #print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(question_id)))
    #    break
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
    #print(tokenizer.convert_ids_to_tokens(rq[1: rq.index(1)]))
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(rq[1: rq.index(1)-1])) + ' ?'

def postprocess_gen(gq):
    if 2 not in gq:
        #print('!!!!!', gq, tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(gq)) +' ?')
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(gq))
    #print(tokenizer.convert_ids_to_tokens(gq[: gq.index(2)]))
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
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
        logger.info('torch.nn.DataParallel!!!!!!!!')
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        
        logger.info('torch.nn.parallel.DistributedDataParallel!!!!!!!!')
        logger.info('n_gpu {}'.format(args.n_gpu))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
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
    #while True:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            #for kk in range(10000):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            questions, question_masks, question_targets, question_targets_masks = get_question_ids(batch[0])

            '''
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
            '''
            inputs = {
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
            #print('latent', latent.shape)
            #emb = outputs[-1]
            #print('emb', emb.shape, emb)
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            #return inputs, emb

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
                #args.logging_steps = 1 #！！！
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #print('logging')
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        _ = evaluate(args, model, tokenizer)
                        """
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        """
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logger.info('global_step: {} loss: {}'.format(global_step, (tr_loss - logging_loss)/args.logging_steps))
                   
                    logging_loss = tr_loss
                    #print('\n')
                    
                    if hasattr(model, 'ae_model'):
                        generator_text = model.ae_model.greedy_decode(latent,
                                                    max_len=30,
                                                    start_id=0)
                    else:
                        generator_text = model.module.ae_model.greedy_decode(latent,
                                                    max_len=30,
                                                    start_id=0)
                    q = questions.cpu().detach().numpy()[0]
                    #print('kk \n', kk)
                    #print('loss', loss)
                    
                    logger.info('questions[0] {}'.format(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(q))))
                    g = generator_text.cpu().detach().numpy()[0]

                    #print('prob[0]', prob[0])
                    #_, next_word = torch.max(prob, dim=-1)
                    #nw = next_word.cpu().detach().numpy()[0]
                    #print(next_word.shape, nw.shape, nw)
                    #print('next_word[0]', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(nw)))
                    logger.info('generator_text[0] {}'.format(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(g))))
                    #print('generator_text[0]', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(g)))
                    logger.info('\n')
                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
            #break #!!        
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        """
        print('start eval')
        print('args.local_rank', args.local_rank)
        if args.local_rank == -1:
            print('args.local_rank!', args.local_rank)
            _ = evaluate(args, model, tokenizer)
        """
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

    # Eval!
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
        #loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        prob = outputs[2]
        latent = outputs[1]
        
        
           
        if hasattr(model, 'ae_model'):
            generator_text = model.ae_model.greedy_decode(latent,
                                            max_len=30,
                                            start_id=0)
        else:
            generator_text = model.module.ae_model.greedy_decode(latent,
                                        max_len=30,
                                        start_id=0)
        q = questions.cpu().detach().numpy().tolist()
        #print('questions[0]', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(q[0])))
        g = generator_text.cpu().detach().numpy().tolist()
        #print('generator_text[0]', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(g[0])))
        raw_questions += q
        gen_questions += g
        #break
    
   
    rouge_1_list = []
    rouge_2_list = []
    rouge_L_list = []
    bleu_list = []
    for i in range(len(raw_questions)):
        rq = raw_questions[i]
        gq = gen_questions[i]
        ground_truth = postprocess_raw(rq).lower()
        gen = postprocess_gen(gq).lower()

        #ground_truth = data['golden'].lower()
        #gen = data['generated'].lower()
        #print(ground_truth)
        #print(gen)
        rouge_1 = rouge.rouge_n(summary=gen, references=ground_truth, n=1)
        rouge_2 = rouge.rouge_n(summary=gen, references=ground_truth, n=2)
        rouge_l = rouge.rouge_l(summary=gen,references=ground_truth)
        bleu_score = bleu.bleu(summary=gen, references=ground_truth)
        #print(rouge_1, rouge_2, rouge_l, bleu_score)
        rouge_1_list.append(rouge_1)
        rouge_2_list.append(rouge_2)
        rouge_L_list.append(rouge_l)
        bleu_list.append(bleu_score)
        #break
    print(len(bleu_list))
    print("ROUGE-1: {:.3f}, ROUGE-2: {:.3f}, ROUGE-L: {:.3f}, BLEU: {:.3f}".format(
        np.mean(rouge_1_list) * 100., np.mean(rouge_2_list) * 100., np.mean(rouge_L_list) * 100., np.mean(bleu_list)
    ).replace(", ", "\n"))
    return raw_questions, gen_questions



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
            #processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            processor = WikiProcessor() if args.version_2_with_negative else SquadV1Processor()

            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features_wiki( 
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


# In[3]:




def main():
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
            
    
    log_path = os.path.join(args.output_dir, 'log.txt')
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)    
    #log_path = 'depth%s_rand_%s_log.txt' % (args.depth, args.rand)
    print('log_path {}'.format(log_path))
    handler = logging.FileHandler(log_path, 'a', 'utf-8')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('log_path {}'.format(log_path))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()


    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    logger.info('mrc_model:{}'.format(args.model_name_or_path))
    config = config_class.from_pretrained(args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained('roberta-large',
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = DAG.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=None)
    
    for p in model.roberta.parameters():
        p.requires_grad = False
        
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    #logger.info("Training/evaluation parameters %s", args)

    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

    _ = train(args, train_dataset, model, tokenizer)



if __name__ == "__main__":
    main()


