import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from ae_modules import *

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


class DAG(RobertaForQuestionAnsweringClassify):
    def __init__(self, config, N=4, d_ff=1024):
        super(DAG, self).__init__(config)
        #print('self.roberta.embeddings', self.roberta.embeddings)
        self.ae_model = make_model(d_vocab=50265, N=N, d_model = 1024, d_ff=d_ff, h=16, dropout=0.1)
        self.ae_criterion = LabelSmoothing(size=50265, padding_idx=1, smoothing=0.1)
        #print('self.roberta.embeddings.requires_grad', self.roberta.embeddings.requires_grad)
        #self.roberta.embeddings.requires_grad = False
        #print('self.roberta.embeddings.requires_grad', self.roberta.embeddings.requires_grad)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None, labels=None, 
                questions=None, question_masks=None, question_targets=None, question_targets_masks=None, input_embds=None):
        """
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if position_ids is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if position_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        head_mask = head_mask.view(-1, head_mask.size(-1)) if position_ids is not None else None
        """
        if input_ids is not None:
            #print('questions', questions.shape)
            emb = self.roberta.embeddings(input_ids=questions, 
                                          token_type_ids=token_type_ids, 
                                          position_ids=position_ids, inputs_embeds=None)
        else:
            emb = input_embds

        #print('emb', emb.sum())
        latent, prob = self.ae_model(src_emb=emb, 
                                     src=questions, 
                                     tgt=questions, 
                                     src_mask=question_masks.unsqueeze(-2), 
                                     tgt_mask=question_targets_masks)
        
        loss_rec = self.ae_criterion(prob.contiguous().view(-1, prob.size(-1)),
                                            question_targets.contiguous().view(-1)) / (questions != 1).data.sum().data

       
        #print('loss_rec', loss_rec)
        outputs = (loss_rec), latent, prob
        return outputs
    
    
    
        
    def mrc_forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None, labels=None, 
                questions=None, question_masks=None, question_targets=None, question_targets_masks=None, output_embedding=False, inputs_embeds=None):
        if input_ids is None:
            outputs = self.roberta(inputs_embeds=inputs_embeds,
                                   fixed_embedding=inputs_embeds,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                output_embedding=output_embedding)
        else:
            outputs = self.roberta(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                output_embedding=output_embedding)
        if output_embedding:
            emb = outputs[-1]
        sequence_output = outputs[0]
        #labels = cls_index
        classification_logits = self.classifier(sequence_output)
        #print('labels======================================', labels)
        #print('num_labels=================================', self.num_labels)
        #outputs = (classification_logits,) + outputs[2:]
        
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                c_loss_fct = MSELoss()
                c_loss = c_loss_fct(classification_logits.view(-1), labels.view(-1))
            else:
                c_loss_fct = CrossEntropyLoss()
                c_loss = c_loss_fct(classification_logits.view(-1, self.num_labels), labels.view(-1))


        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits, classification_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if labels is not None:
                total_loss += 0.5 * c_loss

            outputs = (total_loss,) + outputs 
            
        if output_embedding:
            outputs = outputs + (emb,) # (loss), start_logits, end_logits, classification_logits, emb
        
        return outputs
            