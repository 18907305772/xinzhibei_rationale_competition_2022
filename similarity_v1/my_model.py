import types
import random
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import numpy as np

from transformers import BertPreTrainedModel, BertModel
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import SequenceClassifierOutput

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


def SoftCrossEntropy(inputs, target, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class myBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, pool_type='cls', classifier_dropout=0.1, multi_sample_dropout_num=1, multi_sample_avg=False, mix_up=False, mix_up_layer='pooler'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pool_type = pool_type

        self.bert = BertModel(config)
        self.multi_sample_dropout_num = multi_sample_dropout_num
        self.multi_sample_avg = multi_sample_avg
        if self.multi_sample_dropout_num == 1:
            self.dropout = nn.Dropout(classifier_dropout)
        else:
            self.dropout = nn.ModuleList([nn.Dropout(classifier_dropout) for _ in range(self.multi_sample_dropout_num)])
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mix_up = mix_up
        self.mix_up_layer = mix_up_layer
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.pool_type == "cls":
            pooled_output = outputs[1]
        elif self.pool_type == "avg":
            last_hidden = outputs[0]
            pooled_output = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError
        if self.multi_sample_dropout_num == 1:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = None
            for i, dropout_op in enumerate(self.dropout):
                if i == 0:
                    out = dropout_op(pooled_output)
                    logits = self.classifier(out)

                else:
                    temp_out = dropout_op(pooled_output)
                    temp_logits = self.classifier(temp_out)
                    logits += temp_logits

            if self.multi_sample_avg is True:
                logits = logits / self.multi_sample_dropout_num

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            if self.mix_up is True:
                index = torch.randperm(len(input_ids)).to(input_ids.device)
                lam = np.random.beta(1, 1)
                labels_mix = F.one_hot(labels, self.num_labels) * lam + F.one_hot(labels[index], self.num_labels) * (1 - lam)

                def single_forward_hook(module, inputs, outputs):
                    mix_input = outputs * lam + outputs[index] * (1 - lam)
                    return mix_input

                def multi_forward_hook(module, inputs, outputs):
                    mix_input = outputs[0] * lam + outputs[0][index] * (1 - lam)
                    return tuple([mix_input])

                if self.mix_up_layer == 'embedding':
                    hook = self.bert.embeddings.register_forward_hook(single_forward_hook)
                elif self.mix_up_layer == 'pooler':
                    hook = self.bert.pooler.register_forward_hook(single_forward_hook)
                elif self.mix_up_layer == 'last':
                    layer_num = -1
                    hook = self.bert.encoder.layer[layer_num].register_forward_hook(multi_forward_hook)
                elif self.mix_up_layer == 'inner':
                    # 随机选一层
                    layer_num = random.randint(1, self.config.num_hidden_layers) - 1
                    hook = self.bert.encoder.layer[layer_num].register_forward_hook(multi_forward_hook)

                outputs1 = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                if self.pool_type == "cls":
                    pooled_output1 = outputs1[1]
                else:
                    raise NotImplementedError
                if self.multi_sample_dropout_num == 1:
                    pooled_output1 = self.dropout(pooled_output1)
                else:
                    raise NotImplementedError
                logits1 = self.classifier(pooled_output1)
                hook.remove()
                loss1 = SoftCrossEntropy(logits1.view(-1, self.num_labels), labels_mix, reduction='average')
                loss = loss + loss1

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
