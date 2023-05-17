#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List, NamedTuple

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.trainer_utils import is_main_process
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from my_trainer import MyTrainer
from my_model import myBertForSequenceClassification

from postprocess_attribution import *
from collections import Counter, defaultdict
from nltk.util import ngrams
import math
from tqdm import tqdm
import pandas as pd

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default='data/LCQMC/train_data.csv', metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default='data/LCQMC/dev_data.csv', metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default='data/test_set_a/test_data.csv', metadata={"help": "A csv or a json file containing the test data."}
    )
    original_test_file: Optional[str] = field(
        default='data/test_set_a/sim_interpretation_A.txt',
        metadata={"help": "A txt file containing the original test data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(default='./model',
                                    metadata={
                                        "help": "Path to pretrained model or model identifier from huggingface.co/models"}
                                    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                    "`DistributedDataParallel`. only set True when use mix_up."
        },
    )
    ddp_broadcast_buffers: Optional[bool] = field(
        default=True,
        metadata={
            "help": "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                    "`DistributedDataParallel`. only set True when use rdrop."
        },
    )
    result_record_csv: str = field(
        default="result_record.csv",
        metadata={"help": "Path to record result"}
    )
    train_dataset_name: str = field(
        default="lcqmc",
        metadata={"help": "one-fold or k-fold"}
    )
    train_valid_split: str = field(
        default="one-fold",
        metadata={"help": "one-fold or k-fold"}
    )
    # training_data_clean: bool = field(
    #     default=False,
    #     metadata={"help": "predict training file label each fold for k-fold clean training data"}
    # )
    bert_grouped_lr: bool = field(
        default=False,
        metadata={"help": "whether to use bert grouped lr"}
    )
    classifier_lr_alpha: float = field(
        default=1.0,
        metadata={"help": "whether to use classifier grouped lr"}
    )
    do_fgm: bool = field(
        default=False,
        metadata={"help": "fgm对抗训练"}
    )
    do_pgd: bool = field(
        default=False,
        metadata={"help": "pgd对抗训练"}
    )
    do_ema: bool = field(
        default=False,
        metadata={"help": "ema移动指数平均"}
    )
    do_rdrop: bool = field(
        default=False,
        metadata={"help": "r-drop操作"}
    )
    rdrop_alpha: float = field(
        default=1.0,
        metadata={"help": "r-drop系数，仅当do_rdrop为True时有效"}
    )
    classifier_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout rate"}
    )
    pool_type: str = field(
        default="cls",
        metadata={"help": "分类向量"}
    )
    multi_sample_dropout_num: int = field(
        default=1,
        metadata={"help": "multi sample dropout num"}
    )
    multi_sample_avg: bool = field(
        default=False,
        metadata={"help": "multi sample dropout avg"}
    )
    mix_up: bool = field(
        default=False,
        metadata={"help": "mix up data aug"}
    )
    mix_up_layer: str = field(
        default="pooler",
        metadata={"help": "embedding, pooler, last, inner"}
    )
    ig_step: int = field(
        default=1,
        metadata={"help": "number of steps in integrated gradients"}
    )
    classification_threshold: float = field(
        default=0.5,
        metadata={"help": "classification threshold"}
    )
    rational_ratio: float = field(
        default=0.705,
        metadata={"help": "important token ratio"}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            # device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from transformers.integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


class InterpretResult(NamedTuple):
    words: List[str]
    word_attributions: List[float]
    pred_label: float
    pred_proba: List[float]
    rationale: List[int]
    non_rationale: List[int]
    rationale_tokens: List[str]
    non_rationale_tokens: List[str]
    rationale_pred_proba: float = None
    non_rationale_pred_proba: float = None


class BLEUScorer(object):
    # BLEU score calculator via GentScorer interface
    # it calculates the BLEU-4 by taking the entire corpus in
    # Calculate based multiple candidates against multiple references
    # code from https://github.com/shawnwun/NNDIAL
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-10
        bp = 1 if c > r else math.exp(1 - float(r) / (float(c) + p0))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


def func1(input_file):
    output_file = input_file.replace(".txt", "_f1.txt")
    predicts = dict()
    fin = open(input_file, 'r')
    out_file = open(output_file, 'w')
    for i, line in enumerate(fin.readlines()):
        line = line.strip().split("\t")
        idx, label, sent1_tokens, sent2_tokens = line[0], line[1], line[2], line[3]
        sent1_tokens = [int(x) for x in sent1_tokens.split(',')]
        sent2_tokens = [int(x) for x in sent2_tokens.split(',')]
        sent1_tokens.sort()
        sent2_tokens.sort()
        predicts[i] = dict()
        predicts[i]["id"] = idx
        predicts[i]["pred_label"] = label
        predicts[i]["rationale_q"] = sent1_tokens
        predicts[i]["rationale_t"] = sent2_tokens
    for key in predicts:
        out_file.write(str(predicts[key]['id']) + '\t' + str(predicts[key]['pred_label']) + '\t')
        for idx in predicts[key]['rationale_q'][:-1]:
            out_file.write(str(idx) + ',')
        out_file.write(str(predicts[key]['rationale_q'][-1]) + '\t')

        for idx in predicts[key]['rationale_t'][:-1]:
            out_file.write(str(idx) + ',')
        out_file.write(str(predicts[key]['rationale_t'][-1]) + '\n')
    out_file.close()


def load_data(file_path):
    """load data"""
    data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip():
                example = json.loads(line)
                data[example[list(example.keys())[0]]] = example
    return data


def load_label(path):
    labels = {}
    with open(path, mode="r") as fp:
        for line in fp.readlines():
            split_line = line.strip().split()
            res = {}
            res['pred_label'] = split_line[1]
            res['rationale_q'] = split_line[2].split(',')
            res['rationale_t'] = split_line[3].split(',')
            labels[int(split_line[0])] = res
        fp.close()
    return labels


def same_pos_neg(data, pred_label, raw_pred_label):
    for key in data:
        # print(key)
        data[key]['pred_label'] = pred_label[key]['pred_label']
        if pred_label[key]['pred_label'] == '1':
            rationale_q = []
            rationale_t = []
            for i, tok in enumerate(data[key]['text_q_seg']):
                if tok in ["，", "？", "！", "。", "：", "、", "”", "（", "）", "【", "】", "{", "}"]:
                    continue
                if tok in data[key]['text_t_seg']:
                    rationale_q.append(i)
            for i, tok in enumerate(data[key]['text_t_seg']):
                if tok in ["，", "？", "！", "。", "：", "、", "”", "（", "）", "【", "】", "{", "}"]:
                    continue
                if tok in data[key]['text_q_seg']:
                    rationale_t.append(i)
            raw_rq_len = len(raw_pred_label[key]["rationale_q"])
            new_rationale_q = []
            for tok in rationale_q:
                if str(tok) in raw_pred_label[key]["rationale_q"] and len(new_rationale_q) < raw_rq_len:
                    new_rationale_q.append(tok)
            if len(new_rationale_q) < raw_rq_len:
                for tok in rationale_q:
                    if tok not in new_rationale_q and len(new_rationale_q) < raw_rq_len:
                        new_rationale_q.append(tok)
            if len(new_rationale_q) < raw_rq_len:
                for tok in raw_pred_label[key]["rationale_q"]:
                    if int(tok) not in new_rationale_q and len(new_rationale_q) < raw_rq_len:
                        new_rationale_q.append(int(tok))
            rationale_q = new_rationale_q
            raw_rt_len = len(raw_pred_label[key]["rationale_t"])
            new_rationale_t = []
            for tok in rationale_t:
                if str(tok) in raw_pred_label[key]["rationale_t"] and len(new_rationale_t) < raw_rt_len:
                    new_rationale_t.append(tok)
            if len(new_rationale_t) < raw_rt_len:
                for tok in rationale_t:
                    if tok not in new_rationale_t and len(new_rationale_t) < raw_rt_len:
                        new_rationale_t.append(tok)
            if len(new_rationale_t) < raw_rt_len:
                for tok in raw_pred_label[key]["rationale_t"]:
                    if int(tok) not in new_rationale_t and len(new_rationale_t) < raw_rt_len:
                        new_rationale_t.append(int(tok))
            rationale_t = new_rationale_t
            assert len(rationale_q) == raw_rq_len
            assert len(rationale_t) == raw_rt_len
        else:
            rationale_q = []
            rationale_t = []
            for i, tok in enumerate(data[key]['text_q_seg']):
                if tok in ["，", "？", "！", "。", "：", "、", "”", "（", "）", "【", "】", "{", "}"]:
                    continue
                if tok in data[key]['text_t_seg']:
                    rationale_q.append(i)
            for i, tok in enumerate(data[key]['text_t_seg']):
                if tok in ["，", "？", "！", "。", "：", "、", "”", "（", "）", "【", "】", "{", "}"]:
                    continue
                if tok in data[key]['text_q_seg']:
                    rationale_t.append(i)
            raw_rq_len = len(raw_pred_label[key]["rationale_q"])
            new_rationale_q = []
            for tok in rationale_q:
                if str(tok) in raw_pred_label[key]["rationale_q"] and len(new_rationale_q) < raw_rq_len:
                    new_rationale_q.append(tok)
            if len(new_rationale_q) < raw_rq_len:
                for tok in raw_pred_label[key]["rationale_q"]:
                    if int(tok) not in new_rationale_q and len(new_rationale_q) < raw_rq_len:
                        new_rationale_q.append(int(tok))
            if len(new_rationale_q) < raw_rq_len:
                for tok in rationale_q:
                    if tok not in new_rationale_q and len(new_rationale_q) < raw_rq_len:
                        new_rationale_q.append(tok)
            rationale_q = new_rationale_q
            raw_rt_len = len(raw_pred_label[key]["rationale_t"])
            new_rationale_t = []
            for tok in rationale_t:
                if str(tok) in raw_pred_label[key]["rationale_t"] and len(new_rationale_t) < raw_rt_len:
                    new_rationale_t.append(tok)
            if len(new_rationale_t) < raw_rt_len:
                for tok in raw_pred_label[key]["rationale_t"]:
                    if int(tok) not in new_rationale_t and len(new_rationale_t) < raw_rt_len:
                        new_rationale_t.append(int(tok))
            if len(new_rationale_t) < raw_rt_len:
                for tok in rationale_t:
                    if tok not in new_rationale_t and len(new_rationale_t) < raw_rt_len:
                        new_rationale_t.append(tok)
            rationale_t = new_rationale_t
            assert len(rationale_q) == raw_rq_len
            assert len(rationale_t) == raw_rt_len
        data[key]['rationale_q'] = rationale_q
        data[key]['rationale_t'] = rationale_t
    return data


def func2(raw_test_file, predict_file):
    data = load_data(raw_test_file)
    pred_label = load_label(predict_file)
    raw_pred_label = load_label(predict_file.replace("_f1.txt", ".txt"))
    predicts = same_pos_neg(data, pred_label, raw_pred_label)
    out_file = open(predict_file.replace(".txt", "_f2.txt"), 'w')
    for key in predicts:
        out_file.write(str(predicts[key]['id']) + '\t' + str(predicts[key]['pred_label']) + '\t')
        for idx in predicts[key]['rationale_q'][:-1]:
            out_file.write(str(idx) + ',')
        out_file.write(str(predicts[key]['rationale_q'][-1]) + '\t')

        for idx in predicts[key]['rationale_t'][:-1]:
            out_file.write(str(idx) + ',')
        out_file.write(str(predicts[key]['rationale_t'][-1]) + '\n')
    out_file.close()


def get_m_data(lcqmc_testset, competition_testset, prediction_txt):
    output_txt1 = prediction_txt.replace(".txt", "_o.txt")
    output_txt2 = prediction_txt.replace(".txt", "_p.txt")

    ori_testset_df = pd.read_csv(lcqmc_testset, header=None, sep="\t")
    rational_df = pd.read_csv(prediction_txt, sep="\\t", header=None)

    testset_list = []
    with open(competition_testset, 'r') as f:
        for line in f.readlines():
            if line.strip():
                example = json.loads(line)
                testset_list.append(example)

    original_sent1 = list(ori_testset_df[0])
    original_sent2 = list(ori_testset_df[1])
    rational_sent1 = list(rational_df[2])
    rational_sent2 = list(rational_df[3])
    prediction_label = list(rational_df[1])

    original_index = 0
    output_o = open(output_txt1, "w")
    output_p = open(output_txt2, "w")
    debug_id = []
    for example, sent1, sent2, pred in zip(testset_list, rational_sent1, rational_sent2, prediction_label):
        flag = False
        id = example['id']
        query = example['query']
        title = example['title']
        if query in original_sent1 and title in original_sent2:
            flag = True
        # example['from_testset'] = flag
        example['sent1_token'] = sent1
        example['sent2_token'] = sent2
        example['prediction'] = pred
        if flag == False:
            output_p.write(json.dumps(example, ensure_ascii=False) + '\n')
        else:
            output_o.write(json.dumps(example, ensure_ascii=False) + '\n')


def read_file(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if line.strip():
                example = json.loads(line)
                data.append(example)
    return data


def get_p_data_map(example_o, examples_p, threshold):
    sentence1 = " ".join(example_o["text_q_seg"] + example_o["text_t_seg"])
    blue_score = BLEUScorer()
    result = []
    for example_p in examples_p:
        sentence2 = " ".join(example_p["text_q_seg"] + example_p["text_t_seg"])
        wrap_generated, wrap_truth = [[sentence2]], [[sentence1]]
        bleu = blue_score.score(zip(wrap_generated, wrap_truth))
        if bleu >= threshold:
            result.append(example_p)
    result_idx = [int(x["id"]) for x in result]
    return result_idx


def get_o_data_map(example_p, examples_o, threshold):
    sentence1 = " ".join(example_p["text_q_seg"] + example_p["text_t_seg"])
    blue_score = BLEUScorer()
    result = []
    for example_o in examples_o:
        sentence2 = " ".join(example_o["text_q_seg"] + example_o["text_t_seg"])
        wrap_generated, wrap_truth = [[sentence2]], [[sentence1]]
        bleu = blue_score.score(zip(wrap_generated, wrap_truth))
        if bleu >= threshold:
            result.append((bleu, example_o))
    if len(result) > 0:
        result.sort(key=lambda x: x[0], reverse=True)
        max_result = result[0][1]
        return [int(max_result["id"])]
    else:
        return [99999]


def get_p_dara_result(o_data, p_data, o_p_idx_map, use_label=False, use_token=False, o_to_p=True):
    final_result = dict()
    if o_to_p is True:
        p_idx_to_data = {int(x["id"]): x for x in p_data}
        for i, o_example in enumerate(tqdm(o_data)):
            final_result[int(o_example["id"])] = {"id": int(o_example["id"]),
                                                  "prediction": int(o_example["prediction"]),
                                                  "sent1_token": [int(x) for x in
                                                                  o_example["sent1_token"].split(',')],
                                                  "sent2_token": [int(x) for x in
                                                                  o_example["sent2_token"].split(',')],
                                                  }
            p_examples = [p_idx_to_data[p_idx] for p_idx in o_p_idx_map[int(o_example["id"])]]
            o_sent1_words = [o_example["text_q_seg"][x] for x in
                             final_result[int(o_example["id"])]["sent1_token"]]
            o_sent2_words = [o_example["text_t_seg"][x] for x in
                             final_result[int(o_example["id"])]["sent2_token"]]
            for p_example in p_examples:
                new_p_sent1_token = []
                for word in o_sent1_words:
                    try:
                        idx = p_example["text_q_seg"].index(word)
                        new_p_sent1_token.append(idx)
                    except:
                        pass
                new_p_sent2_token = []
                for word in o_sent2_words:
                    try:
                        idx = p_example["text_t_seg"].index(word)
                        new_p_sent2_token.append(idx)
                    except:
                        pass
                if use_label:
                    p_prediction = int(o_example["prediction"])
                else:
                    p_prediction = int(p_example["prediction"])
                if use_token:
                    p_sent1_token = new_p_sent1_token
                    p_sent2_token = new_p_sent2_token
                else:
                    p_sent1_token = [int(x) for x in p_example["sent1_token"].split(',')]
                    p_sent2_token = [int(x) for x in p_example["sent2_token"].split(',')]
                final_result[int(p_example["id"])] = {"id": int(p_example["id"]),
                                                      "prediction": p_prediction,
                                                      "sent1_token": p_sent1_token,
                                                      "sent2_token": p_sent2_token
                                                      }
        for p_example in p_data:
            if int(p_example["id"]) not in final_result:
                final_result[int(p_example["id"])] = {"id": int(p_example["id"]),
                                                      "prediction": p_example["prediction"],
                                                      "sent1_token": [int(x) for x in
                                                                      p_example["sent1_token"].split(',')],
                                                      "sent2_token": [int(x) for x in
                                                                      p_example["sent2_token"].split(',')]
                                                      }
    else:
        o_idx_to_data = {int(x["id"]): x for x in o_data}
        for i, o_example in enumerate(tqdm(o_data)):
            final_result[int(o_example["id"])] = {"id": int(o_example["id"]),
                                                  "prediction": int(o_example["prediction"]),
                                                  "sent1_token": [int(x) for x in
                                                                  o_example["sent1_token"].split(',')],
                                                  "sent2_token": [int(x) for x in
                                                                  o_example["sent2_token"].split(',')],
                                                  }
            assert len(final_result[int(o_example["id"])]["sent1_token"]) > 0
            assert len(final_result[int(o_example["id"])]["sent2_token"]) > 0
        for i, p_example in enumerate(tqdm(p_data)):
            if int(p_example["id"]) in [76, 877, 1653, 2225, 3802, 3026]:
                p_prediction = int(p_example["prediction"])
                p_sent1_token = [int(x) for x in p_example["sent1_token"].split(',')]
                p_sent2_token = [int(x) for x in p_example["sent2_token"].split(',')]
                final_result[int(p_example["id"])] = {"id": int(p_example["id"]),
                                                      "prediction": p_prediction,
                                                      "sent1_token": p_sent1_token,
                                                      "sent2_token": p_sent2_token
                                                      }
                continue
            assert len(o_p_idx_map[int(p_example["id"])]) == 1
            o_idx = o_p_idx_map[int(p_example["id"])][0]
            o_example = o_idx_to_data[o_idx] if o_idx != 99999 else None
            if o_example is not None:
                o_sent1_words = [o_example["text_q_seg"][x] for x in
                                 final_result[int(o_example["id"])]["sent1_token"]]
                o_sent2_words = [o_example["text_t_seg"][x] for x in
                                 final_result[int(o_example["id"])]["sent2_token"]]
                new_p_sent1_token = []
                for word in o_sent1_words:
                    try:
                        idx = p_example["text_q_seg"].index(word)
                        new_p_sent1_token.append(idx)
                    except:
                        pass
                new_p_sent2_token = []
                for word in o_sent2_words:
                    try:
                        idx = p_example["text_t_seg"].index(word)
                        new_p_sent2_token.append(idx)
                    except:
                        pass
                if use_label:
                    p_prediction = int(o_example["prediction"])
                else:
                    p_prediction = int(p_example["prediction"])
                if use_token:
                    p_sent1_token = new_p_sent1_token
                    p_sent2_token = new_p_sent2_token
                else:
                    p_sent1_token = [int(x) for x in p_example["sent1_token"].split(',')]
                    p_sent2_token = [int(x) for x in p_example["sent2_token"].split(',')]
                final_result[int(p_example["id"])] = {"id": int(p_example["id"]),
                                                      "prediction": p_prediction,
                                                      "sent1_token": p_sent1_token,
                                                      "sent2_token": p_sent2_token
                                                      }
            else:
                p_prediction = int(p_example["prediction"])
                p_sent1_token = [int(x) for x in p_example["sent1_token"].split(',')]
                p_sent2_token = [int(x) for x in p_example["sent2_token"].split(',')]
                final_result[int(p_example["id"])] = {"id": int(p_example["id"]),
                                                      "prediction": p_prediction,
                                                      "sent1_token": p_sent1_token,
                                                      "sent2_token": p_sent2_token
                                                      }
            assert len(final_result[int(p_example["id"])]["sent1_token"]) > 0
            assert len(final_result[int(p_example["id"])]["sent2_token"]) > 0
    tmp_results = sorted(final_result.items(), key=lambda d: d[0])
    new_final_result = dict()
    for tmp_result in tmp_results:
        k, v = tmp_result[0], tmp_result[1]
        new_final_result[k] = v
    return new_final_result


def func3(o_file, p_file, threshold=None, use_label=False, use_token=False, o_to_p=True):
    o_data = read_file(o_file)
    p_data = read_file(p_file)
    o_p_idx_map = dict()
    if o_to_p is True:
        for example_o in tqdm(o_data):
            result_idx = get_p_data_map(example_o=example_o, examples_p=p_data, threshold=threshold)
            o_p_idx_map[int(example_o["id"])] = result_idx
    else:
        for example_p in tqdm(p_data):
            result_idx = get_o_data_map(example_p=example_p, examples_o=o_data, threshold=threshold)
            o_p_idx_map[int(example_p["id"])] = result_idx
    predicts = get_p_dara_result(o_data, p_data, o_p_idx_map, use_label, use_token, o_to_p)
    output_file = o_file.replace("o.txt", f"f3.txt")
    out_file = open(output_file, 'w')
    for key in predicts:
        if key == 1:
            print(predicts[key])
        out_file.write(str(predicts[key]['id']) + '\t' + str(predicts[key]['prediction']) + '\t')
        for idx in predicts[key]['sent1_token'][:-1]:
            out_file.write(str(idx) + ',')
        out_file.write(str(predicts[key]['sent1_token'][-1]) + '\t')

        for idx in predicts[key]['sent2_token'][:-1]:
            out_file.write(str(idx) + ',')
        out_file.write(str(predicts[key]['sent2_token'][-1]) + '\n')
    out_file.close()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tricks = ""
    if training_args.fp16:
        tricks += "_fp16"
    if training_args.weight_decay != 0.0:
        tricks += "_wd{}".format(training_args.weight_decay)
    if training_args.warmup_ratio != 0.0:
        tricks += "_warmup{}".format(training_args.warmup_ratio)
    if training_args.lr_scheduler_type != "linear":
        tricks += "_{}-scheduler".format(str(training_args.lr_scheduler_type).replace("SchedulerType.", ""))
    if training_args.bert_grouped_lr:
        tricks += "_bert_grouped_lr"
    if training_args.classifier_lr_alpha > 1.0:
        tricks += "_classifier_grouped_lr_alpha{}".format(training_args.classifier_lr_alpha)
    if training_args.do_fgm:
        tricks += "_fgm"
    if training_args.do_pgd:
        tricks += "_pgd"
    if training_args.do_ema:
        tricks += "_ema"
    if training_args.do_rdrop:
        tricks += "_rdrop-alpha{}".format(training_args.rdrop_alpha)
    if training_args.classifier_dropout > 0.1:
        tricks += "_classifier_drop{}".format(training_args.classifier_dropout)
    if training_args.multi_sample_dropout_num > 1:
        avg_or_sum = "avg" if training_args.multi_sample_avg is True else "sum"
        tricks += "_multi_sample_drop{}_{}".format(avg_or_sum, training_args.multi_sample_dropout_num)
    if training_args.pool_type:
        tricks += "_{}".format(training_args.pool_type)
    if training_args.mix_up is True:
        tricks += "_mixup_{}".format(training_args.mix_up_layer)

    training_args.output_dir += "{}_{}_{}_bs{}_accumulate{}_lr{}_epoch{}{}/".format(
        training_args.train_dataset_name, training_args.train_valid_split,
        "hfl_chinese-roberta-wwm-ext", training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps, training_args.learning_rate, training_args.num_train_epochs, tricks)
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if config.model_type == "bert" or config.model_type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = myBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            pool_type=training_args.pool_type,
            classifier_dropout=training_args.classifier_dropout,
            multi_sample_dropout_num=training_args.multi_sample_dropout_num,
            multi_sample_avg=training_args.multi_sample_avg,
            mix_up=training_args.mix_up,
            mix_up_layer=training_args.mix_up_layer
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if len(non_label_column_names) == 3:
            sentence1_key, sentence2_key, sentence3_key = non_label_column_names[:3]
        elif len(non_label_column_names) == 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
            sentence3_key = None
        else:
            sentence1_key, sentence2_key, sentence3_key = non_label_column_names[0], None, None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        if sentence3_key is None and sentence2_key is None:
            args = (examples[sentence1_key],)
        elif sentence3_key is None:
            args = (examples[sentence1_key], examples[sentence2_key])
        else:
            exps1 = [examples[sentence1_key][i] + tokenizer.sep_token + examples[sentence2_key][i] for i in
                     range(len(examples[sentence1_key]))]
            args = (exps1, examples[sentence3_key])
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        if sentence3_key is not None:
            sep = tokenizer.convert_tokens_to_ids('[SEP]')
            sep_idx = [result.input_ids[i].index(sep) for i in
                       range(np.array(result.input_ids).shape[0])]  # 找到第一个 [SEP]
            for i in range(len(sep_idx)):  # 反转之后的所有值
                reverse_token = result.token_type_ids[i][sep_idx[i] + 1:]
                tmp = np.array(reverse_token) == 0
                tmp = tmp.tolist()
                tmp = [int(tmp[i]) for i in range(len(tmp))]
                result.token_type_ids[i] = result.token_type_ids[i][:sep_idx[i] + 1] + tmp

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l in label_to_id else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if training_args.classification_threshold != 0.5:
            margin = training_args.classification_threshold - 0.5
            preds[:, 0] += margin
            preds[:, 1] -= margin
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.

    if data_args.pad_to_max_length:
        if training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = default_data_collator
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        bert_grouped_lr=training_args.bert_grouped_lr,
        classifier_lr_alpha=training_args.classifier_lr_alpha,
        do_fgm=training_args.do_fgm,
        do_pgd=training_args.do_pgd,
        do_ema=training_args.do_ema,
        do_rdrop=training_args.do_rdrop,
        rdrop_alpha=training_args.rdrop_alpha
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            print(checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)
            # record validation result
            if trainer.is_world_process_zero():
                if not os.path.exists(training_args.result_record_csv):
                    record_writer = csv.writer(open(training_args.result_record_csv, 'w'), delimiter=',')
                    csv_head = ["model", "dataset", "train valid split", "max len",
                                "per gpu batch size", "gradient accumulate steps", "learning rate", "epoch", "tricks",
                                "eval acc", "test acc"]
                    record_writer.writerow(csv_head)
                else:
                    record_writer = csv.writer(open(training_args.result_record_csv, 'a+'), delimiter=',')
                tricks = "tricks" + tricks
                valid_record = [model_args.model_name_or_path.replace("/", "_"), training_args.train_dataset_name,
                                training_args.train_valid_split, data_args.max_seq_length,
                                training_args.per_device_train_batch_size, training_args.gradient_accumulation_steps,
                                training_args.learning_rate, training_args.num_train_epochs, tricks,
                                metrics["eval_accuracy"], " "]
                record_writer.writerow(valid_record)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            output = trainer.predict(predict_dataset, metric_key_prefix="predict", total_step=training_args.ig_step)
            # predictions, attributions, all_ids = output.predictions, output.attributions, output.all_ids
            predictions, attributions = output.predictions, output.attributions
            if training_args.classification_threshold != 0.5:
                margin = training_args.classification_threshold - 0.5
                predictions[:, 0] += margin
                predictions[:, 1] -= margin
            pred_label = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            pred_label = pred_label.tolist()
            attributions = attributions.tolist()
            # contexts = [tokenizer.decode(ids) for ids in all_ids]
            # tokens = [tokenizer.decode(ids).split() for ids in all_ids]

            data = []
            special_tokens = [tokenizer.cls_token, tokenizer.sep_token]
            contexts = []
            standard_split = []
            with open(data_args.original_test_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        example = json.loads(line)
                        data.append(example)
                        contexts.append(tokenizer.cls_token + example['query'] + tokenizer.sep_token + example[
                            'title'] + tokenizer.sep_token)
                        standard_split.append(
                            [tokenizer.cls_token] + example['text_q_seg'] + [tokenizer.sep_token] + example[
                                'text_t_seg'] + [tokenizer.sep_token])

            ori_offset_maps = []
            standard_split_offset_maps = []
            for i in range(len(contexts)):
                ori_offset_map = tokenizer(contexts[i], return_offsets_mapping=True, add_special_tokens=False)[
                    'offset_mapping']
                ori_offset_maps.append(ori_offset_map)
                standard_split_offset_maps.append(get_word_offset(contexts[i], standard_split[i]))

            align_res = []
            for i in range(len(contexts)):
                words = standard_split[i]
                context = contexts[i]
                word_offset_map = standard_split_offset_maps[i]
                subword_offset_map = ori_offset_maps[i]
                attr = attributions[i]

                assert subword_offset_map[-1][1] == word_offset_map[-1][
                    1], "error offset_map, please check word_offset_maps and subword_offset_maps"

                # merge speical tokens for subword_offset_map
                subword_offset_map = merge_subword_special_idx(words, word_offset_map, subword_offset_map,
                                                               special_tokens)

                # get word attributions
                word_attributions = get_word_attributions(words, word_offset_map, subword_offset_map, attr)
                # get ratioanles and non-rationales
                ratioanle_result = get_rationales_and_non_ratioanles(words,
                                                                     word_attributions,
                                                                     special_tokens=special_tokens,
                                                                     rationale_num=5)
                interpret_result = InterpretResult(words=words,
                                                   word_attributions=word_attributions,
                                                   pred_label=pred_label[i],
                                                   pred_proba=None,
                                                   rationale=ratioanle_result['rationale_ids'],
                                                   non_rationale=ratioanle_result['non_rationale_ids'],
                                                   rationale_tokens=ratioanle_result['rationale_tokens'],
                                                   non_rationale_tokens=ratioanle_result['non_rationale_tokens'])
                align_res.append(interpret_result)

            if trainer.is_world_process_zero():
                # Generate results for evaluation
                predicts = prepare_eval_data(data, align_res, RATIONALE_RATIO=training_args.rational_ratio)
                if training_args.rational_ratio != 0.705:
                    out_path = 'submit/' + model_args.model_name_or_path.replace("../similarity_v0/result/", "") \
                               + f"_{training_args.ig_step}_ratio-{training_args.rational_ratio}" + '/'
                    if "test_set_b" in data_args.original_test_file:
                        out_path = out_path[:-1] + '_test_b' + '/'
                else:
                    out_path = 'submit/' + model_args.model_name_or_path.replace("../similarity_v0/result/", "") \
                               + f"_{training_args.ig_step}" + '/'
                    if "test_set_b" in data_args.original_test_file:
                        out_path = out_path[:-1] + '_test_b' + '/'
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                out_file = open(out_path + 'sim_rationale.txt', 'w')
                for key in predicts:
                    out_file.write(str(predicts[key]['id']) + '\t' + str(predicts[key]['pred_label']) + '\t')
                    for idx in predicts[key]['rationale_q'][:-1]:
                        out_file.write(str(idx) + ',')
                    out_file.write(str(predicts[key]['rationale_q'][-1]) + '\t')

                    for idx in predicts[key]['rationale_t'][:-1]:
                        out_file.write(str(idx) + ',')
                    out_file.write(str(predicts[key]['rationale_t'][-1]) + '\n')
                out_file.close()
                input_file = out_path + 'sim_rationale.txt'
                raw_testset = data_args.original_test_file
                lcqmc_testset = "data/LCQMC/test.tsv"
                func1(input_file)
                func2(raw_testset, input_file.replace(".txt", "_f1.txt"))
                get_m_data(lcqmc_testset, raw_testset, input_file.replace(".txt", "_f1_f2.txt"))
                func3(input_file.replace(".txt", "_f1_f2_o.txt"), input_file.replace(".txt", "_f1_f2_p.txt"),
                      threshold=0.0, use_label=True, use_token=True, o_to_p=False)
            '''
            output_predict_file = "submit/" + training_args.output_dir[7:-1] + ".txt"
            output_file = open(output_predict_file, 'w', encoding='utf-8')
            for toks, attrs, pred in zip(tokens, attributions, predictions):
                datapoint = {'label':int(pred), 'attributions':[]}
                for token, attr in zip(toks, attrs):
                    if token==tokenizer.pad_token:
                        break
                    datapoint['attributions'].append((token, float(attr)))
                output_file.write(json.dumps(datapoint, ensure_ascii=False)+'\n')
            '''

            # output_predict_file = "submit/" + training_args.output_dir[7:-1] + ".csv"
            # if trainer.is_world_process_zero():
            # writer = csv.writer(open(output_predict_file, 'w'), delimiter=',')
            # logger.info(f"***** Predict results {task} *****")
            # writer.writerow(['Label'])
            # for idx, item in enumerate(predictions):
            # item = label_list[item]
            # writer.writerow([item])

    # if training_args.training_data_clean is True:
    #     logger.info("*** Predict Training data ***")
    #
    #     predict_datasets = [train_dataset, eval_dataset]
    #     datasets_name = ["train", "valid"]
    #     datasets_file = [data_args.train_file, data_args.validation_file]
    #     for idx, predict_dataset in enumerate(predict_datasets):
    #         # Removing the `label` columns because it contains -1 and Trainer won't like that.
    #         predict_dataset = predict_dataset.remove_columns("label")
    #         predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    #         predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
    #         output_predict_file = training_args.output_dir + "{}_predict.csv".format(datasets_name[idx])
    #         reader = csv.reader(open(datasets_file[idx]), delimiter=',')
    #         next(reader)
    #         titles, abstracts, labels = [], [], []
    #         for line in reader:
    #             title, abstract, label = line[0], line[1], line[2]
    #             titles.append(title)
    #             abstracts.append(abstract)
    #             labels.append(label)
    #         if trainer.is_world_process_zero():
    #             writer = csv.writer(open(output_predict_file, 'w'), delimiter=',')
    #             logger.info(f"***** Predict Training results *****")
    #             writer.writerow(['Title', 'Abstract', 'Topic(Label)', 'Topic(Label)_predict'])
    #             for idx, item in enumerate(predictions):
    #                 pred_label = label_list[item]
    #                 writer.writerow([titles[idx], abstracts[idx], labels[idx], pred_label])

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    # if data_args.task_name is not None:
    #     kwargs["language"] = "en"
    #     kwargs["dataset_tags"] = "glue"
    #     kwargs["dataset_args"] = data_args.task_name
    #     kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
