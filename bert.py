# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Data generators for PTB data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import collections
import os
import sys
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


EOS = text_encoder.EOS



@registry.register_problem
class Bert(text_problems.Text2ClassProblem):

    # for next sentence predict
    @property
    def num_classes(self):
        return 2

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    @property
    def vocab_filename(self):
        return "vocab"

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        return self.generate_samples(data_dir, tmp_dir, dataset_split)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        vocab_path = "C:\\Users\\gt\\PycharmProjects\\bert-tf\\data\\vocab"

        word2id = {}
        word2id["<PAD>"] = 0
        word2id["<SEP>"] = 1
        f = open(vocab_path, encoding="utf-8", mode="r")
        lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.replace("\n", "")
            word2id[word] = i

        data_file = "C:\\Users\\gt\\PycharmProjects\\bert-tf\\data\\next_sentence"

        f = open(data_file, encoding="utf-8", mode="r")

        lines = f.readlines()

        all_sentence = []
        for i, line in enumerate(lines):
            two_sentence = line.replace("\n", "")
            sentences = two_sentence.split("\t")
            all_sentence.extend(sentences)

        for i,line in enumerate(lines):
            two_sentence = line.replace("\n","")
            sentences = two_sentence.split("\t")
            prob = random.random()
            if prob < 0.5:
                one_sentence = sentences[0] + " <SEP> " + sentences[1]
                one_sentence_ = []
                for word in one_sentence:
                    if word in word2id:
                        one_sentence_.append(word2id[word.lower()])
                one_sentence_ = one_sentence_ + [0]*(100-len(one_sentence_))
                label = [1,0]
                yield {
                    "inputs":one_sentence_,
                    "targets":label
                }
            else:
                result = all_sentence[random.randrange(len(all_sentence))]
                if result==sentences[1]:
                    result = all_sentence[random.randrange(len(all_sentence))]
                one_sentence = sentences[0] + " <SEP> " + result
                one_sentence_ = []
                for word in one_sentence:
                    if word in word2id:
                        one_sentence_.append(word2id[word.lower()])
                one_sentence_ = one_sentence_ + [0] * (100 - len(one_sentence_))
                label = [0,1]
                yield {
                    "inputs":one_sentence_,
                    "targets":label
                }


