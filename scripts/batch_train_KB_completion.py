# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
import lama.utils as utils
from lama.utils import print_sentence_predictions, load_vocab
import lama.options as options
from tqdm import tqdm
from random import shuffle
import os
import json
import spacy
import lama.modules.base_connector as base
from pprint import pprint
import logging.config
import logging
import pickle
from multiprocessing.pool import ThreadPool
import multiprocessing
import lama.evaluation_metrics as metrics
import time, sys
import torch
import numpy as np

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]

def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        masked_sentences = sample["masked_sentences"]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg


def lowercase_samples(samples):
    new_samples = []
    for sample in samples:
        sample["obj_label"] = sample["obj_label"].lower()
        sample["sub_label"] = sample["sub_label"].lower()
        lower_masked_sentences = []
        for sentence in sample["masked_sentences"]:
            sentence = sentence.lower()
            sentence = sentence.replace(base.MASK.lower(), base.MASK)
            lower_masked_sentences.append(sentence)
        sample["masked_sentences"] = lower_masked_sentences
        new_samples.append(sample)
    return new_samples


def filter_samples(model, samples, vocab_subset, max_sentence_length, template):
    msg = ""
    new_samples = []
    samples_exluded = 0
    for sample in samples:
        excluded = False
        if "obj_label" in sample and "sub_label" in sample:

            obj_label_ids = model.get_id(sample["obj_label"])

            if obj_label_ids:
                recostructed_word = " ".join(
                    [model.vocab[x] for x in obj_label_ids]
                ).strip()
            else:
                recostructed_word = None

            excluded = False
            if not template or len(template) == 0:
                masked_sentences = sample["masked_sentences"]
                text = " ".join(masked_sentences)
                if len(text.split()) > max_sentence_length:
                    msg += "\tEXCLUDED for exeeding max sentence length: {}\n".format(
                        masked_sentences
                    )
                    samples_exluded += 1
                    excluded = True

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if vocab_subset:
                for x in sample["obj_label"].split(" "):
                    if x not in vocab_subset:
                        excluded = True
                        msg += "\tEXCLUDED object label {} not in vocab subset\n".format(
                            sample["obj_label"]
                        )
                        samples_exluded += 1
                        break

            if excluded:
                pass
            elif obj_label_ids is None:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            elif not recostructed_word or recostructed_word != sample["obj_label"]:
                msg += "\tEXCLUDED object label {} not in model vocabulary\n".format(
                    sample["obj_label"]
                )
                samples_exluded += 1
            # elif vocab_subset is not None and sample['obj_label'] not in vocab_subset:
            #   msg += "\tEXCLUDED object label {} not in vocab subset\n".format(sample['obj_label'])
            #   samples_exluded+=1
            elif "judgments" in sample:
                # only for Google-RE
                num_no = 0
                num_yes = 0
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no > num_yes:
                    # SKIP NEGATIVE EVIDENCE
                    pass
                else:
                    new_samples.append(sample)
            else:
                new_samples.append(sample)
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg


def main(args, model):

    if len(args.models_names) > 1:
        raise ValueError('Please specify a single language model (e.g., --lm "bert").')

    msg = ""

    [model_type_name] = args.models_names

    for name, param in model.masked_bert_model.named_parameters():
        if 'word_embeddings' not in name:
            param.requires_grad = False
        else:
            print(name)
            sys.stdout.flush()
    model_name = "BERT_{}".format(args.bert_model_name)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.masked_bert_model.parameters(), lr=1e-3)

    # deal with vocab subset
    vocab_subset = None
    index_list = None

    data = load_file(args.trainset_filename)

    print(len(data))
    all_samples, ret_msg = filter_samples(
        model, data, vocab_subset, args.max_sentence_length, args.template
    )
    print(len(all_samples))

    # if template is active (1) use a single example for (sub,obj) and (2) ...
    if args.template and len(args.template) != 0 and args.template[0] != "":
        facts = []
        for sample in all_samples:
            sub = sample["sub_label"]
            obj = sample["obj_label"]
            if (sub, obj) not in facts:
                facts.append((sub, obj))
        all_samples = []
        for fact in facts:
            for template_ in args.template:
                (sub, obj) = fact
                sample = {}
                sample["sub_label"] = sub
                sample["obj_label"] = obj
                # sobstitute all sentences with a standard template
                sample["masked_sentences"] = parse_template(
                    template_.strip(), sample["sub_label"].strip(), base.MASK
                )
                all_samples.append(sample)

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    for _ in range(200):
        shuffle(all_samples)
        samples_batches, sentences_batches, ret_msg = batchify(all_samples, args.batch_size)
        loss_epoch = 0
        #for i in tqdm(range(len(samples_batches))):
        for i in range(len(samples_batches)):
            optimizer.zero_grad()
            samples_b = samples_batches[i]
            sentences_b = sentences_batches[i]
            label_index = []
            for sample in samples_b:
                obj_label_id = model.get_id(sample["obj_label"])[0] # assume single word object
                label_index.append(obj_label_id)

            model.try_cuda()
            label_index = torch.tensor(np.array(label_index)).to(model._model_device)
            tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = model.get_input_tensors_batch(sentences_b)
            #print(tokenized_text_list)
            logits = model.masked_bert_model(
                input_ids=tokens_tensor.to(model._model_device),
                token_type_ids=segments_tensor.to(model._model_device),
                attention_mask=attention_mask_tensor.to(model._model_device),
            )[0]
            logits = logits[torch.arange(len(masked_indices_list)), masked_indices_list[0]]
            #print(logits.shape, label_index.shape)
            output = loss(logits, label_index)
            #print(output)
            output.backward()
            loss_epoch += output.detach().cpu()
            #output.grad.data[:-9,:].fill_(0)
            for name, param in model.masked_bert_model.named_parameters():
                if 'word_embeddings' in name:
                    param.grad.data[:-9,:].fill_(0.0)
            optimizer.step()
        print('loss: {}'.format(loss_epoch / len(samples_batches)))


    return model


if __name__ == "__main__":
    parser = options.get_eval_KB_completion_parser()
    args = options.parse_args(parser)
    main(args)
