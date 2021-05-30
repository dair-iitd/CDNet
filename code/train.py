import copy
import numpy as np
from models.mlm import MLMModel
from models.aggnet import AggNet
import os
import torch
import pickle as pickle

import sys

from collections import Counter
from nltk.util import ngrams

from utils.config import args, DEVICE

import math

import logging
import gc

logging.getLogger().setLevel(logging.INFO)


class Trainer(object):
    def __init__(
        self, model, handler, ckpt_path, num_epochs, learning_rate, clip, args
    ):
        self.args = args
        self.handler = handler
        self.model = model
        self.ckpt_path = ckpt_path
        self.epochs = num_epochs
        self.learning_rate = learning_rate
        self.clip = clip
        self.params = set()

        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def save_model(self, model, ckpt_path, epoch):
        directory = (
            ckpt_path
            + "/AGG-"
            + self.args["name"]
            + "-DS-"
            + str(self.args["dataset"])
            + "-MOD-"
            + str([self.args["model"]])
            + "lr"
            + str(self.args["lr"])
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model, directory + "/agg_" + epoch + ".pt")

    def trainData(self, init_epoch=0):
        curEpoch = init_epoch
        step = 0
        epochLoss = []
        epochCopyLoss = []
        epochSentLoss = []
        epochSelLoss = []
        epochSimLoss = []
        logging.info("Training the model")
        best_f1 = 0.0

        mode = [1, 0, 0, 1]
        while curEpoch <= self.epochs:
            step = step + 1
            self.optimizer.zero_grad()

            batch, epoch_done = self.handler.get_batch(data="trn")
            feedDict = self.model.get_feed_dict(batch)

            if curEpoch < 10:
                # Optimise sentence and similarity loss
                mode = [1, 0, 0, 1]
            elif curEpoch < 15:
                # Optimise sentence and copy loss
                mode = [1, 1, 0, 0]
            else:
                # Optimise sentence, copy and distillation loss
                mode = [1, 1, 1, 0]

            torch.cuda.empty_cache()
            self.model.train()
            if self.args["model"] == 1:
                loss, copy_loss, sent_loss, sel_loss, similarity_loss, _, _ = self.model.forward(
                    feedDict["inp_utt"],
                    feedDict["copy_inp_pos"],
                    feedDict["inp_mask"],
                    feedDict["inp_len"],
                    feedDict["context_len"],
                    feedDict["kb"],
                    feedDict["copy_kb_pos"],
                    feedDict["kb_mask"],
                    feedDict["keys"],
                    feedDict["keys_mask"],
                    feedDict["db_empty"],
                    feedDict["out_utt"],
                    feedDict["copy_out_utt"],
                    feedDict["sketch_tags"],
                    feedDict["copy_sketch_tag_pos"],
                    feedDict["out_len"],
                    feedDict["max_out_utt_len"],
                    feedDict["sketch_mask"],
                    feedDict["sketch_loss_mask"],
                    True,
                    mode
                )
            else:
                (
                    loss,
                    copy_loss,
                    sent_loss,
                    sel_loss,
                    similarity_loss,
                    _,
                    _,
                ) = self.model.forward(
                    feedDict["did"],
                    feedDict["window_inp_utt"],
                    feedDict["window_inp_len"],
                    feedDict["window_context_len"],
                    feedDict["inp_utt"],
                    feedDict["copy_inp_pos"],
                    feedDict["inp_mask"],
                    feedDict["inp_len"],
                    feedDict["context_len"],
                    feedDict["kb"],
                    feedDict["copy_kb_pos"],
                    feedDict["kb_mask"],
                    feedDict["keys"],
                    feedDict["keys_mask"],
                    feedDict["db_empty"],
                    feedDict["out_utt"],
                    feedDict["copy_out_utt"],
                    feedDict["sketch_tags"],
                    feedDict["copy_sketch_tag_pos"],
                    feedDict["out_len"],
                    feedDict["max_out_utt_len"],
                    feedDict["sketch_mask"],
                    feedDict["sketch_loss_mask"],
                    feedDict["beta_filter"],
                    feedDict["type_kb_entities"],
                    True,
                    mode,
                    "DEBUG_STUFF"
                )
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            # Update parameters with optimizers
            self.optimizer.step()

            epochLoss.append(loss.detach().item())
            epochCopyLoss.append(copy_loss.detach().item())
            epochSentLoss.append(sent_loss.detach().item())
            epochSelLoss.append(sel_loss.detach().item())
            epochSimLoss.append(similarity_loss.detach().item())

            del loss
            del copy_loss
            del sent_loss
            del sel_loss
            del similarity_loss
            del feedDict
            gc.collect()
            torch.cuda.empty_cache()

            if step % 40 == 0:
                outstr = (
                    "Step: "
                    + str(step)
                    + " Loss: "
                    + str(np.mean(np.asarray(epochLoss)))
                    + " copy loss: "
                    + str(np.mean(np.asarray(epochCopyLoss)))
                    + " sent loss: "
                    + str(np.mean(np.asarray(epochSentLoss)))
                    + " sel loss: "
                    + str(np.mean(np.asarray(epochSelLoss)))
                    + " similarity loss: "
                    + str(np.mean(np.asarray(epochSimLoss)))
                )
                logging.info(outstr)

            if epoch_done:
                print("---Training Epoch:", curEpoch, "---", flush=True)
                train_loss = np.sum(np.asarray(epochLoss))
                train_sel_loss = np.sum(np.asarray(epochSelLoss))
                train_sent_loss = np.sum(np.asarray(epochSentLoss))
                train_copy_loss = np.sum(np.asarray(epochCopyLoss))
                train_sim_loss = np.sum(np.asarray(epochSimLoss))
                # Set eval mode on
                self.model.eval()

                val_epoch_done = False
                valstep = 0
                needed = {}
                needed["sentences"] = []
                needed["predicted"] = []
                needed["output"] = []
                needed["type"] = []
                needed["gold_entities"] = []
                needed["kb"] = []

                while not val_epoch_done:
                    valstep = valstep + 1
                    val_batch, val_epoch_done = self.handler.get_batch(
                        data="val")
                    val_feedDict = self.model.get_feed_dict(val_batch)
                    with torch.no_grad():
                        if self.args["model"] == 1:
                            sentences2, sketch_tags2 = self.model.forward(
                                val_feedDict["inp_utt"],
                                val_feedDict["copy_inp_pos"],
                                val_feedDict["inp_mask"],
                                val_feedDict["inp_len"],
                                val_feedDict["context_len"],
                                val_feedDict["kb"],
                                val_feedDict["copy_kb_pos"],
                                val_feedDict["kb_mask"],
                                val_feedDict["keys"],
                                val_feedDict["keys_mask"],
                                val_feedDict["db_empty"],
                                val_feedDict["out_utt"],
                                val_feedDict["copy_out_utt"],
                                val_feedDict["sketch_tags"],
                                val_feedDict["copy_sketch_tag_pos"],
                                val_feedDict["out_len"],
                                val_feedDict["max_out_utt_len"],
                                val_feedDict["sketch_mask"],
                                val_feedDict["sketch_loss_mask"],
                                False,
                                [0, 0, 0]
                            )
                        else:
                            sentences2, sketch_tags2, gb, _, _ = self.model.forward(
                                val_feedDict["did"],
                                val_feedDict["window_inp_utt"],
                                val_feedDict["window_inp_len"],
                                val_feedDict["window_context_len"],
                                val_feedDict["inp_utt"],
                                val_feedDict["copy_inp_pos"],
                                val_feedDict["inp_mask"],
                                val_feedDict["inp_len"],
                                val_feedDict["context_len"],
                                val_feedDict["kb"],
                                val_feedDict["copy_kb_pos"],
                                val_feedDict["kb_mask"],
                                val_feedDict["keys"],
                                val_feedDict["keys_mask"],
                                val_feedDict["db_empty"],
                                val_feedDict["out_utt"],
                                val_feedDict["copy_out_utt"],
                                val_feedDict["sketch_tags"],
                                val_feedDict["copy_sketch_tag_pos"],
                                val_feedDict["out_len"],
                                val_feedDict["max_out_utt_len"],
                                val_feedDict["sketch_mask"],
                                val_feedDict["sketch_loss_mask"],
                                val_feedDict["beta_filter"],
                                val_feedDict["type_kb_entities"],
                                False,
                                [0, 0, 0, 0],
                                "DEBUG_STUFF"
                            )

                    sentences = sentences2.detach().cpu()
                    if sketch_tags2 is not None:
                        sketch_tags = sketch_tags2.detach().cpu()
                        del sketch_tags2
                    del sentences2
                    del val_feedDict
                    gc.collect()
                    torch.cuda.empty_cache()
                    if 1 not in val_batch["dummy"]:
                        if self.args["model"] == 2:
                            predicted = self.get_predicted(
                                sentences, sketch_tags)
                        elif self.args["model"] == 1:
                            predicted = self.get_predicted(sentences, None)

                        needed["predicted"].extend(predicted)
                        needed["output"].extend(val_batch["copy_out_pos"])
                        needed["kb"].extend(val_batch["knowledge"])
                        needed["gold_entities"].extend(
                            val_batch["gold_entities"])
                        if (self.args["dataset"] == 1) or (self.args["dataset"] == 3):
                            needed["type"].extend(val_batch["type"])
                    else:
                        index = val_batch["dummy"].index(1)
                        if self.args["model"] == 2:
                            predicted = self.get_predicted(
                                sentences[0:index], sketch_tags[0:index]
                            )
                        elif self.args["model"] == 1:
                            predicted = self.get_predicted(
                                sentences[0:index], None)
                        needed["predicted"].extend(predicted)
                        needed["output"].extend(
                            val_batch["copy_out_pos"][0:index])
                        needed["kb"].extend(val_batch["knowledge"][0:index])
                        needed["gold_entities"].extend(
                            val_batch["gold_entities"][0:index]
                        )
                        if (self.args["dataset"] == 1) or (self.args["dataset"] == 3):
                            needed["type"].extend(val_batch["type"][0:index])

                outstr = (
                    "Train-info: "
                    + "Epoch: "
                    + str(curEpoch)
                    + " Loss: "
                    + str(train_loss.item())
                    + " Copy Loss: "
                    + str(train_copy_loss.item())
                    + " Sent Loss: "
                    + str(train_sent_loss.item())
                    + " Sel Loss: "
                    + str(train_sel_loss.item())
                    + " Similarity Loss: "
                    + str(train_sim_loss.item())
                )
                logging.info(outstr)

                print("", flush=True)
                if self.args["dataset"] == 1:
                    current_f1 = self.evaluate(needed, self.handler.vocab, ["schedule", "navigate", "weather"])
                elif self.args["dataset"] == 2:
                    current_f1 = self.evaluate_camrest(needed, self.handler.vocab)
                elif self.args["dataset"] == 3:
                    current_f1 = self.evaluate(needed, self.handler.vocab, ["hotel", "attraction", "restaurant"])
                
                print("", flush=True)
                sys.stdout.flush()

                epochLoss = []
                epochCopyLoss = []
                epochSentLoss = []
                epochSelLoss = []
                epochSimLoss = []

                # if current_f1 >= best_f1:
                #     best_f1 = current_f1
                #     print("---Test---", flush=True)
                #     self.save_model(self.model, self.ckpt_path, str(curEpoch))
                #     self.test()
                #     print("----------\n", flush=True)

                print("---Test---", flush=True)
                best_f1 = current_f1
                self.save_model(self.model, self.ckpt_path, str(curEpoch))
                self.test()
                print("----------\n", flush=True)

                curEpoch = curEpoch + 1
                # Set back train mode on
                gc.collect()
                torch.cuda.empty_cache()
                self.model.train()

    def get_predicted(self, generated, sketch_scores):
        vocab = self.handler.vocab

        predicted = []
        for i in range(0, len(generated)):
            sentence = generated[i]
            sentence = sentence.tolist()
            if vocab["vocab_mapping"]["$STOP$"] not in sentence:
                index = len(sentence)
            else:
                index = sentence.index(vocab["vocab_mapping"]["$STOP$"])
            outputs = [str(sentence[j]) for j in range(0, index)]
            outputs_anon = []
            for word in outputs:
                if word in vocab["rev_mapping"]:
                    outputs_anon.append(vocab["rev_mapping"][word])
                else:
                    outputs_anon.append(vocab["rev_unk_mapping"][word])

            sentence_predicted = []
            sketch_predicted = []
            if self.args["model"] == 2:
                sketch_score = sketch_scores[i].tolist()
                kb_words_used = []
                for j in range(0, len(outputs_anon)):
                    word = outputs_anon[j]
                    if "sketch_" in word:
                        word_scores = np.asarray(sketch_score[j])
                        while True:
                            word_index = np.argmax(word_scores)
                            if str(word_index) in vocab["rev_mapping"]:
                                kb_word = vocab["rev_mapping"][str(word_index)]
                            else:
                                kb_word = vocab["rev_unk_mapping"][str(
                                    word_index)]

                            if kb_word == "$PAD$":
                                word_scores[word_index] = -1
                                word_index = np.argmax(word_scores)
                                if str(word_index) in vocab["rev_mapping"]:
                                    kb_word = vocab["rev_mapping"][str(
                                        word_index)]
                                else:
                                    kb_word = vocab["rev_unk_mapping"][str(
                                        word_index)]
                            if kb_word in kb_words_used:
                                word_scores[word_index] = -1
                            else:
                                break
                        kb_words_used.append(kb_word)
                        sentence_predicted.append(kb_word)
                        sketch_predicted.append(word)
                    else:
                        sentence_predicted.append(word)
                        sketch_predicted.append(word)
            elif self.args["model"] == 1:
                for j in range(0, len(outputs_anon)):
                    word = outputs_anon[j]
                    sentence_predicted.append(word)
                    sketch_predicted.append(word)
            predicted.append(
                (copy.copy(sentence_predicted), copy.copy(sketch_predicted))
            )
        return predicted

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
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict(
                        (ng, min(count, max_counts[ng]))
                        for ng, count in hypcnts.items()
                    )
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        if c > 0:
            bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        else:
            print("bp is 0", flush=True)
            bp = 1
        p_ns = [float(clip_count[i]) / float(count[i] + p0) +
                p0 for i in range(4)]
        s = math.fsum(w * math.log(p_n)
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu

    def get_all_kbentities(self, kb):
        kb_entitites = []
        for item in kb:
            for key in item:
                entity = item[key]
                if entity not in kb_entitites:
                    kb_entitites.append(entity)
        return kb_entitites

    def evaluate_camrest(self, data, vocab):
        outs = []
        golds = []
        macro_f1 = 0.0
        macro_mse_f1 = 0.0
        total_cnt = 0

        for i in range(0, len(data["predicted"])):
            ground = data["output"][i]
            ground = list(ground)
            gold_entities = data["gold_entities"][i]
            kb_entitites = self.get_all_kbentities(data["kb"][i])
            index = ground.index(vocab["vocab_mapping"]["$STOP$"])
            ground_truth = [str(ground[j]) for j in range(0, index)]

            gold_anon = []
            for word in ground_truth:
                if word in vocab["rev_mapping"]:
                    gold_anon.append(vocab["rev_mapping"][word])
                else:
                    gold_anon.append(vocab["rev_unk_mapping"][word])

            # gold_anon = [vocab['rev_mapping'][word] for word in ground_truth ]
            out_anon = data["predicted"][i][0]
            sketch_anon = data["predicted"][i][1]
            loc_f1, cnt = self.get_f1(
                out_anon,
                gold_entities,
                gold_anon,
                kb_entitites,
                self.handler.all_entities,
            )
            loc_mse_f1, _ = self.get_mse_f1(
                out_anon,
                gold_entities,
                gold_anon,
                kb_entitites,
                self.handler.all_entities,
            )
            macro_f1 = macro_f1 + cnt * loc_f1
            macro_mse_f1 = macro_mse_f1 + cnt * loc_mse_f1
            total_cnt += cnt

            gold = gold_anon
            out = out_anon
            golds.append(" ".join(gold))
            outs.append(" ".join(out))
            if self.args["logs"]:
                # gb = [round(x,4) for x in data['gb'][i]]
                # for l in data['kb'][i]:
                #     print("restaurant:kb:",l)
                # for l in data['context'][i]:
                #     print("restaurant:context:",l)
                print("restaurant:gold:", " ".join(gold), flush=True)
                print("restaurant:gold_entities:", gold_entities, flush=True)
                print("restaurant:sketch:", " ".join(sketch_anon), flush=True)
                print("restaurant:out:", " ".join(out), flush=True)
                print("restaurant:loc_f1:", cnt * loc_f1, flush=True)
                # print("restaurant:gbcr:",gb[:len(data['kb'][i])])
                # print('')
                # print("loc_sketch_f1:", cnt*loc_sketch_f1, flush=True)
        wrap_generated = [[_] for _ in outs]
        wrap_truth = [[_] for _ in golds]
        print(
            "Bleu: %.3f, F1: %.3f, MACRO-MSE-F1: %.3f"
            % (
                self.score(zip(wrap_generated, wrap_truth)),
                macro_f1 / total_cnt,
                macro_mse_f1 / total_cnt,
            ),
            flush=True,
        )
        return macro_mse_f1 / total_cnt

    def get_mse_f1(self, out_anon, gold_entities, gold_anon, ent1, ent2):
        if len(gold_entities) != 0:
            predict_map = {}
            for word in out_anon:
                if word in ent1 or word in ent2:
                    predict_map[word] = predict_map.get(word, 0) + 1

            gold_map = {}
            for word in gold_anon:
                if word in ent1 or word in ent2:
                    gold_map[word] = gold_map.get(word, 0) + 1

            TP, FP, FN = 0, 0, 0
            for word in predict_map:
                if word not in gold_map:
                    FP += predict_map[word]
                elif predict_map[word] > gold_map[word]:
                    FP += predict_map[word] - gold_map[word]

            for word in gold_map:
                if word in predict_map:
                    TP += min(predict_map[word], gold_map[word])
                    if gold_map[word] > predict_map[word]:
                        FN += gold_map[word] - predict_map[word]
                else:
                    FN += gold_map[word]
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = (
                2 * precision * recall / float(precision + recall)
                if (precision + recall) != 0
                else 0
            )
            return F1, 1
        else:
            return 0, 0

    def get_f1(self, out_anon, gold_entities, gold_anon, ent1, ent2):
      if len(gold_entities) != 0:
          TP, FP, FN = 0, 0, 0
          for word in out_anon:
              if word in ent1 or word in ent2:
                  if word not in gold_entities:
                      FP += 1

          for word in gold_entities:
              if word in out_anon:
                  TP += 1
              else:
                  FN += 1
          precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
          recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
          F1 = 2 * precision * recall / float(precision+recall) if (precision+recall)!=0 else 0
          return F1, 1
      else:
          return 0,0

    def evaluate(self, data, vocab, domains):
        entity_to_sketch = self.handler.mentionToTypeMap
        sketch_types = self.handler.unique_types

        outs = []
        golds = []
        domain_wise = {}
        for domain in domains:
            domain_wise[domain] = {}
            domain_wise[domain]["total_cnt"] = 0
            domain_wise[domain]["macro_f1"] = 0.0
            domain_wise[domain]["macro_mse_f1"] = 0.0
            domain_wise[domain]["macro_sketch_f1"] = 0.0
            domain_wise[domain]["gold"] = []
            domain_wise[domain]["output"] = []

        macro_f1 = 0.0
        macro_mse_f1 = 0.0
        macro_sketch_f1 = 0.0
        total_cnt = 0

        for i in range(0, len(data["predicted"])):
            domain = data["type"][i]
            ground = data["output"][i]
            gold_entities = data["gold_entities"][i]
            gold_sketch = ["sketch_" + entity_to_sketch[gd]
                           for gd in gold_entities]

            kb_entitites = self.get_all_kbentities(data["kb"][i])
            ground = list(ground)
            index = ground.index(vocab["vocab_mapping"]["$STOP$"])
            ground_truth = [str(ground[j]) for j in range(0, index)]

            gold_anon = []
            for word in ground_truth:
                if word in vocab["rev_mapping"]:
                    gold_anon.append(vocab["rev_mapping"][word])
                else:
                    gold_anon.append(vocab["rev_unk_mapping"][word])

            out_anon = data["predicted"][i][0]
            sketch_anon = data["predicted"][i][1]

            (
                loc_sketch_f1,
                sketch_cnt
            ) = self.get_f1(
                sketch_anon, gold_sketch, gold_sketch, list(sketch_types), []
            )
            macro_sketch_f1 = macro_sketch_f1 + sketch_cnt * loc_sketch_f1
            domain_wise[domain]["macro_sketch_f1"] = (
                domain_wise[domain]["macro_sketch_f1"] +
                sketch_cnt * loc_sketch_f1
            )

            loc_f1, cnt = self.get_f1(
                out_anon,
                gold_entities,
                gold_anon,
                kb_entitites,
                self.handler.all_entities,
            )
            loc_mse_f1, _ = self.get_mse_f1(
                out_anon,
                gold_entities,
                gold_anon,
                kb_entitites,
                self.handler.all_entities,
            )
            macro_f1 = macro_f1 + cnt * loc_f1
            macro_mse_f1 = macro_mse_f1 + cnt * loc_mse_f1
            total_cnt += cnt
            domain_wise[domain]["macro_f1"] = (
                domain_wise[domain]["macro_f1"] + cnt * loc_f1
            )
            domain_wise[domain]["macro_mse_f1"] = (
                domain_wise[domain]["macro_mse_f1"] + cnt * loc_mse_f1
            )
            domain_wise[domain]["total_cnt"] += cnt

            gold = gold_anon
            out = out_anon
            if self.args["logs"]:
                if cnt > 0:
                    try:
                        # gb = [round(x,4) for x in data['gb'][i]]
                        # gblt = [round(x,4) for x in data['gblt'][i]]
                        # gbht = [round(x,4) for x in data['gbht'][i]]

                        # for l in data['kb'][i]:
                        #     print(str(domain)+":kb:",l)
                        # for l in data['context'][i]:
                        #     print(str(domain)+":context:",l)
                        print(str(domain) + ":gold:",
                              " ".join(gold), flush=True)
                        print(str(domain) + ":gold_entities:",
                              ",".join(gold_entities), flush=True)
                        print(
                            str(domain) + ":sketch:", " ".join(sketch_anon), flush=True
                        )
                        print(str(domain) + ":out:", " ".join(out), flush=True)
                        print(str(domain) + ":loc_f1:",
                              cnt * loc_f1, flush=True)
                        print(
                            str(domain) + ":loc_sketch_f1:",
                            cnt * loc_sketch_f1,
                            flush=True,
                        )
                        # print(str(domain)+":gbcr:",gb[:len(data['kb'][i])])
                        # print(str(domain)+":netgb:",gblt[:len(data['kb'][i])])
                        # print(str(domain)+":gbht:",gbht[:len(data['kb'][i])])
                        # beta_filter = data['beta_filter'][i][:len(data['kb'][i])]
                        # beta_filter = [x / sum(data['beta_filter'][i][:len(data['kb'][i])]) for x in beta_filter]
                        # print(str(domain)+":heuristic:",beta_filter)
                        # print('')
                    except Exception as e:
                        print("Error:", e)
                        import ipdb

                        ipdb.set_trace()
            domain_wise[domain]["gold"].append(" ".join(gold))
            golds.append(" ".join(gold))
            domain_wise[domain]["output"].append(" ".join(out))
            outs.append(" ".join(out))

        wrap_generated = [[_] for _ in outs]
        wrap_truth = [[_] for _ in golds]

        print(
            "Bleu: %.3f, F1: %.3f, MACRO-MSE-F1: %.3f"
            % (
                self.score(zip(wrap_generated, wrap_truth)),
                macro_f1 / total_cnt,
                macro_mse_f1 / total_cnt,
            ),
            flush=True,
        )
        for domain in domains:
            wrap_generated = [[_] for _ in domain_wise[domain]["output"]]
            wrap_truth = [[_] for _ in domain_wise[domain]["gold"]]

            if domain_wise[domain]["total_cnt"] > 0:
                print(
                    "Domain: "
                    + str(domain)
                    + ", Bleu: %.3f, F1: %.3f, MACRO-MSE-F1: %.3f"
                    % (
                        self.score(zip(wrap_generated, wrap_truth)),
                        domain_wise[domain]["macro_f1"]
                        / domain_wise[domain]["total_cnt"],
                        domain_wise[domain]["macro_mse_f1"]
                        / domain_wise[domain]["total_cnt"],
                    ),
                    flush=True,
                )
            else:
                print(
                    "Domain: "
                    + str(domain)
                    + ", Bleu: %.3f, F1: %.3f, Sketch F1: %.3f"
                    % (self.score(zip(wrap_generated, wrap_truth)), 0, 0),
                    flush=True,
                )

        return macro_mse_f1 / total_cnt

    def test(self):
        self.model.eval()
        test_epoch_done = False

        teststep = 0
        needed = {}
        needed["predicted"] = []
        needed["output"] = []
        needed["type"] = []
        needed["context"] = []
        needed["kb"] = []
        needed["gold_entities"] = []

        needed["gb"] = []
        needed["gblt"] = []
        needed["gbht"] = []
        needed["beta_filter"] = []

        while not test_epoch_done:
            teststep = teststep + 1
            batch, test_epoch_done = self.handler.get_batch(data="tst")
            feedDict = self.model.get_feed_dict(batch)
            with torch.no_grad():
                if self.args["model"] == 1:
                    sentences2, sketch_tags2 = self.model.forward(
                        feedDict["inp_utt"],
                        feedDict["copy_inp_pos"],
                        feedDict["inp_mask"],
                        feedDict["inp_len"],
                        feedDict["context_len"],
                        feedDict["kb"],
                        feedDict["copy_kb_pos"],
                        feedDict["kb_mask"],
                        feedDict["keys"],
                        feedDict["keys_mask"],
                        feedDict["db_empty"],
                        feedDict["out_utt"],
                        feedDict["copy_out_utt"],
                        feedDict["sketch_tags"],
                        feedDict["copy_sketch_tag_pos"],
                        feedDict["out_len"],
                        feedDict["max_out_utt_len"],
                        feedDict["sketch_mask"],
                        feedDict["sketch_loss_mask"],
                        False,
                        [0, 0, 0],
                    )
                else:
                    sentences2, sketch_tags2, gb, _, _ = self.model.forward(
                        feedDict["did"],
                        feedDict["window_inp_utt"],
                        feedDict["window_inp_len"],
                        feedDict["window_context_len"],
                        feedDict["inp_utt"],
                        feedDict["copy_inp_pos"],
                        feedDict["inp_mask"],
                        feedDict["inp_len"],
                        feedDict["context_len"],
                        feedDict["kb"],
                        feedDict["copy_kb_pos"],
                        feedDict["kb_mask"],
                        feedDict["keys"],
                        feedDict["keys_mask"],
                        feedDict["db_empty"],
                        feedDict["out_utt"],
                        feedDict["copy_out_utt"],
                        feedDict["sketch_tags"],
                        feedDict["copy_sketch_tag_pos"],
                        feedDict["out_len"],
                        feedDict["max_out_utt_len"],
                        feedDict["sketch_mask"],
                        feedDict["sketch_loss_mask"],
                        feedDict["beta_filter"],
                        feedDict["type_kb_entities"],
                        False,
                        [0, 0, 0, 0],
                        "DEBUG_STUFF",
                    )
                    # gb = gb.detach().cpu().tolist()
                    # gbht = gbht.detach().cpu().tolist()
                    # gblt = gblt.detach().cpu().tolist()
            # print(sentences)
            sentences = sentences2.detach().cpu()
            if sketch_tags2 is not None:
                sketch_tags = sketch_tags2.detach().cpu()
                del sketch_tags2
            del sentences2
            del feedDict
            gc.collect()

            if 1 not in batch["dummy"]:
                if self.args["model"] == 2:
                    predicted = self.get_predicted(sentences, sketch_tags)
                elif self.args["model"] == 1:
                    predicted = self.get_predicted(sentences, None)
                # predicted = self.get_predicted(sentences, sketch_tags)
                needed["predicted"].extend(predicted)
                needed["output"].extend(batch["copy_out_pos"])
                needed["context"].extend(batch["context"])
                needed["gold_entities"].extend(batch["gold_entities"])
                needed["kb"].extend(batch["knowledge"])
                if (self.args["dataset"] == 1) or (self.args["dataset"] == 3):
                    needed["type"].extend(batch["type"])
                # needed["gb"].extend(gb)
                # needed['gbht'].extend(gbht)
                # needed['gblt'].extend(gblt)
                needed["beta_filter"].extend(batch["beta_filter"])
            else:
                index = batch["dummy"].index(1)
                if self.args["model"] == 2:
                    predicted = self.get_predicted(
                        sentences[0:index], sketch_tags[0:index]
                    )
                elif self.args["model"] == 1:
                    predicted = self.get_predicted(sentences[0:index], None)
                # predicted = self.get_predicted(sentences[0:index], sketch_tags[0:index])
                needed["predicted"].extend(predicted)
                needed["output"].extend(batch["copy_out_pos"][0:index])
                needed["context"].extend(batch["context"][0:index])
                needed["gold_entities"].extend(batch["gold_entities"][0:index])
                needed["kb"].extend(batch["knowledge"][0:index])
                if (self.args["dataset"] == 1) or (self.args["dataset"] == 3):
                    needed["type"].extend(batch["type"][0:index])
                # needed["gb"].extend(gb[:index])
                # needed['gbht'].extend(gbht[:index])
                # needed['gblt'].extend(gblt[:index])
                needed["beta_filter"].extend(batch["beta_filter"][0:index])
        pickle.dump(needed, open("needed.p", "wb"))
        if self.args["dataset"] == 1:
            self.evaluate(needed, self.handler.vocab, ["schedule", "navigate", "weather"])
        elif self.args["dataset"] == 2:
            self.evaluate_camrest(needed, self.handler.vocab)
        elif self.args["dataset"] == 3:
            self.evaluate(needed, self.handler.vocab, ["hotel", "attraction", "restaurant"])

        torch.cuda.empty_cache()
        self.model.train()


def main():
    if args["seed"]:
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        # torch.set_deterministic(True)
    logging.info("Loading Data")

    if args['dataset'] == 3:
        from data_handler_woz import DataHandler
    else:
        from data_handler import DataHandler

    handler = DataHandler(
        emb_dim=args["emb_dim"],
        batch_size=args["batch"],
        train_path=args["data"] + "train.json",
        val_path=args["data"] + "val.json",
        test_path=args["data"] + "test.json",
        entities_path=args["data"] + "entities.json",
        vocab_path=args["data"] + args["vocab"],
        args=args,
    )

    logging.info("Loading Architecture")

    args["device"] = DEVICE
    if args["load"]:
        model = torch.load(str(args["load"])).to(DEVICE)
    else:
        if args["model"] == 1:
            model = MLMModel(
                args=args,
                emb_init=handler.emb_init,
                generate_size=handler.generate_vocab_size,
                out_vocab_size=handler.output_vocab_size,
                total_copy_vocab_size=handler.total_copy_vocab_size,
                eos=handler.vocab["vocab_mapping"]["$STOP$"],
            ).to(DEVICE)
        else:
            model = AggNet(
                args=args,
                emb_init=handler.emb_init,
                generate_size=handler.generate_vocab_size,
                out_vocab_size=handler.output_vocab_size,
                total_copy_vocab_size=handler.total_copy_vocab_size,
                eos=handler.vocab["vocab_mapping"]["$STOP$"],
                rev_vocab=handler.vocab["rev_mapping"],
            ).to(DEVICE)
    logging.info("Loading Trainer")

    trainer = Trainer(
        model=model,
        handler=handler,
        ckpt_path=args["ckpt_path"],
        num_epochs=args["num_epochs"],
        learning_rate=args["lr"],
        clip=args["clip"],
        args=args,
    )

    if args["load"]:
        trainer.test()
        if args["test"] is False:
            init_load = (
                int(args["load"].split("/")[-1].strip().split("_")
                    [1].split(".")[0].strip())
                + 1
            )
            trainer.trainData(init_epoch=init_load)
    else:
        trainer.trainData()

main()
