import json
import copy
import random

# import nltk
import os
import sys
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)


class DataHandler(object):
    def __init__(
        self,
        emb_dim,
        batch_size,
        train_path,
        val_path,
        test_path,
        entities_path,
        vocab_path,
        args,
    ):
        # self.global_count = {}
        # self.global_count2 = {}
        self.args = args
        if self.args["seed"]:
            np.random.seed(self.args["seed"])
            random.seed(self.args["seed"])

        self.batch_size = batch_size
        self.train_path = train_path
        # self.vocab_threshold = 3
        self.val_path = val_path
        self.test_path = test_path
        self.vocab_path = vocab_path
        self.emb_dim = emb_dim
        self.entities_path = entities_path

        full_entities = json.load(open(self.entities_path, "r", encoding="utf-8"))
        # if self.args['dataset'] == 1:
        self.unique_types = set()
        self.mentionToTypeMap = {}
        for entity_type, mentions in full_entities["all_entities"].items():
            for mention in mentions:
                self.mentionToTypeMap[mention] = entity_type
            self.unique_types.add("sketch_" + entity_type)
        self.all_entities = full_entities["all_entities_list"]

        self.vocab = self.load_vocab()
        self.input_vocab_size = self.vocab["input_vocab_size"]
        self.output_vocab_size = self.vocab["output_vocab_size"]
        self.total_copy_vocab_size = self.vocab["total_copy_vocab_size"]
        self.generate_vocab_size = self.vocab["generate_vocab_size"]
        self.emb_init = self.load_glove_vectors()

        self.train_data = json.load(open(self.train_path, "r", encoding="utf-8"))
        self.val_data = json.load(open(self.val_path, "r", encoding="utf-8"))
        self.test_data = json.load(open(self.test_path, "r", encoding="utf-8"))

        random.shuffle(self.train_data)
        # random.shuffle(self.val_data)
        # random.shuffle(self.test_data)

        self.val_data_full = self.append_dummy_data(self.val_data)
        self.test_data_full = self.append_dummy_data(self.test_data)

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.train_num = len(self.train_data)
        self.val_num = len(self.val_data_full)
        self.test_num = len(self.test_data_full)

    def append_dummy_data(self, data):
        new_data = []
        for i in range(0, len(data)):
            data[i]["dummy"] = 0
            new_data.append(copy.copy(data[i]))

        last = data[-1]
        last["dummy"] = 1
        for _ in range(0, self.batch_size - len(data) % self.batch_size):
            new_data.append(copy.copy(last))

        return copy.copy(new_data)

    def load_glove_vectors(self):
        if self.args["abl_glove"]:
            logging.info("Loading pre-trained Word Embeddings")
            filename = "../data/glove.6B.200d.txt"
            # filename = "../data/cc.en.300.vec"
            glove = {}
            file = open(filename, "r", encoding="utf-8")
            row_count = 0
            for line in file.readlines():
                # if row_count >= 1:
                row = line.strip().split(" ")
                glove[row[0]] = np.asarray(row[1:])
                # row_count += 1

            logging.info("Loaded GloVe!")
            file.close()
            embeddings_init = np.random.normal(
                size=(self.vocab["input_vocab_size"], self.emb_dim)
            ).astype("f")
            count = 0
            for word in self.vocab["vocab_mapping"]:
                if word in glove:
                    count = count + 1
                    embeddings_init[self.vocab["vocab_mapping"][word]] = glove[word]
            del glove

            logging.info("Loaded " + str(count) + " pre-trained Word Embeddings")
        else:
            embeddings_init = np.random.normal(
                size=(self.vocab["input_vocab_size"], self.emb_dim)
            ).astype("f")

        self.args["pad"] = self.vocab["vocab_mapping"]["$PAD$"]
        return embeddings_init

    def load_vocab(self):
        if os.path.isfile(self.vocab_path):
            logging.info("Loading vocab from file")
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                vocab_dict = json.load(f)
                self.args["unk_index"] = vocab_dict["vocab_mapping"]["$UNK$"]
                return vocab_dict
        else:
            logging.info("Vocab file not found. Computing Vocab")
            with open(self.train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            with open(self.val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            with open(self.test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            full_data = []
            full_data.extend(train_data)
            full_data.extend(val_data)
            # full_data.extend(test_data)

            # total_data = []
            # total_data.extend(train_data)
            # total_data.extend(val_data)
            # total_data.extend(test_data)

            return self.get_vocab(full_data, test_data)

    def get_vocab(self, data, test_data):
        vocab = {}

        gen_words = set()
        copy_words = set()
        for dial in data:
            for word in dial['sketch_outs']:
                gen_words.add(word)

        for dial in data:
            for word in dial['gold_entities']:
                copy_words.add(word)

        common_words = copy_words.intersection(gen_words)
        # print("-----------:", 'on' in common_words)

        for d in data:
            utts = []
            utts.append(" ".join(d["sketch_outs"]))
            utts.append(" ".join(d["sketch_tags"]))
            utts.append(d["output"])
            utts.extend(d["context"])
            for utt in utts:
                tokens = utt.split(" ")
                for token in tokens:
                    if token != "null_tag":
                        if token.lower() not in vocab:
                            vocab[token.lower()] = 1
                        else:
                            vocab[token.lower()] = vocab[token.lower()] + 1

            for item in d["kb"]:
                for key in item:
                    if key.lower() not in vocab:
                        vocab[key.lower()] = 1
                    else:
                        vocab[key.lower()] = vocab[key.lower()] + 1
                    token = item[key]
                    if token.lower() not in vocab:
                        vocab[token.lower()] = 1
                    else:
                        vocab[token.lower()] = vocab[token.lower()] + 1

        test_vocab = {}
        for d in test_data:
            utts = []
            utts.append(" ".join(d["sketch_outs"]))
            utts.append(" ".join(d["sketch_tags"]))
            utts.append(d["output"])
            utts.extend(d["context"])
            for utt in utts:
                tokens = utt.split(" ")
                for token in tokens:
                    if token.lower() not in vocab:
                        if token != "null_tag":
                            if token.lower() not in test_vocab:
                                test_vocab[token.lower()] = 1
                            else:
                                test_vocab[token.lower()] = (
                                    test_vocab[token.lower()] + 1
                                )

            for item in d["kb"]:
                for key in item:
                    if key.lower() not in vocab:
                        if key.lower() not in test_vocab:
                            test_vocab[key.lower()] = 1
                        else:
                            test_vocab[key.lower()] = test_vocab[key.lower()] + 1
                    token = item[key]

                    if token.lower() not in vocab:
                        if token.lower() not in test_vocab:
                            test_vocab[token.lower()] = 1
                        else:
                            test_vocab[token.lower()] = test_vocab[token.lower()] + 1

        test_words = list(test_vocab.keys())
        words = list(vocab.keys())
        words.append("$STOP$")
        words.append("$PAD$")
        words.append("$UNK$")
        words.append("$TAG$")

        length_dialog = 0
        if self.args["dataset"] == 1:
            length_dialog = 6
        elif self.args["dataset"] == 2:
            length_dialog = 9
        elif self.args["dataset"] == 3:
            length_dialog = 12
        words.append("$u")
        words.append("$s")

        for i in range(1, length_dialog):
            words.append("$" + str(i))
            words.append("$" + str(i))
        words.append("$" + str(length_dialog))

        generate_words = []
        copy_words = []
        for word in words:
            # if (word in self.all_entities) or (("_" in word) and ("sketch_" not in word)):
            if (word in self.all_entities) and (word not in common_words):
                # if self.args["dataset"] == 2:
                #     if word != "api_call":
                #         copy_words.append(word)
                #     else:
                #         generate_words.append(word)
                # else:
                copy_words.append(word)
            else:
                generate_words.append(word)

        output_vocab_size = len(words) + 1
        total_copy_vocab_size = output_vocab_size + len(test_words)

        generate_indices = [i for i in range(1, len(generate_words) + 1)]
        copy_indices = [i for i in range(len(generate_words) + 1, output_vocab_size)]
        left_over_indices = [i for i in range(output_vocab_size, total_copy_vocab_size)]

        random.shuffle(generate_indices)
        random.shuffle(copy_indices)
        random.shuffle(left_over_indices)

        mapping = {}
        rev_mapping = {}
        unk_mapping = {}
        rev_unk_mapping = {}

        for i in range(0, len(generate_words)):
            mapping[generate_words[i]] = generate_indices[i]
            rev_mapping[str(generate_indices[i])] = generate_words[i]

        for i in range(0, len(copy_words)):
            mapping[copy_words[i]] = copy_indices[i]
            rev_mapping[str(copy_indices[i])] = copy_words[i]

        for i in range(0, len(test_words)):
            unk_mapping[test_words[i]] = left_over_indices[i]
            rev_unk_mapping[str(left_over_indices[i])] = test_words[i]

        mapping["$GO$"] = 0
        rev_mapping["0"] = "$GO$"
        vocab_dict = {}
        vocab_dict["vocab_mapping"] = mapping
        vocab_dict["rev_mapping"] = rev_mapping
        vocab_dict["unk_mapping"] = unk_mapping
        vocab_dict["rev_unk_mapping"] = rev_unk_mapping
        self.args["unk_index"] = vocab_dict["vocab_mapping"]["$UNK$"]
        vocab_dict["input_vocab_size"] = len(words) + 1
        vocab_dict["generate_vocab_size"] = len(generate_words) + 1
        vocab_dict["output_vocab_size"] = output_vocab_size
        vocab_dict["total_copy_vocab_size"] = total_copy_vocab_size

        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f)

        logging.info("Vocab file created")

        return vocab_dict

    def get_sentinel(self, i, context):
        if i % 2 == 0:
            speaker = "u"
            turn = (context - i + 1) / 2
        else:
            speaker = "s"
            turn = (context - i) / 2
        return ["$" + speaker, "$" + str(int(turn))]

    def vectorize(self, batch, train):
        vectorized = {}
        vectorized["did"] = []

        vectorized["window_inp_utt"] = []
        vectorized["window_inp_len"] = []
        vectorized["window_context_len"] = []

        vectorized["inp_mask"] = []
        vectorized["inp_utt"] = []
        vectorized["out_utt"] = []
        vectorized["inp_len"] = []
        vectorized["context_len"] = []
        vectorized["out_len"] = []
        vectorized["kb"] = []
        vectorized["kb_mask"] = []
        vectorized["keys"] = []
        vectorized["keys_mask"] = []
        vectorized["type"] = []
        vectorized["dummy"] = []
        vectorized["empty"] = []
        vectorized["sketch_outs"] = []
        vectorized["sketch_tags"] = []
        vectorized["sketch_mask"] = []
        vectorized["sketch_loss_mask"] = []
        vectorized["gold_entities"] = []

        vectorized["selector"] = []  # For global pointer business
        vectorized["row_selector"] = []
        vectorized["key_selector"] = []
        vectorized["beta_key_selector"] = []
        vectorized["beta_filter"] = []

        # same as inp_utt but at places of unk has a unique index corresponding to correct entity to accumulate weights
        vectorized["copy_inp_pos"] = []
        vectorized["copy_out_pos"] = []
        vectorized["copy_sketch_out_pos"] = []
        vectorized["copy_sketch_tag_pos"] = []
        vectorized["copy_kb_pos"] = []

        vectorized["type_kb_entities"] = []

        vectorized["knowledge"] = []
        vectorized["context"] = []
        max_inp_utt_len = 0
        max_out_utt_len = 0
        max_context_len = 0
        max_window_context_len = 0
        kb_len = 0
        keys_len = 6
        if self.args["dataset"] == 2:
            keys_len = 7
        elif self.args["dataset"] == 3:
            keys_len = 10

        for item in batch:
            if len(item["context"]) > max_context_len:
                max_context_len = len(item["context"])

            window_context = 0
            for utt in item["context"]:
                tokens = utt.split(" ")

                if len(tokens) > max_inp_utt_len:
                    max_inp_utt_len = len(tokens)

                if self.args["abl_window"] > 0:
                    window_context += max(1, len(tokens) - self.args["abl_window"] + 1)

            tokens = item["output"].split(" ")
            if train and (self.args["abl_window"] > 0):
                window_context += max(1, len(tokens) - self.args["abl_window"] + 1)
            max_window_context_len = max(max_window_context_len, window_context)
            if len(tokens) > max_out_utt_len:
                max_out_utt_len = len(tokens)

            if len(item["kb"]) > kb_len:
                kb_len = len(item["kb"])

        max_inp_utt_len = max_inp_utt_len + 2

        max_out_utt_len = max_out_utt_len + 1
        vectorized["max_out_utt_len"] = max_out_utt_len

        for item in batch:
            vectorized["did"].append(item["did"])
            vectorized["context"].append(item["context"])
            vectorized["knowledge"].append(item["kb"])
            vectorized["gold_entities"].append(item["gold_entities"])
            if (self.args["dataset"] == 1) or (self.args["dataset"] == 3):
                vectorized["type"].append(item["type"])

            if item["kb"] == []:
                vectorized["empty"].append(0)
            else:
                vectorized["empty"].append(1)
            if not train:
                vectorized["dummy"].append(item["dummy"])
            vector_inp = []
            vector_copy_inp = []
            vector_len = []

            # Dataset modification for correction in missing gold entities and null tags
            # changed here
            modified_sketch_outs = copy.deepcopy(item["sketch_outs"])
            modified_sketch_tags = copy.deepcopy(item["sketch_tags"])

            # This check is only for Incar Dataset
            if (self.args["dataset"] == 1):
                if len(modified_sketch_outs) > len(modified_sketch_tags):
                    modified_sketch_tags = ["null_tag"] * len(modified_sketch_outs)

                for idx, token in enumerate(item["sketch_outs"]):
                    if token in self.vocab["vocab_mapping"]:
                        vocab_id = self.vocab["vocab_mapping"][token]
                        done = vocab_id >= self.generate_vocab_size
                    elif (token in self.all_entities) or (
                        ("_" in token) and ("sketch_" not in token)
                    ):
                        done = True
                    else:
                        vocab_id = -1
                        done = False

                    if done:
                        if token not in self.mentionToTypeMap:
                            print("Missing", token)
                            sys.exit(0)
                        modified_sketch_outs[idx] = (
                            "sketch_" + self.mentionToTypeMap[token]
                        )
                        modified_sketch_tags[idx] = token
                        vectorized["gold_entities"][-1].append(token)

            vector_sketch = []
            vector_copy_sketch = []
            # sketch_tokens = copy.copy(item['sketch_outs'])
            sketch_tokens = modified_sketch_outs
            sketch_tokens.append("$STOP$")
            for token in sketch_tokens:
                if token in self.vocab["vocab_mapping"]:
                    vector_sketch.append(self.vocab["vocab_mapping"][token])
                    vector_copy_sketch.append(self.vocab["vocab_mapping"][token])
                else:
                    vector_sketch.append(self.vocab["vocab_mapping"]["$UNK$"])
                    vector_copy_sketch.append(self.vocab["unk_mapping"][token])

            for _ in range(0, max_out_utt_len - len(sketch_tokens)):
                vector_sketch.append(self.vocab["vocab_mapping"]["$PAD$"])
                vector_copy_sketch.append(self.vocab["vocab_mapping"]["$PAD$"])
            vectorized["copy_sketch_out_pos"].append(copy.copy(vector_copy_sketch))
            vectorized["sketch_outs"].append(copy.copy(vector_sketch))
            vectorized["out_len"].append(len(sketch_tokens))

            vector_tags = []
            vector_copy_tags = []
            sketch_mask = []
            sketch_loss_mask = 1
            # tag_tokens = copy.copy(item['sketch_tags'])
            tag_tokens = modified_sketch_tags
            tag_tokens.append("null_tag")
            for token in tag_tokens:
                if token != "null_tag":
                    sketch_mask.append(1)
                    if token in self.vocab["vocab_mapping"]:
                        vector_tags.append(self.vocab["vocab_mapping"][token])
                        vector_copy_tags.append(self.vocab["vocab_mapping"][token])
                    else:
                        vector_tags.append(self.vocab["vocab_mapping"]["$UNK$"])
                        vector_copy_tags.append(self.vocab["unk_mapping"][token])
                else:
                    sketch_mask.append(0)
                    vector_tags.append(self.vocab["vocab_mapping"]["$TAG$"])
                    vector_copy_tags.append(self.vocab["vocab_mapping"]["$TAG$"])

            for _ in range(0, max_out_utt_len - len(tag_tokens)):
                sketch_mask.append(0)
                vector_tags.append(self.vocab["vocab_mapping"]["$PAD$"])
                vector_copy_tags.append(self.vocab["vocab_mapping"]["$PAD$"])

            if 1 not in sketch_mask:
                sketch_mask[max_out_utt_len - 1] = 1
                sketch_loss_mask = 0
            vectorized["copy_sketch_tag_pos"].append(copy.copy(vector_copy_tags))
            vectorized["sketch_tags"].append(copy.copy(vector_tags))
            vectorized["sketch_mask"].append(copy.copy(sketch_mask))
            vectorized["sketch_loss_mask"].append(sketch_loss_mask)
            # changed here

            for i in range(0, len(item["context"])):
                utt = item["context"][i]
                inp = []
                copy_inp = []
                sentinel = self.get_sentinel(i, len(item["context"]))
                tokens = utt.split(" ") + sentinel
                for token in tokens:
                    if token in self.vocab["vocab_mapping"]:
                        if token in vectorized["gold_entities"][-1]:
                            drop = random.random()
                            if drop < self.args["dropout"] and train:
                                inp.append(self.vocab["vocab_mapping"]["$UNK$"])
                                copy_inp.append(self.vocab["vocab_mapping"][token])
                            else:
                                inp.append(self.vocab["vocab_mapping"][token])
                                copy_inp.append(self.vocab["vocab_mapping"][token])
                        else:
                            inp.append(self.vocab["vocab_mapping"][token])
                            copy_inp.append(self.vocab["vocab_mapping"][token])
                    else:
                        inp.append(self.vocab["vocab_mapping"]["$UNK$"])
                        copy_inp.append(self.vocab["unk_mapping"][token])

                vector_len.append(len(tokens))
                for _ in range(0, max_inp_utt_len - len(tokens)):
                    inp.append(self.vocab["vocab_mapping"]["$PAD$"])
                    copy_inp.append(self.vocab["vocab_mapping"]["$PAD$"])
                vector_inp.append(copy.copy(inp))
                vector_copy_inp.append(copy.copy(copy_inp))

            vectorized["context_len"].append(len(item["context"]))

            for _ in range(0, max_context_len - len(item["context"])):
                vector_len.append(0)
                inp = []
                copy_inp = []
                for _ in range(0, max_inp_utt_len):
                    inp.append(self.vocab["vocab_mapping"]["$PAD$"])
                    copy_inp.append(self.vocab["vocab_mapping"]["$PAD$"])
                vector_inp.append(copy.copy(inp))
                vector_copy_inp.append(copy.copy(copy_inp))

            vectorized["copy_inp_pos"].append(copy.copy(vector_copy_inp))
            vectorized["inp_utt"].append(copy.copy(vector_inp))
            vectorized["inp_len"].append(vector_len)

            if self.args["abl_window"] > 0:
                vector_window_inp = []
                vector_window_len = []
                if train:
                    # In training append the response also
                    my_context = item["context"] + [item["output"]]
                else:
                    my_context = item["context"]

                for i in range(0, len(my_context)):
                    utt = my_context[i]
                    # num_windows = len(utt.split(" ")) - self.args["abl_window"] + 1
                    inp = []
                    sentinel = self.get_sentinel(i, len(my_context))
                    tokens = utt.split(" ")
                    index = 0
                    min_1_window = False
                    for token in tokens:
                        if token in self.vocab["vocab_mapping"]:
                            inp.append(self.vocab["vocab_mapping"][token])
                        else:
                            inp.append(self.vocab["vocab_mapping"]["$UNK$"])
                        index += 1

                        if index >= self.args["abl_window"]:
                            min_1_window = True
                            # Add sentinel token after each window
                            # vector_window_inp.append(copy.copy(inp)+[self.vocab['vocab_mapping'][sentinel[0]]])
                            vector_window_inp.append(copy.copy(inp))
                            vector_window_len.append(self.args["abl_window"])
                            inp = inp[1:]

                    if min_1_window is False:
                        # inp.append(self.vocab['vocab_mapping'][sentinel[0]])
                        vector_window_len.append(len(inp))
                        # +1 for sentinel
                        # for _ in range(0, self.args['abl_window'] - len(inp)+1):
                        for _ in range(0, self.args["abl_window"] - len(inp)):
                            inp.append(self.vocab["vocab_mapping"]["$PAD$"])
                        vector_window_inp.append(copy.copy(inp))

                vectorized["window_context_len"].append(len(vector_window_inp))

                for _ in range(0, max_window_context_len - len(vector_window_inp)):
                    vector_window_len.append(0)
                    inp = []
                    # for _ in range(0, self.args['abl_window']+1):
                    for _ in range(0, self.args["abl_window"]):
                        inp.append(self.vocab["vocab_mapping"]["$PAD$"])
                    vector_window_inp.append(copy.copy(inp))

                vectorized["window_inp_utt"].append(copy.deepcopy(vector_window_inp))
                vectorized["window_inp_len"].append(copy.copy(vector_window_len))

            vector_out = []
            vector_copy_out = []
            tokens = item["output"].split(" ")
            tokens.append("$STOP$")
            for token in tokens:
                if token in self.vocab["vocab_mapping"]:
                    vector_out.append(self.vocab["vocab_mapping"][token])
                    vector_copy_out.append(self.vocab["vocab_mapping"][token])
                else:
                    vector_out.append(self.vocab["vocab_mapping"]["$UNK$"])
                    vector_copy_out.append(self.vocab["unk_mapping"][token])

            for _ in range(0, max_out_utt_len - len(tokens)):
                vector_out.append(self.vocab["vocab_mapping"]["$PAD$"])
                vector_copy_out.append(self.vocab["vocab_mapping"]["$PAD$"])
            vectorized["out_utt"].append(copy.copy(vector_out))
            vectorized["copy_out_pos"].append(copy.copy(vector_copy_out))

            context_words_allowed_to_copy = []
            context_rows_allowed_to_copy = []
            row_mask = []
            for i in range(0, len(item["context"])):
                utt = item["context"][i]
                words_allowed_to_copy = []
                # Split tokens again so that I can remove $u types from copyable list
                tokens = utt.split(" ")
                outputs = item["output"].split(" ")
                flag = False
                for token in tokens:
                    if token in outputs:
                        words_allowed_to_copy.append(1)
                        if not flag:
                            context_rows_allowed_to_copy.append(1)
                            flag = True
                    elif token in self.all_entities:  # Entity word
                        words_allowed_to_copy.append(1)
                        if not flag:
                            context_rows_allowed_to_copy.append(1)
                            flag = True
                    else:
                        words_allowed_to_copy.append(0)
                if not flag:
                    context_rows_allowed_to_copy.append(0)
                # Add a 0 for sentinel tag and pads
                for _ in range(0, max_inp_utt_len - len(tokens)):
                    words_allowed_to_copy.append(0)
                context_words_allowed_to_copy.append(copy.copy(words_allowed_to_copy))
                row_mask.append(1)

            for _ in range(0, max_context_len - len(item["context"])):
                words_allowed_to_copy = []
                for _ in range(0, max_inp_utt_len):
                    words_allowed_to_copy.append(0)
                context_words_allowed_to_copy.append(copy.copy(words_allowed_to_copy))
                context_rows_allowed_to_copy.append(0)
                row_mask.append(0)

            vectorized["selector"].append(copy.copy(context_words_allowed_to_copy))
            vectorized["row_selector"].append(copy.copy(context_rows_allowed_to_copy))
            vectorized["inp_mask"].append(copy.copy(row_mask))

            vector_keys = []
            vector_keys_mask = []
            vector_kb = []
            vector_copy_kb = []
            vector_kb_mask = []
            vector_beta_filters = []
            if (self.args["dataset"] == 1):
                vector_selectors = copy.deepcopy(item["key_ptr"])
                vector_beta_selectors = copy.copy(item["kb_ptr"])
            else:
                vector_selectors = []
                vector_beta_selectors = []

            words_context = {}
            for line in item["context"] + [item["output"]]:
                for word in line.split(" "):
                    if word in words_context:
                        words_context[word] += 1
                    else:
                        words_context[word] = 1

            kb_key_types = {}
            count = 0
            for result in item["kb"]:
                beta_filter_found = 0
                if self.args["dataset"] != 1:
                    beta_found = 0
                    vector_select = []
                vector_result = []
                vector_copy_result = []
                vector_result_keys = []
                vector_result_keys_mask = []
                vector_kb_mask.append(1)
                # response = item['output'].split(" ")
                gold_entities = vectorized["gold_entities"][-1]

                sketch = modified_sketch_outs
                tag = modified_sketch_tags
                sketches = []
                for t in range(len(tag)):
                    if tag[t] != "null_tag":
                        sketches.append(sketch[t])

                for key in result:
                    # if beta_filter_found == 0 and result[key] in words_context:
                    if result[key] in words_context:
                        beta_filter_found += words_context[result[key]]

                    if self.args["dataset"] != 1:
                        if "sketch_" + str(key) in sketches:
                            if result[key] in gold_entities:
                                beta_found += 1
                            vector_select.append(1)
                        else:
                            vector_select.append(0)
                    if result[key] in self.vocab["vocab_mapping"]:
                        kb_key_types[key] = kb_key_types.get(key, []) + [
                            self.vocab["vocab_mapping"][result[key]]
                        ]
                        vector_result.append(self.vocab["vocab_mapping"][result[key]])
                        vector_copy_result.append(
                            self.vocab["vocab_mapping"][result[key]]
                        )
                    else:
                        kb_key_types[key] = kb_key_types.get(key, []) + [
                            self.vocab["vocab_mapping"]["$UNK$"]
                        ]
                        vector_result.append(self.vocab["vocab_mapping"]["$UNK$"])
                        vector_copy_result.append(self.vocab["unk_mapping"][result[key]])
                    if key in self.vocab["vocab_mapping"]:
                        vector_result_keys.append(self.vocab["vocab_mapping"][key])
                    else:
                        vector_result_keys.append(self.vocab["vocab_mapping"]["$UNK$"])

                    vector_result_keys_mask.append(1)

                for _ in range(0, keys_len - len(result.keys())):
                    if (self.args["dataset"] == 1):
                        vector_selectors[count].append(0)
                    else:
                        vector_select.append(0)
                    vector_result_keys.append(self.vocab["vocab_mapping"]["$PAD$"])
                    vector_result_keys_mask.append(0)
                    vector_result.append(self.vocab["vocab_mapping"]["$PAD$"])
                    vector_copy_result.append(self.vocab["vocab_mapping"]["$PAD$"])
                vector_keys.append(copy.copy(vector_result_keys))
                vector_keys_mask.append(copy.copy(vector_result_keys_mask))
                vector_kb.append(copy.copy(vector_result))
                vector_copy_kb.append(copy.copy(vector_copy_result))
                # vector_beta_filters.append(int(beta_filter_found > 0))
                vector_beta_filters.append(beta_filter_found)

                count += 1

                if self.args["dataset"] != 1:
                    vector_selectors.append(copy.copy(vector_select))
                    if beta_found > 0:
                        vector_beta_selectors.append(1)
                    else:
                        vector_beta_selectors.append(0)

            if item["kb"] == []:
                vector_kb_mask.append(0)
                vector_kb.append(
                    [self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0, keys_len)]
                )
                vector_copy_kb.append(
                    [self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0, keys_len)]
                )
                vector_keys.append(
                    [self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0, keys_len)]
                )
                vector_keys_mask.append([0] + [0 for _ in range(0, keys_len - 1)])
                vector_selectors.append([0 for _ in range(0, keys_len)])
                vector_beta_selectors.append(0)
                vector_beta_filters.append(0)

            current_kb_len = len(vector_kb_mask)

            if (len(item["kb"]) > 0) and sum(vector_beta_filters) == 0:
                vector_beta_filters = []
                for _ in range(current_kb_len):
                    vector_beta_filters.append(1)

            for _ in range(0, kb_len - current_kb_len):
                vector_kb_mask.append(0)
                vector_kb.append(
                    [self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0, keys_len)]
                )
                vector_copy_kb.append(
                    [self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0, keys_len)]
                )
                vector_keys.append(
                    [self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0, keys_len)]
                )
                vector_keys_mask.append([0] + [0 for _ in range(0, keys_len - 1)])
                vector_selectors.append([0 for _ in range(0, keys_len)])
                vector_beta_selectors.append(0)
                vector_beta_filters.append(0)

            kb_key_types = [list(set(kb_key_types[xx])) for xx in kb_key_types]
            for xx in range(len(kb_key_types)):
                for _ in range(len(kb_key_types[xx]), kb_len):
                    kb_key_types[xx].append(self.vocab["vocab_mapping"]["$PAD$"])

            for _ in range(len(kb_key_types), keys_len):
                temp = []
                for _ in range(kb_len):
                    temp.append(self.vocab["vocab_mapping"]["$PAD$"])
                kb_key_types.append(temp)
            # print('did', item['did'])
            # print('filter:', vector_beta_filters)
            # print('kb_mask:', vector_kb_mask)
            # print('beta_filter:', len(vector_beta_filters))
            # print('vector_selectors', [len(vc) for vc in vector_selectors])
            # for xx in range(len(vector_beta_filters)):
            # 	dom = -1
            # 	if vector_beta_selectors[xx] > 0:
            # 		self.global_count2[item['type']] = self.global_count2.get(item['type'], 0) + 1
            # 	if vector_beta_filters[xx] == 0 and vector_beta_selectors[xx] > 0:
            # 		self.global_count[item['type']] = self.global_count.get(item['type'], 0) + 1

            vectorized["type_kb_entities"].append(copy.deepcopy(kb_key_types))
            vectorized["kb"].append(copy.copy(vector_kb))
            vectorized["copy_kb_pos"].append(copy.copy(vector_copy_kb))
            vectorized["kb_mask"].append(copy.copy(vector_kb_mask))
            vectorized["keys"].append(copy.copy(vector_keys))
            vectorized["keys_mask"].append(copy.copy(vector_keys_mask))
            vectorized["key_selector"].append(copy.copy(vector_selectors))
            vectorized["beta_key_selector"].append(copy.copy(vector_beta_selectors))
            vectorized["beta_filter"].append(copy.copy(vector_beta_filters))

        # print('max_window_context_len:', max_window_context_len)
        # print('window_context_len:', vectorized['window_context_len'])
        # print('window_inp_utt', vectorized['window_inp_utt'])
        # print('window_inp_len', vectorized['window_inp_len'])
        # print('out_utt', vectorized['out_utt'])

        return vectorized

    def get_batch(self, data):

        epoch_done = False
        train = False
        if data == "trn":
            train = True

        if data == "trn":
            index = self.train_index
            batch = self.vectorize(
                self.train_data[index : index + self.batch_size], train
            )
            self.train_index = self.train_index + self.batch_size

            if self.train_index + self.batch_size > self.train_num:
                self.train_index = 0
                random.shuffle(self.train_data)
                epoch_done = True

        elif data == "val":
            index = self.val_index
            batch = self.vectorize(
                self.val_data_full[index : index + self.batch_size], train
            )
            self.val_index = self.val_index + self.batch_size

            if self.val_index + self.batch_size > self.val_num:
                self.val_index = 0
                # random.shuffle(self.val_data)
                # self.val_data_full = self.append_dummy_data(self.val_data)
                epoch_done = True
        else:
            index = self.test_index
            batch = self.vectorize(
                self.test_data_full[index : index + self.batch_size], train
            )
            self.test_index = self.test_index + self.batch_size

            if self.test_index + self.batch_size > self.test_num:
                self.test_index = 0
                # random.shuffle(self.test_data)
                # self.test_data_full = self.append_dummy_data(self.test_data)
                epoch_done = True

        return batch, epoch_done
