import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import gc
import json
torch.set_printoptions(profile="full")
# outfile = open("out_smd_embeddings.csv", 'a')
gv = {}
# torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True)
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = True


class AggNet(nn.Module):
    def __init__(
        self,
        args,
        emb_init,
        generate_size,
        out_vocab_size,
        total_copy_vocab_size,
        eos,
        rev_vocab,
    ):
        super(AggNet, self).__init__()
        self.args = args
        if self.args["seed"]:
            torch.manual_seed(self.args["seed"])
            torch.cuda.manual_seed(args["seed"])
            np.random.seed(self.args["seed"])
            random.seed(self.args["seed"])
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            print("Seed set to be:", self.args["seed"])
        self.eos = int(eos)
        self.rev_vocab = rev_vocab
        self.lr = args["lr"]
        self.batch = args["batch"]
        self.device = args["device"]
        self.emb_dim = args["emb_dim"]
        self.enc_hid_dim = args["enc_hid_dim"]
        self.dec_hid_dim = args["dec_hid_dim"]
        self.attn_size = args["attn_size"]
        self.generate_size = generate_size
        self.out_vocab_size = out_vocab_size
        self.total_copy_vocab_size = total_copy_vocab_size
        pretrained_embeddings = torch.from_numpy(emb_init)
        self.dropout_layer = nn.Dropout(args["gru_drop"])
        self.embeddings = nn.Embedding(
            pretrained_embeddings.size(0),
            pretrained_embeddings.size(1),
            padding_idx=self.args["pad"],
        )
        self.embeddings.weight.data.copy_(pretrained_embeddings)

        self.encoder_1 = nn.GRU(
            input_size=args["emb_dim"],
            hidden_size=args["enc_hid_dim"],
            dropout=args["gru_drop"],
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_2 = nn.GRU(
            input_size=2 * args["enc_hid_dim"],
            hidden_size=2 * args["enc_hid_dim"],
            dropout=args["gru_drop"],
            batch_first=True,
        )
        self.decoder_cell = nn.GRUCell(
            input_size=self.emb_dim, hidden_size=self.dec_hid_dim
        )
        self.create_output_unit()

    def sequence_mask_entropy(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand

    def masked_cross_entropy(self, logits, target, length, mask=None):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.

        Returns:
            loss: An average loss value masked by the length.
        """
        length = length.type(torch.LongTensor)
        if self.args["gpu"]:
            length = Variable(torch.LongTensor(length)).to(self.device)
        else:
            length = Variable(torch.LongTensor(length))

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.contiguous().view(
            -1, logits.size(-1)
        )  # -1 means infered from other dimentions
        # log_probs_flat: (batch * max_len, num_classes)
        logits_flat = torch.where(
            logits_flat <= 1e-20,
            torch.tensor(1e-20).expand_as(logits_flat).to(self.device),
            logits_flat,
        )

        log_probs_flat = torch.log(logits_flat)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask_2 = self.sequence_mask_entropy(
            sequence_length=length, max_len=target.size(1)
        )

        if mask is not None:
            mask_2 = mask_2 * mask.float()

        losses = losses * mask_2.float()
        loss = losses.sum() / mask_2.float().sum()
        return loss

    def beta_masked_cross_entropy(self, logits, target, length_mask, mask=None):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, num_classes) which contains the
                normalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, num_classes) which contains the normalized
                probability of each class.
            length_mask: A Variable containing a LongTensor of size (batch,num_classes)
                which contains 1s at places of values and 0s at PADs.

        Returns:
            loss: An average loss value masked by the length.
        """
        # logits_flat: (batch, num_classes)
        logits_flat = logits.contiguous().view(
            -1, logits.size(-1)
        )  # -1 means infered from other dimentions

        logits_flat = torch.where(
            logits_flat <= 1e-20,
            torch.tensor(1e-20).expand_as(logits_flat).to(self.device),
            logits_flat,
        )

        # log_probs_flat: (batch, num_classes)
        log_probs_flat = torch.log(logits_flat)
        # target_flat: (batch, 1)
        losses1 = -torch.einsum("ij,ij->ij", target, log_probs_flat)

        losses = losses1
        # mask: (batch, num_classes)
        # Here mask represents what beta values are actual or where actually sketch is generated
        if mask is not None:
            length_mask = torch.einsum("i,ij->ij", mask, length_mask)
            # length_mask = length_mask * mask.float()

        losses = losses * length_mask.float()
        if length_mask.float().sum() > 0:
            loss = losses.sum() / length_mask.float().sum()
        else:
            loss = losses.sum()
        return loss

    def masked_softmax(
        self,
        vector: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
        memory_efficient: bool = False,
        mask_fill_value: float = -1e32,
    ) -> torch.Tensor:
        """
        ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
        masked positions so that the probabilities of those positions would be approximately 0.
        This is not accurate in math, but works for most cases and consumes less memory.
        In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
        returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
        a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
        will treat every element as equal, and do softmax over equal numbers.
        """
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = F.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill(
                    (1 - mask).type(torch.BoolTensor).to(self.device), mask_fill_value
                )
                result = F.softmax(masked_vector, dim=dim)
        return result

    def masked_sigmoid(self, vector: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            result = torch.sigmoid(vector)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)

            result = torch.sigmoid(vector)
            result = result * mask
        return result

    def masked_log_softmax(
        self,
        vector: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
        memory_efficient: bool = False,
        mask_fill_value: float = -1e32,
    ) -> torch.Tensor:
        """
        ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
        masked positions so that the probabilities of those positions would be approximately 0.
        This is not accurate in math, but works for most cases and consumes less memory.
        In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
        returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
        a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
        will treat every element as equal, and do softmax over equal numbers.
        """
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = F.log_softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill(
                    (1 - mask).type(torch.BoolTensor).to(self.device), mask_fill_value
                )
                result = F.log_softmax(masked_vector, dim=dim)
        return result

    def profile_gpu(self):
        logs = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    logs.append(obj)
            except Exception as e:
                print(e)
                pass
        print("Total GPU Tensors: ", len(logs))

    # @profile
    def forward(
        self,
        did,
        window_inp_utt,
        window_inp_len,
        window_context_len,
        inp_utt,
        copy_inp_utt,
        inp_mask,
        inp_len,
        context_len,
        kb,
        copy_kb,
        kb_mask,
        keys,
        keys_mask,
        db_empty,
        out_utt,
        copy_out_utt,
        sketch_tags,
        copy_sketch_tags,
        out_len,
        max_out_utt_len,
        sketch_mask,
        sketch_loss_mask,
        beta_filter,
        type_kb_entities,
        train=False,
        mode=[1, 1, 1, 1],
        debug_stuff=None
    ):
        # self.profile_gpu()
        batch_size = inp_utt.size()[0]
        inp_utt_emb_nd = self.embeddings(inp_utt)
        inp_utt_emb = self.dropout_layer(inp_utt_emb_nd)

        flat_inp_emb = torch.reshape(
            inp_utt_emb, shape=(-1, inp_utt.size()[2], self.emb_dim)
        )
        flat_inp_len = torch.reshape(inp_len, shape=(-1,))
        flat_ctx_len = torch.reshape(context_len, shape=(-1,))

        flat_inp_len = flat_inp_len.clamp(min=1)
        flat_inp_emb_packed = nn.utils.rnn.pack_padded_sequence(
            flat_inp_emb, flat_inp_len, batch_first=True, enforce_sorted=False
        )
        outputs, output_states = self.encoder_1(flat_inp_emb_packed)
        flat_encoder_states, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )

        utt_rep_second = torch.cat((output_states[0], output_states[1]), dim=1)
        utt_rep_second = torch.reshape(
            utt_rep_second, shape=(batch_size, -1, 2 * self.enc_hid_dim)
        )  # Shape (batch, context, 2*hid_dim)

        utt_rep_second_packed = nn.utils.rnn.pack_padded_sequence(
            utt_rep_second, flat_ctx_len, batch_first=True, enforce_sorted=False
        )
        _, inp_utt_rep = self.encoder_2(utt_rep_second_packed)
        inp_utt_rep = (
            torch.squeeze(inp_utt_rep, dim=0)
            .type(torch.FloatTensor)
            .to(self.device)
        )  # Squeeze along 0 to remove 1

        encoder_states = torch.reshape(
            torch.reshape(
                flat_encoder_states,
                shape=(batch_size, -1, inp_utt.size()[2], 2 * self.enc_hid_dim),
            ),
            shape=(batch_size, -1, 2 * self.enc_hid_dim),
        )

        kb_emb_nd = self.embeddings(kb)
        kb_emb = self.dropout_layer(kb_emb_nd)
        keys_emb_nd = self.embeddings(keys)
        keys_emb = self.dropout_layer(keys_emb_nd)

        denom_keys_mask = torch.sum(keys_mask, 2)
        denom_keys_mask = torch.where(
            denom_keys_mask <= 1e-20,
            torch.tensor(1e-20).expand_as(denom_keys_mask).to(self.device),
            denom_keys_mask,
        )
        result_rep = torch.div(
            torch.sum(torch.einsum("ijk,ijkl->ijkl", keys_mask, kb_emb), 2),
            denom_keys_mask.unsqueeze(2).expand(-1, -1, kb_emb.shape[3]),
        )

        if self.args["abl_global_beta"]:
            # Similarity score global beta
            if self.args["abl_window"] > 0:
                cos = torch.nn.CosineSimilarity(dim=4, eps=1e-20)
                window_inp_utt_emb_nd = self.embeddings(window_inp_utt)
                window_inp_utt_emb = self.dropout_layer(window_inp_utt_emb_nd)

                window_seq_mask = self.sequence_mask(
                    window_inp_len, window_inp_utt.shape[2], torch.float32
                )
                window_inp_len = torch.where(
                    window_inp_len <= 1e-20,
                    torch.tensor(1e-20).expand_as(window_inp_len).to(self.device),
                    window_inp_len,
                )
                if self.args["abl_window"] > 1:
                    total_window_inp_emb = torch.div(
                        torch.sum(
                            torch.einsum(
                                "ijk,ijkl->ijkl", window_seq_mask, window_inp_utt_emb
                            ),
                            dim=2,
                        ),
                        window_inp_len.unsqueeze(2).expand(
                            -1, -1, window_inp_utt_emb.shape[3]
                        ),
                    )
                else:
                    total_window_inp_emb = torch.sum(
                        torch.einsum(
                            "ijk,ijkl->ijkl", window_seq_mask, window_inp_utt_emb
                        ),
                        dim=2,
                    )

                final_window_utt_emb = (
                    total_window_inp_emb.unsqueeze(1)
                    .expand(-1, keys_mask.shape[1], -1, -1)
                    .unsqueeze(2)
                    .expand(-1, -1, keys_mask.shape[2], -1, -1)
                )
                total_kb_emb = torch.einsum("ijk,ijkl->ijkl", keys_mask, kb_emb)
                final_kb_emb = total_kb_emb.unsqueeze(3).expand(
                    -1, -1, -1, total_window_inp_emb.shape[1], -1
                )
                similarity_score = cos(
                    final_window_utt_emb, final_kb_emb
                )  # (batch,#result,#entities,#num_window)

                total_similarity_score = torch.sum(
                    torch.sum(similarity_score, dim=3), dim=2
                )

                masked_total_similarity_score = torch.einsum(
                    "ij,ij->ij", kb_mask, total_similarity_score
                )

                global_beta = self.masked_softmax(vector=masked_total_similarity_score.type(torch.FloatTensor).to(self.device), mask=kb_mask.type(torch.BoolTensor).to(self.device), dim=1, memory_efficient=True)
                global_pred = global_beta
            else:
                # Calculate KB row attention or Beta attention
                inp_attn_expanded_result = torch.unsqueeze(inp_utt_rep, 1).expand(
                    -1, kb.shape[1], -1
                )
                result_attn_rep = torch.cat(
                    (
                        result_rep.type(torch.FloatTensor).to(self.device),
                        inp_attn_expanded_result,
                    ),
                    2,
                )
                result_attn_rep = torch.tanh(
                    torch.einsum(
                        "ijk,kl->ijl",
                        torch.tanh(
                            torch.einsum("ijk,kl->ijl", result_attn_rep, self.W_1)
                        ),
                        self.W_12,
                    )
                )
                beta_logits_naive = torch.squeeze(
                    torch.einsum("ijk,kl->ijl", result_attn_rep, self.r_1), 2
                )

                global_beta = self.masked_softmax(vector=beta_logits_naive.type(torch.FloatTensor).to(self.device), mask=kb_mask.type(torch.BoolTensor).to(self.device), dim=1, memory_efficient=True)
                global_pred = global_beta
        else:
            global_beta = None

        start_token = torch.LongTensor([0] * batch_size).to(self.device)
        processed_x = torch.transpose(out_utt, 0, 1)

        h0 = inp_utt_rep
        x_t_nd = self.embeddings(start_token)

        x_t = self.dropout_layer(x_t_nd)
        h_tm1 = h0
        g_predictions = []
        gen_x = []
        gen_sketch = []
        loss_beta_supervision = (
            torch.zeros(1).type(torch.FloatTensor).to(self.device)
        )

        for i in range(0, max_out_utt_len):
            h_t = self.decoder_cell(x_t, h_tm1).type(torch.float32)
            s_t, o_t, _ = self.unit(
                did=did,
                hidden_state=h_t,
                inp_utt=inp_utt,
                copy_inp_utt=copy_inp_utt,
                inp_len=inp_len,
                inp_mask=inp_mask,
                encoder_states=encoder_states,
                utterance_states=utt_rep_second,
                kb=kb,
                copy_kb=copy_kb,
                db_empty=db_empty,
                kb_mask=kb_mask,
                result_rep=result_rep,
                keys_emb=keys_emb,
                keys_mask=keys_mask,
                batch_size=batch_size,
                out_vocab_size=self.out_vocab_size,
                total_copy_vocab_size=self.total_copy_vocab_size,
                beta_filter=beta_filter,
                global_beta=global_beta,
            )
            next_token = torch.argmax(o_t, dim=1)
            gen_x.append(next_token)

            if train:
                g_predictions.append(o_t)
                teacher_force = random.random()
                if teacher_force > self.args["teacher_forcing"]:
                    mask = torch.tensor(
                        [
                            int("sketch_" in self.rev_vocab[str(int(word.item()))])
                            for word in next_token
                        ]
                    ).to(self.device)
                    x_tp1 = self.dropout_layer(self.embeddings(next_token))
                else:
                    mask = torch.tensor(
                        [
                            int("sketch_" in self.rev_vocab[str(int(word.item()))])
                            for word in processed_x[i]
                        ]
                    ).to(self.device)
                    x_tp1 = self.dropout_layer(self.embeddings(processed_x[i]))
            else:
                mask = torch.tensor(
                    [
                        int("sketch_" in self.rev_vocab[str(int(word.item()))])
                        for word in next_token
                    ]
                ).to(self.device)
                x_tp1 = self.embeddings(next_token)

            gen_sketch.append(s_t)
            x_t = x_tp1
            h_tm1 = h_t

        if train:
            g_predictions = torch.stack(g_predictions)
            g_predictions = torch.transpose(g_predictions, 0, 1)

        gen_x = torch.stack(gen_x)
        gen_x = torch.transpose(gen_x, 0, 1)

        gen_sketch = torch.stack(gen_sketch)
        gen_sketch = torch.transpose(gen_sketch, 0, 1)

        # Defining Loss
        if train:
            sentence_loss = self.masked_cross_entropy(g_predictions, out_utt, out_len)
            if self.args["abl_similarity_loss"] and self.args["abl_global_beta"]:
                kb_type_mask = keys_mask.transpose(1, 2).unsqueeze(3)
                similarity_mask = torch.einsum(
                    "ijkl,ijlm->ijkm", kb_type_mask, kb_type_mask.transpose(2, 3)
                )
                type_kb_entities = self.embeddings(type_kb_entities)
                type_kb_entities1 = type_kb_entities.unsqueeze(3).expand(
                    -1, -1, -1, type_kb_entities.shape[2], -1
                )
                type_kb_entities2 = type_kb_entities.unsqueeze(2).expand(
                    -1, -1, type_kb_entities.shape[2], -1, -1
                )
                similarity_matrix = cos(type_kb_entities1, type_kb_entities2)
                similarity_matrix = similarity_matrix * similarity_mask
                final_similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
                sum_similarity = torch.sum(final_similarity_matrix)
                sum_mask = torch.sum(torch.triu(similarity_mask, diagonal=1))
                if sum_mask > 0:
                    loss_similarity = sum_similarity / sum_mask
                else:
                    loss_similarity = torch.tensor(0.0)
            else:
                loss_similarity = (
                    torch.zeros(1).type(torch.FloatTensor).to(self.device)
                )

            if self.args["abl_beta_supvis"] and self.args["abl_global_beta"]:
                gold_denominator = torch.sum(beta_filter, 1)
                gold_denominator = torch.where(
                    gold_denominator <= 1e-20,
                    torch.tensor(1e-20).expand_as(gold_denominator).to(self.device),
                    gold_denominator,
                )
                gold_global_beta = torch.div(
                    beta_filter,
                    gold_denominator.unsqueeze(1).expand(-1, beta_filter.shape[1]),
                )

                loss_beta_supervision = self.beta_masked_cross_entropy(
                    global_pred, gold_global_beta, kb_mask, None
                )
            else:
                loss_beta_supervision = (
                    torch.zeros(1).type(torch.FloatTensor).to(self.device)
                )

            copy_loss = self.masked_cross_entropy(
                gen_sketch, copy_sketch_tags, out_len, sketch_mask
            )

            total_loss = (
                mode[0] * sentence_loss.type(torch.FloatTensor).to(self.device)
                + mode[1] * copy_loss.type(torch.FloatTensor).to(self.device)
                + mode[2] * loss_beta_supervision.type(torch.FloatTensor).to(self.device)
                + mode[3] * loss_similarity.type(torch.FloatTensor).to(self.device)
            )
            loss = total_loss

            return (
                loss,
                mode[1] * copy_loss,
                mode[0] * sentence_loss,
                mode[2] * loss_beta_supervision,
                mode[3] * loss_similarity,
                gen_x,
                gen_sketch,
            )
        else:
            gb1 = None
            gb2 = None
            return gen_x, gen_sketch, global_beta, gb1, gb2

    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.device)
        matrix = torch.unsqueeze(lengths, dim=-1).type(torch.LongTensor).to(self.device)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def create_output_unit(self):
        init_Wr = nn.init.xavier_uniform_(
            torch.empty(2 * self.enc_hid_dim, 2 * self.enc_hid_dim)
        )
        self.Wr = nn.Parameter(init_Wr)
        init_Wo = nn.init.xavier_uniform_(
            torch.empty(2 * self.enc_hid_dim, 2 * self.enc_hid_dim)
        )
        self.Wo = nn.Parameter(init_Wo)
        init_Wl = nn.init.xavier_uniform_(
            torch.empty(2 * self.enc_hid_dim, self.dec_hid_dim)
        )
        self.Wl = nn.Parameter(init_Wl)

        if self.args["hops"] > 1:
            self.W1 = []
            self.W2 = []
            self.w = []

            # local beta
            for i in range(self.args["hops"]):
                init_W1 = nn.init.xavier_uniform_(
                    torch.empty(
                        2 * self.enc_hid_dim + self.dec_hid_dim, 2 * self.enc_hid_dim
                    )
                )
                self.W1.append(nn.Parameter(init_W1).to(self.device))
                init_W2 = nn.init.xavier_uniform_(
                    torch.empty(2 * self.enc_hid_dim, self.attn_size)
                )
                self.W2.append(nn.Parameter(init_W2).to(self.device))
                init_w = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
                self.w.append(nn.Parameter(init_w).to(self.device))
        else:
            init_W1 = nn.init.xavier_uniform_(
                torch.empty(
                    2 * self.enc_hid_dim + self.dec_hid_dim, 2 * self.enc_hid_dim
                )
            )
            self.W1 = nn.Parameter(init_W1)
            init_W2 = nn.init.xavier_uniform_(
                torch.empty(2 * self.enc_hid_dim, self.attn_size)
            )
            self.W2 = nn.Parameter(init_W2)
            init_w = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
            self.w = nn.Parameter(init_w)

        # Local Beta
        init_W_lb1 = nn.init.xavier_uniform_(
            torch.empty(
                self.emb_dim + self.dec_hid_dim + 2 * self.enc_hid_dim,
                2 * self.dec_hid_dim,
            )
        )
        self.W_lb1 = nn.Parameter(init_W_lb1)
        init_W_lb12 = nn.init.xavier_uniform_(
            torch.empty(2 * self.dec_hid_dim, self.attn_size)
        )
        self.W_lb12 = nn.Parameter(init_W_lb12)
        init_r_lb1 = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
        self.r_lb1 = nn.Parameter(init_r_lb1)

        init_W_d = nn.init.xavier_uniform_(
            torch.empty(
                self.emb_dim + self.dec_hid_dim + 2 * self.enc_hid_dim,
                2 * self.dec_hid_dim,
            )
        )
        self.W_d = nn.Parameter(init_W_d)
        init_W_dd = nn.init.xavier_uniform_(
            torch.empty(2 * self.dec_hid_dim, self.attn_size)
        )
        self.W_dd = nn.Parameter(init_W_dd)

        init_W_d2 = nn.init.xavier_uniform_(
            torch.empty(2 * self.enc_hid_dim, self.emb_dim)
        )
        self.W_d2 = nn.Parameter(init_W_d2)

        # Global Beta
        init_W_1 = nn.init.xavier_uniform_(
            torch.empty(self.emb_dim + 2 * self.enc_hid_dim, 2 * self.enc_hid_dim)
        )
        self.W_1 = nn.Parameter(init_W_1)
        init_W_12 = nn.init.xavier_uniform_(
            torch.empty(2 * self.enc_hid_dim, self.attn_size)
        )
        self.W_12 = nn.Parameter(init_W_12)
        init_r_1 = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
        self.r_1 = nn.Parameter(init_r_1)

        init_U = nn.init.xavier_uniform_(
            torch.empty(self.dec_hid_dim + 2 * self.enc_hid_dim, self.generate_size)
        )
        self.U = nn.Parameter(init_U)
        init_W_2 = nn.init.xavier_uniform_(
            torch.empty(
                self.emb_dim + self.dec_hid_dim + 2 * self.enc_hid_dim,
                2 * self.dec_hid_dim,
            )
        )
        self.W_2 = nn.Parameter(init_W_2)
        init_W_22 = nn.init.xavier_uniform_(
            torch.empty(2 * self.dec_hid_dim, self.attn_size)
        )
        self.W_22 = nn.Parameter(init_W_22)
        init_r_2 = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
        self.r_2 = nn.Parameter(init_r_2)

        limit = np.sqrt(6 / (self.generate_size + 1))
        init_b1 = torch.nn.init.uniform_(
            torch.empty(self.generate_size), a=-1 * limit, b=limit
        )
        self.b1 = nn.Parameter(init_b1)

        limit_2 = np.sqrt(3)
        init_b2 = torch.nn.init.uniform_(torch.empty(1), a=-1 * limit_2, b=limit_2)
        self.b2 = nn.Parameter(init_b2)

        limit_2 = np.sqrt(3)
        init_b3 = torch.nn.init.uniform_(torch.empty(1), a=-1 * limit_2, b=limit_2)
        self.b3 = nn.Parameter(init_b3)

        init_W3 = nn.init.xavier_uniform_(
            torch.empty(self.dec_hid_dim + 2 * self.enc_hid_dim + self.emb_dim, 1)
        )
        self.W3 = nn.Parameter(init_W3)
        init_W4 = nn.init.xavier_uniform_(
            torch.empty(self.dec_hid_dim + 2 * self.enc_hid_dim + self.emb_dim, 1)
        )
        self.W4 = nn.Parameter(init_W4)

    def unit(
        self,
        did,
        hidden_state,
        inp_utt,
        copy_inp_utt,
        inp_len,
        inp_mask,
        encoder_states,
        utterance_states,
        kb,
        copy_kb,
        db_empty,
        kb_mask,
        result_rep,
        keys_emb,
        keys_mask,
        batch_size,
        out_vocab_size,
        total_copy_vocab_size,
        beta_filter,
        global_beta,
    ):
        queries_decode = [hidden_state]
        query = hidden_state
        inp_len_mask = self.sequence_mask(
            inp_len, inp_utt.shape[2], torch.float32
        )  # Modified
        attn_mask = torch.reshape(inp_len_mask, shape=(batch_size, -1))
        for hop in range(self.args["hops"]):
            query = queries_decode[-1]
            hidden_state_expanded_attn = query.unsqueeze(1).expand(
                -1, encoder_states.shape[1], -1
            )
            attn_rep1 = torch.cat(
                (encoder_states, hidden_state_expanded_attn), dim=2
            )
            if self.args['hops'] > 1:            
                attn_rep = torch.tanh(
                    torch.einsum(
                        "ijk,kl->ijl",
                        torch.tanh(
                            torch.einsum("ijk,kl->ijl", attn_rep1, self.W1[hop])
                        ),
                        self.W2[hop],
                    )
                )
                u_i = torch.squeeze(
                    torch.einsum("ijk,kl->ijl", attn_rep, self.w[hop]), 2
                )
            else:
                attn_rep = torch.tanh(
                    torch.einsum(
                        "ijk,kl->ijl",
                        torch.tanh(torch.einsum("ijk,kl->ijl", attn_rep1, self.W1)),
                        self.W2,
                    )
                )
                u_i = torch.squeeze(torch.einsum("ijk,kl->ijl", attn_rep, self.w), 2)
            
            exp_u_i_masked = torch.mul(
                attn_mask.type(torch.FloatTensor).to(self.device),
                torch.exp(u_i.type(torch.FloatTensor).to(self.device)),
            )
            # Calculation of aij
            a = (
                torch.einsum(
                    "i,ij->ij",
                    torch.pow(torch.sum(exp_u_i_masked, 1), -1),
                    exp_u_i_masked,
                )
                .type(torch.FloatTensor)
                .to(self.device)
            )

            inp_attn = torch.sum(torch.einsum("ij,ijk->ijk", a, encoder_states), 1)
            if hop == 0:
                original_inp_attn = inp_attn
            queries_decode.append(query + inp_attn)

        generate_dist = nn.functional.softmax(
            torch.matmul(
                torch.cat((hidden_state, original_inp_attn), dim=1), self.U
            )
            + self.b1,
            dim=1,
        )

        final_inp_attn = inp_attn

        hidden_state_expanded_keys = torch.unsqueeze(
            torch.unsqueeze(hidden_state, 1), 1
        ).expand(-1, kb.shape[1], kb.shape[2], -1)
        inp_attn_expanded_keys = torch.unsqueeze(
            torch.unsqueeze(final_inp_attn, 1), 1
        ).expand(-1, kb.shape[1], kb.shape[2], -1)

        hidden_state_expanded_result = torch.unsqueeze(hidden_state, 1).expand(
            -1, kb.shape[1], -1
        )
        inp_attn_expanded_result = torch.unsqueeze(final_inp_attn, 1).expand(
            -1, kb.shape[1], -1
        )
        result_attn_rep = torch.cat(
            (
                result_rep.type(torch.FloatTensor).to(self.device),
                hidden_state_expanded_result,
                inp_attn_expanded_result,
            ),
            2,
        )

        result_attn_rep = torch.tanh(
            torch.einsum(
                "ijk,kl->ijl",
                torch.tanh(torch.einsum("ijk,kl->ijl", result_attn_rep, self.W_lb1)),
                self.W_lb12,
            )
        )
        beta_logits = torch.squeeze(
            torch.einsum("ijk,kl->ijl", result_attn_rep, self.r_lb1), 2
        )

        local_beta = self.masked_softmax(
            vector=beta_logits.type(torch.FloatTensor).to(self.device),
            mask=kb_mask.type(torch.BoolTensor).to(self.device),
            dim=1,
            memory_efficient=True,
        )

        local_beta = global_beta * local_beta

        result_key_rep = torch.cat(
            (keys_emb, hidden_state_expanded_keys, inp_attn_expanded_keys), 3
        )
        result_key_rep = torch.tanh(
            torch.einsum(
                "ijkl,lm->ijkm",
                torch.tanh(torch.einsum("ijkl,lm->ijkm", result_key_rep, self.W_2)),
                self.W_22,
            )
        )
        gaama_logits = torch.squeeze(
            torch.einsum("ijkl,lm->ijkm", result_key_rep, self.r_2), 3
        )
        gaama_masked = torch.mul(
            keys_mask.type(torch.FloatTensor).to(self.device),
            torch.exp(gaama_logits.type(torch.FloatTensor).to(self.device)),
        )

        denom_gaama_masked = torch.sum(gaama_masked, 2)
        denom_gaama_masked = torch.where(
            denom_gaama_masked <= 1e-20,
            torch.tensor(1e-20).expand_as(denom_gaama_masked).to(self.device),
            denom_gaama_masked,
        )
        gamma2 = torch.div(
            gaama_masked,
            denom_gaama_masked.unsqueeze(2).expand(-1, -1, gaama_masked.shape[2]),
        )

        gaama = (
            torch.einsum("ij,ijk->ijk", local_beta, gamma2)
            .type(torch.FloatTensor)
            .to(self.device)
        )

        batch_nums_context = torch.unsqueeze(torch.arange(0, batch_size, 1), 1)
        batch_nums_tiled_context = (
            batch_nums_context.repeat(1, encoder_states.shape[1])
            .type(torch.LongTensor)
            .to(self.device)
        )
        flat_inp_utt = torch.reshape(copy_inp_utt, (batch_size, -1))
        indices_context = torch.stack(
            [batch_nums_tiled_context, flat_inp_utt], dim=2
        )  # Modified
        shape = [batch_size, total_copy_vocab_size]

        context_copy_dist = torch.zeros(shape, dtype=torch.float32).to(self.device)
        context_copy_dist = context_copy_dist.index_put(
            tuple(indices_context.permute(2, 0, 1)), a, accumulate=True
        )  # Modified

        db_rep = (
            torch.sum(torch.einsum("ij,ijk->ijk", local_beta, result_rep), 1)
            .type(torch.FloatTensor)
            .to(self.device)
        )

        p_db = torch.sigmoid(
            torch.matmul(torch.cat((hidden_state, final_inp_attn, db_rep), 1), self.W4)
            + self.b3
        )
        p_db = p_db.repeat(1, total_copy_vocab_size)
        one_minus_pdb = 1 - p_db  # Modified

        p_gens = torch.sigmoid(
            torch.matmul(torch.cat((hidden_state, final_inp_attn, db_rep), 1), self.W3)
            + self.b2
        )
        p_gens = p_gens.repeat(1, total_copy_vocab_size)
        one_minus_pgens = 1 - p_gens  # Modified

        batch_nums = torch.unsqueeze(torch.arange(0, batch_size, 1), 1).to(self.device)
        kb_ids = torch.reshape(copy_kb, (batch_size, -1))
        num_kb_ids = kb_ids.shape[1]
        batch_nums_tiled = (
            batch_nums.repeat(1, num_kb_ids).type(torch.LongTensor).to(self.device)
        )
        indices = torch.stack([batch_nums_tiled, kb_ids], dim=2)  # Modified
        updates = torch.reshape(gaama, (batch_size, -1))
        shape = (batch_size, total_copy_vocab_size)

        kb_dist = torch.zeros(shape, dtype=torch.float32).to(self.device)
        kb_dist = kb_dist.index_put(
            tuple(indices.permute(2, 0, 1)), updates, accumulate=True
        )  # Modified
        kb_dist = torch.einsum("i,ij->ij", db_empty, kb_dist)
        copy_dist = torch.mul(p_db, kb_dist) + torch.mul(
            one_minus_pdb, context_copy_dist
        )

        return copy_dist, generate_dist, None


    def get_feed_dict(self, batch):
        out_utt = torch.tensor(batch["sketch_outs"], dtype=torch.long).to(
            self.device
        )
        copy_out_utt = torch.tensor(
            batch["copy_sketch_out_pos"], dtype=torch.long
        ).to(self.device)

        fd = {
            "inp_mask": torch.tensor(batch["inp_mask"], dtype=torch.float32).to(
                self.device
            ),
            "copy_inp_pos": torch.tensor(batch["copy_inp_pos"], dtype=torch.long).to(
                self.device
            ),
            "inp_utt": torch.tensor(batch["inp_utt"], dtype=torch.long).to(self.device),
            "inp_len": torch.tensor(batch["inp_len"], dtype=torch.float32).to(
                self.device
            ),
            "context_len": torch.tensor(batch["context_len"], dtype=torch.float32).to(
                self.device
            ),
            "window_inp_utt": torch.tensor(
                batch["window_inp_utt"], dtype=torch.long
            ).to(self.device),
            "window_inp_len": torch.tensor(
                batch["window_inp_len"], dtype=torch.float32
            ).to(self.device),
            "window_context_len": torch.tensor(
                batch["window_context_len"], dtype=torch.float32
            ).to(self.device),
            "out_utt": out_utt,
            "copy_out_utt": copy_out_utt,
            "sketch_tags": torch.tensor(batch["sketch_tags"], dtype=torch.long).to(
                self.device
            ),
            "copy_sketch_tag_pos": torch.tensor(
                batch["copy_sketch_tag_pos"], dtype=torch.long
            ).to(self.device),
            "sketch_mask": torch.tensor(batch["sketch_mask"], dtype=torch.float32).to(
                self.device
            ),
            "sketch_loss_mask": torch.tensor(
                batch["sketch_loss_mask"], dtype=torch.float32
            ).to(self.device),
            "out_len": torch.tensor(batch["out_len"], dtype=torch.float32).to(
                self.device
            ),
            "type_kb_entities": torch.tensor(
                batch["type_kb_entities"], dtype=torch.long
            ).to(self.device),
            "kb": torch.tensor(batch["kb"], dtype=torch.long).to(self.device),
            "copy_kb_pos": torch.tensor(batch["copy_kb_pos"], dtype=torch.long).to(
                self.device
            ),
            "kb_mask": torch.tensor(batch["kb_mask"], dtype=torch.float32).to(
                self.device
            ),
            "keys": torch.tensor(batch["keys"], dtype=torch.long).to(self.device),
            "keys_mask": torch.tensor(batch["keys_mask"], dtype=torch.float32).to(
                self.device
            ),
            "db_empty": torch.tensor(batch["empty"], dtype=torch.float32).to(
                self.device
            ),
            "max_out_utt_len": torch.tensor(batch["max_out_utt_len"]).to(self.device),
            "selector": torch.tensor(batch["selector"], dtype=torch.float32).to(
                self.device
            ),
            "key_selector": torch.tensor(batch["key_selector"], dtype=torch.float32).to(
                self.device
            ),
            "beta_key_selector": torch.tensor(
                batch["beta_key_selector"], dtype=torch.float32
            ).to(self.device),
            "beta_filter": torch.tensor(batch["beta_filter"], dtype=torch.float32).to(
                self.device
            ),
            "row_selector": torch.tensor(batch["row_selector"], dtype=torch.float32).to(
                self.device
            ),
            "did": torch.tensor(batch["did"], dtype=torch.float32).to(self.device),
        }

        return fd
