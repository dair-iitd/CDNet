import torch
import numpy as np 
from torch import nn
import os
from torch.nn import functional
from torch.autograd import Variable
# from memory_profiler import profile 
#torch.manual_seed(42)
torch.set_printoptions(profile="full")
# torch.autograd.set_detect_anomaly(True)

class MLMModel(nn.Module):
	def __init__(self, args, emb_init, generate_size, out_vocab_size, total_copy_vocab_size, eos):
		super(MLMModel, self).__init__()
		self.args = args
		if self.args['seed']:
			torch.manual_seed(self.args['seed'])
			print('Seed set to be:', self.args['seed'])   
		self.eos = int(eos)
		self.emb_dim = args['emb_dim']
		self.enc_hid_dim = args['enc_hid_dim']
		self.dec_hid_dim = args['dec_hid_dim']
		self.attn_size = args['attn_size']
		self.generate_size = generate_size
		self.out_vocab_size = out_vocab_size
		self.total_copy_vocab_size = total_copy_vocab_size
		pretrained_embeddings = torch.from_numpy(emb_init)
		self.dropout_layer = nn.Dropout(args['gru_drop'])
		self.embeddings = nn.Embedding(pretrained_embeddings.size(0), pretrained_embeddings.size(1), padding_idx = self.args['pad'])
		self.embeddings.weight.data.copy_(pretrained_embeddings)
		#self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(emb_init))
		
		self.encoder_1 = nn.GRU(input_size=args['emb_dim'], hidden_size=args['enc_hid_dim'], dropout=args["gru_drop"], batch_first=True, bidirectional=True)
		self.encoder_2 = nn.GRU(input_size=2*args['enc_hid_dim'], hidden_size=2*args['enc_hid_dim'], dropout=args["gru_drop"], batch_first=True)
		self.decoder_cell = nn.GRUCell(input_size=self.emb_dim, hidden_size=self.dec_hid_dim)
		self.lr = args['lr']
		self.batch = args['batch']
		# self.output_unit = self.create_output_unit()
		self.create_output_unit()
		self.device = args['device']

	# def scatter_nd_gather(self, base_matrix, indices, updates, accumulate):
	# 	for i in range(updates.shape[0]):
	# 		for j in range(updates.shape[1]):
	# 			base_matrix[indices[i,j,0],indices[i,j,1]] += updates[i,j]
	# 	# return base_matrix
	def sequence_mask_entropy(self, sequence_length, max_len=None):
		if max_len is None:
			max_len = sequence_length.data.max()
		batch_size = sequence_length.size(0)
		seq_range = torch.arange(0, max_len).long()
		seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
		seq_range_expand = Variable(seq_range_expand)
		if sequence_length.is_cuda:
			seq_range_expand = seq_range_expand.cuda()
		seq_length_expand = (sequence_length.unsqueeze(1)
							 .expand_as(seq_range_expand))
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
		if self.args['gpu']:
			length = Variable(torch.LongTensor(length)).to(self.device)
		else:
			length = Variable(torch.LongTensor(length))

		# logits_flat: (batch * max_len, num_classes)
		logits_flat = logits.contiguous().view(-1, logits.size(-1)) ## -1 means infered from other dimentions
		# log_probs_flat: (batch * max_len, num_classes)
		# log_probs_flat = functional.log_softmax(logits_flat, dim=1)
		logits_flat += 1e-20
		log_probs_flat = torch.log(logits_flat)
		# target_flat: (batch * max_len, 1)
		target_flat = target.view(-1, 1)
		# losses_flat: (batch * max_len, 1)
		losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
		# losses: (batch, max_len)
		losses = losses_flat.view(*target.size())
		# mask: (batch, max_len)
		mask_2 = self.sequence_mask_entropy(sequence_length=length, max_len=target.size(1)) 

		if mask is not None:
			mask_2 = mask_2 * mask.float() 

		losses = losses * mask_2.float()
		loss = losses.sum() / mask_2.float().sum()
		return loss

	def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32) -> torch.Tensor:
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
				result = torch.nn.functional.softmax(vector * mask, dim=dim)
				result = result * mask
				result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
			else:
				masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
				result = torch.nn.functional.softmax(masked_vector, dim=dim)
		return result

	# @profile
	def forward(self, inp_utt, copy_inp_utt, inp_mask, inp_len, context_len, kb, copy_kb, kb_mask, keys, keys_mask, db_empty, out_utt, copy_out_utt, sketch_tags, copy_sketch_tags, out_len, max_out_utt_len, sketch_mask, sketch_loss_mask, train=False, mode='joint'):
		batch_size = inp_utt.size()[0]
		inp_utt_emb_nd = self.embeddings(inp_utt)
		inp_utt_emb = self.dropout_layer(inp_utt_emb_nd)

		flat_inp_emb = torch.reshape(inp_utt_emb, shape=(-1,inp_utt.size()[2],self.emb_dim))
		flat_inp_len = torch.reshape(inp_len, shape=(-1,))
		flat_ctx_len = torch.reshape(context_len, shape=(-1,))

		flat_inp_len = flat_inp_len.clamp(min=1)
		flat_inp_emb_packed = nn.utils.rnn.pack_padded_sequence(flat_inp_emb, flat_inp_len, batch_first=True, enforce_sorted=False)
		outputs, output_states = self.encoder_1(flat_inp_emb_packed)
		flat_encoder_states, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

		# utt_reps = torch.reshape(output_states, shape=(output_states.size()[0],2*self.enc_hid_dim)) #--- Check the shape

		utt_rep_second = torch.cat((output_states[0], output_states[1]), dim=1)
		utt_rep_second = torch.reshape(utt_rep_second, shape=(batch_size, -1, 2*self.enc_hid_dim)) # Shape (batch, context, 2*hid_dim)
		
		utt_rep_second_packed = nn.utils.rnn.pack_padded_sequence(utt_rep_second, flat_ctx_len, batch_first=True, enforce_sorted=False)
		_, inp_utt_rep = self.encoder_2(utt_rep_second_packed)
		inp_utt_rep = torch.squeeze(inp_utt_rep, dim=0) #Squeeze along 0 to remove 1

		# inp_utt_rep = inp_utt_rep2 + utt_rep_second[:,-1,:].squeeze(1)

		encoder_states = torch.reshape(torch.reshape(flat_encoder_states, shape=(batch_size, -1, inp_utt.size()[2], 2*self.enc_hid_dim)), shape=(batch_size, -1, 2*self.enc_hid_dim))

		soft_selector = torch.ones(encoder_states.shape[0], encoder_states.shape[1])
		soft_row_selector = torch.ones(encoder_states.shape[0], utt_rep_second.shape[1])

		kb_emb_nd = self.embeddings(kb)
		kb_emb = self.dropout_layer(kb_emb_nd)
		keys_emb_nd = self.embeddings(keys)
		keys_emb = self.dropout_layer(keys_emb_nd)

		keys_mask += 1e-20
		result_rep = torch.einsum('ij,ijk->ijk', torch.pow(torch.sum(keys_mask, 2), -1), torch.sum(torch.einsum('ijk,ijkl->ijkl', keys_mask, kb_emb), 2))
		# print(result_rep)
		soft_beta_key_selector = torch.ones(result_rep.shape[0], result_rep.shape[1])
		soft_key_selector = torch.ones(keys_emb.shape[0], keys_emb.shape[1], keys_emb.shape[2])

		start_token = torch.LongTensor([0]*batch_size).to(self.device)
		out_utt_emb_nd = self.embeddings(out_utt)
		out_utt_emb = self.dropout_layer(out_utt_emb_nd)

		processed_x = torch.transpose(out_utt_emb, 0, 1)
		
		h0 = inp_utt_rep
		x_t_nd = self.embeddings(start_token)

		x_t = self.dropout_layer(x_t_nd)
		h_tm1 = h0
		g_predictions = []
		gen_x = []
		gen_sketch = []
		for i in range(0, max_out_utt_len):
			h_t = self.decoder_cell(x_t, h_tm1).type(torch.float32)
			s_t, o_t = self.unit(
					hidden_state = h_t,
					inp_utt = inp_utt, 
					copy_inp_utt = copy_inp_utt, 
					inp_len = inp_len,
					inp_mask = inp_mask,
					encoder_states = encoder_states, 
					utterance_states = utt_rep_second,
					kb = kb, 
					copy_kb = copy_kb, 
					db_empty = db_empty, 
					kb_mask = kb_mask, 
					result_rep = result_rep, 
					keys_emb = keys_emb, 
					keys_mask = keys_mask, 
					batch_size = batch_size, 
					out_vocab_size = self.out_vocab_size,
					total_copy_vocab_size = self.total_copy_vocab_size,
					soft_selector = soft_selector,
					soft_row_selector = soft_row_selector,
					soft_key_selector = soft_key_selector,
					soft_beta_key_selector = soft_beta_key_selector)
			# print("Batches: ", h_t.shape, s_t.shape, o_t.shape, x_t.shape)
			next_token = torch.argmax(o_t, dim=1)
			gen_x.append(next_token)
			
			if train:
				g_predictions.append(o_t)
				x_tp1 = processed_x[i]
			else:
				x_tp1 = self.embeddings(next_token)
				
			gen_sketch.append(s_t)
			x_t = x_tp1
			h_tm1 = h_t

		# torch.set_printoptions(profile="full")
		if train:
			g_predictions = torch.stack(g_predictions)
			g_predictions = torch.transpose(g_predictions, 0, 1)
		
		gen_x = torch.stack(gen_x)
		gen_x = torch.transpose(gen_x, 0, 1)

		# Defining Loss 
		if train:
			sentence_loss = self.masked_cross_entropy(g_predictions, out_utt, out_len)
			loss = sentence_loss
			return loss, loss, loss, loss, loss, gen_x, None						
		else:
			return gen_x, None

	def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
		if maxlen is None:
			maxlen = lengths.max()
		row_vector = torch.arange(0, maxlen, 1).to(self.device)
		matrix = torch.unsqueeze(lengths, dim=-1)
		mask = row_vector < matrix

		mask.type(dtype)
		return mask

	def create_output_unit(self):
		init_Wr = nn.init.xavier_uniform_(torch.empty(2*self.enc_hid_dim, 2*self.enc_hid_dim))
		self.Wr = nn.Parameter(init_Wr)
		init_Wo = nn.init.xavier_uniform_(torch.empty(2*self.enc_hid_dim, 2*self.enc_hid_dim))
		self.Wo = nn.Parameter(init_Wo)
		init_Wl = nn.init.xavier_uniform_(torch.empty(2*self.enc_hid_dim, self.dec_hid_dim))
		self.Wl = nn.Parameter(init_Wl)


		init_W1 = nn.init.xavier_uniform_(torch.empty(2*self.enc_hid_dim+self.dec_hid_dim, 2*self.enc_hid_dim))
		self.W1 = nn.Parameter(init_W1)
		init_W2 = nn.init.xavier_uniform_(torch.empty(2*self.enc_hid_dim, self.attn_size))
		self.W2 = nn.Parameter(init_W2)
		init_w = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
		self.w = nn.Parameter(init_w)
		init_U = nn.init.xavier_uniform_(torch.empty(self.dec_hid_dim+2*self.enc_hid_dim, self.generate_size))
		self.U = nn.Parameter(init_U)
		init_W_1 = nn.init.xavier_uniform_(torch.empty(self.emb_dim+self.dec_hid_dim+2*self.enc_hid_dim, 2*self.dec_hid_dim))
		self.W_1 = nn.Parameter(init_W_1)
		init_W_2 = nn.init.xavier_uniform_(torch.empty(self.emb_dim+self.dec_hid_dim+2*self.enc_hid_dim, 2*self.dec_hid_dim))
		self.W_2 = nn.Parameter(init_W_2)
		init_W_12 = nn.init.xavier_uniform_(torch.empty(2*self.dec_hid_dim, self.attn_size))
		self.W_12 = nn.Parameter(init_W_12)
		init_W_22 = nn.init.xavier_uniform_(torch.empty(2*self.dec_hid_dim, self.attn_size))
		self.W_22 = nn.Parameter(init_W_22)
		init_r_1 = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
		self.r_1 = nn.Parameter(init_r_1)
		init_r_2 = nn.init.xavier_uniform_(torch.empty(self.attn_size, 1))
		self.r_2 = nn.Parameter(init_r_2)
		# print(self.generate_size.shape)
		# init_b1 = nn.init.xavier_uniform_(torch.empty(self.generate_size))
		limit = np.sqrt(6/(self.generate_size+1))
		init_b1 = torch.nn.init.uniform_(torch.empty(self.generate_size), a=-1*limit, b=limit)
		# init_b1 = nn.init.constant_(torch.empty(self.generate_size), 0)
		self.b1 = nn.Parameter(init_b1)
		# init_b2 = nn.init.xavier_uniform_(torch.empty(1))
		limit_2 = np.sqrt(3)
		init_b2 = torch.nn.init.uniform_(torch.empty(1), a=-1*limit_2, b=limit_2)
		# init_b2 = nn.init.constant_(torch.empty(1), 0)
		self.b2 = nn.Parameter(init_b2)
		# init_b3 = nn.init.xavier_uniform_(torch.empty(1))
		limit_2 = np.sqrt(3)
		init_b3 = torch.nn.init.uniform_(torch.empty(1), a=-1*limit_2, b=limit_2)
		# init_b3 = nn.init.constant_(torch.empty(1), 0)
		self.b3 = nn.Parameter(init_b3)
		init_W3 = nn.init.xavier_uniform_(torch.empty(self.dec_hid_dim+2*self.enc_hid_dim+self.emb_dim, 1))
		self.W3 = nn.Parameter(init_W3)
		init_W4 = nn.init.xavier_uniform_(torch.empty(self.dec_hid_dim+2*self.enc_hid_dim+self.emb_dim, 1))
		self.W4 = nn.Parameter(init_W4)
	
	# @profile
	def unit(self, hidden_state, inp_utt, copy_inp_utt, inp_len, inp_mask, encoder_states, utterance_states, kb, copy_kb, db_empty, kb_mask, result_rep, keys_emb, keys_mask, batch_size, out_vocab_size, total_copy_vocab_size, soft_selector, soft_row_selector, soft_key_selector, soft_beta_key_selector):
		hidden_state_expanded_attn = hidden_state.unsqueeze(1).repeat(1, encoder_states.shape[1], 1)
		attn_rep = torch.cat((encoder_states, hidden_state_expanded_attn), dim=2)
		attn_rep = torch.tanh(torch.einsum('ijk,kl->ijl', torch.tanh(torch.einsum("ijk,kl->ijl", attn_rep, self.W1)), self.W2))
		u_i = torch.squeeze(torch.einsum('ijk,kl->ijl', attn_rep, self.w), 2)
		inp_len_mask = self.sequence_mask(inp_len, inp_utt.shape[2], torch.float32) #Modified
		attn_mask = torch.reshape(inp_len_mask, shape=(batch_size, -1))
		exp_u_i_masked = torch.mul(attn_mask.type(torch.DoubleTensor).to(self.device), torch.exp(u_i.type(torch.DoubleTensor).to(self.device)))

		# Calculation of aij
		a = torch.einsum('i,ij->ij', torch.pow(torch.sum(exp_u_i_masked, 1), -1), exp_u_i_masked).type(torch.FloatTensor).to(self.device)

		inp_attn = torch.sum(torch.einsum('ij,ijk->ijk', a, encoder_states), 1)
		
		generate_dist = nn.functional.softmax(torch.matmul(torch.cat((hidden_state, inp_attn), dim=1), self.U) + self.b1, dim=1)
		extra_zeros = torch.zeros([batch_size, total_copy_vocab_size - self.generate_size]).to(self.device)
		extended_generate_dist = torch.cat((generate_dist,extra_zeros), dim=1)

		hidden_state_expanded_result = torch.unsqueeze(hidden_state, 1).repeat(1, kb.shape[1], 1)
		inp_attn_expanded_result = torch.unsqueeze(inp_attn, 1).repeat(1, kb.shape[1], 1)
		result_attn_rep = torch.cat((result_rep.type(torch.FloatTensor).to(self.device), hidden_state_expanded_result, inp_attn_expanded_result), 2)
		result_attn_rep = torch.tanh(torch.einsum("ijk,kl->ijl", torch.tanh(torch.einsum("ijk,kl->ijl", result_attn_rep, self.W_1)), self.W_12))
		beta_logits = torch.squeeze(torch.einsum('ijk,kl->ijl', result_attn_rep, self.r_1), 2)
		beta_masked = torch.mul(kb_mask.type(torch.DoubleTensor).to(self.device), torch.exp(beta_logits.type(torch.DoubleTensor).to(self.device)))
		beta_masked += 1e-20
		beta = torch.einsum('i,ij->ij', torch.pow(torch.sum(beta_masked, 1), -1), beta_masked).type(torch.FloatTensor).to(self.device)

		hidden_state_expanded_keys = torch.unsqueeze(torch.unsqueeze(hidden_state, 1), 1).repeat(1, kb.shape[1], kb.shape[2], 1)
		inp_attn_expanded_keys = torch.unsqueeze(torch.unsqueeze(inp_attn, 1), 1).repeat(1, kb.shape[1], kb.shape[2], 1)
		result_key_rep = torch.cat((keys_emb, hidden_state_expanded_keys, inp_attn_expanded_keys), 3)
		result_key_rep = torch.tanh(torch.einsum('ijkl,lm->ijkm', torch.tanh(torch.einsum('ijkl,lm->ijkm', result_key_rep, self.W_2)), self.W_22))
		gaama_logits = torch.squeeze(torch.einsum('ijkl,lm->ijkm', result_key_rep, self.r_2), 3)
		gaama_masked = torch.mul(keys_mask.type(torch.DoubleTensor).to(self.device), torch.exp(gaama_logits.type(torch.DoubleTensor).to(self.device)))
		gaama_masked = gaama_masked + 1e-20
		gaama = torch.einsum('ij,ijk->ijk', beta, torch.einsum('ij,ijk->ijk', torch.pow(torch.sum(gaama_masked, 2), -1), gaama_masked)).type(torch.FloatTensor).to(self.device)

		batch_nums_context = torch.unsqueeze(torch.arange(0, batch_size, 1), 1)
		batch_nums_tiled_context = batch_nums_context.repeat(1, encoder_states.shape[1]).type(torch.LongTensor).to(self.device)
		flat_inp_utt = torch.reshape(copy_inp_utt, (batch_size, -1))
		indices_context = torch.stack([batch_nums_tiled_context, flat_inp_utt],dim=2)  #Modified
		shape = [batch_size, total_copy_vocab_size]
		
		context_copy_dist = torch.zeros(shape, dtype = torch.float32).to(self.device)
		context_copy_dist = context_copy_dist.index_put(tuple(indices_context.permute(2,0,1)), a, accumulate=True) #Modified

		db_rep = torch.sum(torch.einsum('ij,ijk->ijk', beta, result_rep), 1).type(torch.FloatTensor).to(self.device)

		p_db = torch.sigmoid(torch.matmul(torch.cat((hidden_state, inp_attn, db_rep), 1), self.W4) + self.b3)
		p_db = p_db.repeat(1, total_copy_vocab_size)
		one_minus_fn = lambda x: 1-x
		one_minus_pdb = 1 - p_db # Modified

		p_gens = torch.sigmoid(torch.matmul(torch.cat((hidden_state, inp_attn, db_rep), 1), self.W3) + self.b2)
		p_gens = p_gens.repeat(1, total_copy_vocab_size)
		one_minus_fn = lambda x: 1-x
		one_minus_pgens = 1-p_gens #Modified

		batch_nums = torch.unsqueeze(torch.arange(0, batch_size, 1), 1).to(self.device)
		kb_ids = torch.reshape(copy_kb, (batch_size, -1))
		num_kb_ids = kb_ids.shape[1]
		batch_nums_tiled = batch_nums.repeat(1, num_kb_ids).type(torch.LongTensor).to(self.device)
		indices = torch.stack([batch_nums_tiled, kb_ids],dim=2) #Modified
		updates = torch.reshape(gaama, (batch_size, -1))
		shape = (batch_size, total_copy_vocab_size)

		kb_dist = torch.zeros(shape, dtype = torch.float32).to(self.device)
		kb_dist = kb_dist.index_put(tuple(indices.permute(2,0,1)), updates, accumulate=True) #Modified
		kb_dist = torch.einsum('i,ij->ij', db_empty, kb_dist)
		copy_dist = torch.mul(p_db, kb_dist) + torch.mul(one_minus_pdb, context_copy_dist)
		final_dist = torch.mul(p_gens,extended_generate_dist) + torch.mul(one_minus_pgens,copy_dist)
		return None, final_dist

	def get_feed_dict(self,batch):
		out_utt = torch.tensor(batch['out_utt'], dtype=torch.long).to(self.device)
		copy_out_utt = torch.tensor(batch['copy_out_pos'], dtype=torch.long).to(self.device)
		
		fd = {
			'inp_mask' : torch.tensor(batch['inp_mask'], dtype=torch.float64).to(self.device),
			'inp_utt' : torch.tensor(batch['inp_utt'], dtype=torch.long).to(self.device),
			'copy_inp_pos' : torch.tensor(batch['copy_inp_pos'], dtype=torch.long).to(self.device),
			'inp_len' : torch.tensor(batch['inp_len'], dtype=torch.float64).to(self.device),
			'context_len' : torch.tensor(batch['context_len'], dtype=torch.float64).to(self.device),
			'out_utt' : out_utt,
			'copy_out_utt' : copy_out_utt,
			'sketch_tags' : torch.tensor(batch['sketch_tags'], dtype=torch.long).to(self.device),
			'copy_sketch_tag_pos' : torch.tensor(batch['copy_sketch_tag_pos'], dtype=torch.long).to(self.device),
			'sketch_mask' : torch.tensor(batch['sketch_mask'], dtype=torch.float64).to(self.device),
			'sketch_loss_mask' : torch.tensor(batch['sketch_loss_mask'], dtype=torch.float64).to(self.device),
			'out_len' : torch.tensor(batch['out_len'], dtype=torch.float64).to(self.device),
			'kb' : torch.tensor(batch['kb'], dtype=torch.long).to(self.device),
			'copy_kb_pos' : torch.tensor(batch['copy_kb_pos'], dtype=torch.long).to(self.device),
			'kb_mask' : torch.tensor(batch['kb_mask'], dtype=torch.float64).to(self.device),
			'keys' : torch.tensor(batch['keys'], dtype=torch.long).to(self.device),
			'keys_mask' : torch.tensor(batch['keys_mask'], dtype=torch.float64).to(self.device),
			'db_empty' : torch.tensor(batch['empty'], dtype=torch.float64).to(self.device),
			'max_out_utt_len' : torch.tensor(batch['max_out_utt_len']).to(self.device),
			'selector' : torch.tensor(batch['selector'], dtype=torch.float32).to(self.device),
			'key_selector' : torch.tensor(batch['key_selector'], dtype=torch.float32).to(self.device),
			'beta_key_selector' : torch.tensor(batch['beta_key_selector'], dtype=torch.float32).to(self.device),
			'row_selector': torch.tensor(batch['row_selector'], dtype=torch.float32).to(self.device)
		}
		
		return fd
