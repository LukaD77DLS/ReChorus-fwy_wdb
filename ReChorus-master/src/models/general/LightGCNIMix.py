# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class LightGCNBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			# D^-1/2 * A * D^-1/2
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)

	def forward(self, feed_dict):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)

		# print("u_embed:", u_embed.shape)
		# print(u_embed)
		# print("i_embed:", i_embed.shape)
		# print(i_embed)
		# print("user:", user.shape)
		# print(user)
		# print("items:", items.shape)
		# print(items)
		# print("pos_items:", items[:,0].shape)
		# print(items[0])
		# print("neg_items:", items[:,1:].shape)
		# print(items[1:])

		pos_items = items[:,0]
		neg_items = items[:,1:]

		# 对每个用户的物品对进行 IMix 操作
		new_u_embed, new_i_embed = self.imix_operation(u_embed, i_embed, user)

		prediction = (new_u_embed[:, None, :] * new_i_embed).sum(dim=-1)  # [batch_size, -1]
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
		# prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		# u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		# i_v = i_embed
		# return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

	def imix_operation(self, u_embed, i_embed, user):
         new_u_embed = []
         new_i_embed = []
         for i in range(len(user)):
            # 用户的嵌入向量
            e_user = u_embed[i]
            
            #正样品的嵌入向量
            e_pos_item = i_embed[i, 0]
            
            #负样品的嵌入向量序列
            e_neg_items = i_embed[i, 1:]
            
            # 随机选择一个负样品的嵌入向量
            num = i_embed.shape[1]
            index_i = np.random.choice(num-1)
            index_j = np.random.choice(num-1)
            while num > 1 and index_i == index_j:
                index_j = np.random.choice(num-1)
            
            #正样品分数
            score_pos = (e_user * e_pos_item).sum()
            
            #负样品分数
            score_neg_i = (e_user * e_neg_items[index_i]).sum()
            score_neg_j = (e_user * e_neg_items[index_j]).sum()
            
            #计算标签值
            yi = 1 if score_pos > score_neg_i else 0
            yj = 1 if score_pos > score_neg_j else 0
            
            lambda_ = np.random.beta(0.5, 0.5)  #计算lambda值
            if yi == yj:
                i_embed[i, index_i] = lambda_ * e_neg_items[index_i] + (1 - lambda_) * e_neg_items[index_j]
            else:
                i_embed[i, 0] = lambda_ * e_pos_item + (1 - lambda_) * e_neg_items[index_j]
                i_embed[i, index_i] = lambda_ * e_neg_items[index_i] + (1 - lambda_) * e_pos_item
            
            new_u_embed.append(u_embed[i])
            new_i_embed.append(i_embed[i])

         return torch.stack(new_u_embed), torch.stack(new_i_embed)

class LightGCNIMix(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}
	
class LightGCNImpression(ImpressionModel, LightGCNBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return LightGCNBase.forward(self, feed_dict)

class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def forward(self, users, items):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]

		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]

		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)

		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]

		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings
