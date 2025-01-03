# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BPRMF
Reference:
	"Bayesian personalized ranking from implicit feedback"
	Rendle et al., UAI'2009.
CMD example:
	python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""
import torch
import numpy as np
import torch.nn as nn

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class BPRMFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self._base_define_params()
		self.apply(self.init_weights)
	
	def _base_define_params(self):	
		self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

	def forward(self, feed_dict):
		self.check_list = []
		u_ids = feed_dict['user_id']  # [batch_size]
		i_ids = feed_dict['item_id']  # [batch_size, -1]

		cf_u_vectors = self.u_embeddings(u_ids)
		cf_i_vectors = self.i_embeddings(i_ids)
  
		new_cf_u_vectors, new_cf_i_vectors = self.imix_operation(cf_u_vectors, cf_i_vectors, u_ids)

		prediction = (new_cf_u_vectors[:, None, :] * new_cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

		# prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
		# u_v = cf_u_vectors.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		# i_v = cf_i_vectors
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

class BPRMFIMix(GeneralModel, BPRMFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict =  BPRMFBase.forward(self, feed_dict)
		return {'prediction': out_dict['prediction']}

class BPRMFImpression(ImpressionModel, BPRMFBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = BPRMFBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return BPRMFBase.forward(self, feed_dict)