'''
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
'''

from copy import deepcopy
import tensorflow as tf

class EWC(object):
    def __init__(self, ppo_networks, params_tf, dataset):

        self.ppo_networks = ppo_networks
        self.dataset = dataset
        
        self.params = {}
        for layer_name, w_b_dict in self.params_tf.items(): 
            if len(w_b_dict.keys()) != 2:
                raise "Model isnot well-formed"
            self.params.update({f"{layer_name}.weight": w_b_dict['w']})
            self.params.update({f"{layer_name}.bias": w_b_dict['b']})
        
        # w/o gradient
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = p
       
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        precision_matrices = {var.name: tf.zeros_like(var) for var in self.model.trainable_variables}

        '''
        for input in self.dataset:
            with tf.GradientTape() as g:
                policy_log_probs = ppo_networks.log_prob(distribution_params, actions)
                output = tf.reshape(output, (1, -1))
                label = tf.argmax(output, axis=1)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)
            
            
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        '''
        return precision_matrices

# params: tf model, layer_name -> w/b -> value
class L2(object):
    def __init__(self, params):
        self.old_params = params
        self.importance = 1
        #self.importance = 500
    
    def regularize(self, new_params):
        loss = 0
        #print(new_params['mlp/~/linear_1']['w'])
        #print(self.old_params['mlp/~/linear_1']['w'])
        for (new_layer_name, new_w_b_dict), (old_layer_name, old_w_b_dict), in zip(new_params.items(), self.old_params.items()): 
            if new_layer_name != old_layer_name:
                raise "Model name isn't well-formed"
            if len(new_w_b_dict.keys()) != 2:
                raise "Model value isn't well-formed"

            _loss = self.importance * (new_w_b_dict['w'] - old_w_b_dict['w']) ** 2
            loss += _loss.sum()
            _loss = self.importance * (new_w_b_dict['b'] - old_w_b_dict['b']) ** 2
            loss += _loss.sum()
        return loss
