import copy
import json
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from clie.config import override_model_args
from clie.inputters import Vocabulary, constant, load_word_embeddings
from clie.model import GCNClassifier, GTNClassifier

logger = logging.getLogger(__name__)


class EventRelationExtractor(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, state_dict=None, vocab_file=None):
        # Book-keeping.
        self.name = 'EventRelationExtractor'
        self.args = args
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if vocab_file:
            self.load_vocabulary(vocab_file)

        self.args.bert_dim = 768 if self.args.use_bert else 0
        if self.args.model_type == 'gcn':
            self.network = GCNClassifier(self.args)
        elif self.args.model_type == 'gtn':
            self.network = GTNClassifier(self.args)
        else:
            raise NotImplementedError
        self.criterion = nn.CrossEntropyLoss()

        # Load saved state
        if state_dict:
            self.network.load_state_dict(state_dict)

    def load_vocabulary(self, vocab_file):
        logger.info('Build word dictionary')
        with open(vocab_file) as f:
            vocab = json.load(f)
        self.word_dict = Vocabulary('regular')
        self.word_dict.add_tokens(vocab['tokens'])
        self.pos_dict = Vocabulary('regular')
        self.pos_dict.add_tokens(vocab['pos_tokens'])
        self.ner_dict = Vocabulary('regular')
        self.ner_dict.add_tokens(vocab['ner_tokens'])
        self.deprel_dict = Vocabulary('regular')
        self.deprel_dict.add_tokens(vocab['deprel_labels'])
        self.type_dict = Vocabulary('regular')
        self.type_dict.add_tokens(vocab['subj_types'] + vocab['obj_types'])
        self.label_dict = Vocabulary()
        self.label_dict.add_tokens(vocab['labels'])
        self.args.word_size = len(self.word_dict)
        self.args.pos_size = len(self.pos_dict)
        self.args.ner_size = len(self.ner_dict)
        self.args.deprel_size = len(self.deprel_dict)
        self.args.type_size = len(self.type_dict)
        self.args.label_size = len(self.label_dict)
        logger.info('Num words = %d, pos = %d, ner = %d, deprel = %d, type = %d, label = %d' %
                    (self.args.word_size, self.args.pos_size, self.args.ner_size,
                     self.args.deprel_size, self.args.type_size, self.args.label_size))

    def load_embeddings(self, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            embedding_file: path to text file of embeddings, space separated.
        """
        embedding_index = load_word_embeddings(embedding_file)
        vocab_size, emsize = self.args.word_size, self.args.word_dim
        pretrained = np.zeros([vocab_size, emsize])
        emb_found = 0
        for i in range(self.args.word_size):
            word = self.word_dict.ind2tok[i]
            if word in embedding_index:
                emb_found += 1
                pretrained[i] = embedding_index[word]
        self.network.model.embedding.word_emb.weight.data. \
            copy_(torch.from_numpy(pretrained))
        logger.info('Loaded embeddings for %d/%d words (%.2f%%).' % (
            emb_found, vocab_size, 100.0 * emb_found / vocab_size))

    def activate_fp16(self):
        if self.args.fp16:
            try:
                global amp
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            # https://github.com/NVIDIA/apex/issues/227
            assert self.optimizer is not None
            self.network, self.optimizer = amp.initialize(self.network,
                                                          self.optimizer,
                                                          opt_level=self.args.fp16_opt_level)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.use_word and self.args.fix_embeddings:
            self.network.model.embedding.word_emb.weight.requires_grad = False

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters, self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        len_rep = ex['len_rep']
        word_rep = ex['word_rep']
        position_rep = ex['position_rep']
        pos_rep = ex['pos_rep']
        ner_rep = ex['ner_rep']
        deprel_rep = ex['deprel_rep']
        head_rep = ex['head_rep']
        type_rep = ex['type_rep']
        labels = ex['labels']
        subject_pos_rep = ex['subject_pos_rep']
        object_pos_rep = ex['object_pos_rep']
        bert_embeds = ex['bert_embeds']
        struct_relpos = ex['struct_relpos_rep']
        adj_mask = ex['adj_mask']
        ex_weights = [self.args.dataset_weights[constant.LANG_MAP[lang]]
                      for lang in ex['language']]
        ex_weights = torch.FloatTensor(ex_weights)

        if self.use_cuda:
            len_rep = len_rep.cuda(non_blocking=True)
            word_rep = word_rep.cuda(non_blocking=True)
            pos_rep = pos_rep.cuda(non_blocking=True)
            ner_rep = ner_rep.cuda(non_blocking=True)
            deprel_rep = deprel_rep.cuda(non_blocking=True)
            head_rep = head_rep.cuda(non_blocking=True)
            type_rep = type_rep.cuda(non_blocking=True)
            subject_pos_rep = subject_pos_rep.cuda(non_blocking=True)
            object_pos_rep = object_pos_rep.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            ex_weights = ex_weights.cuda(non_blocking=True)
            struct_relpos = struct_relpos.cuda(non_blocking=True) \
                if struct_relpos is not None else None
            position_rep = position_rep.cuda(non_blocking=True) \
                if position_rep is not None else None
            bert_embeds = bert_embeds.cuda(non_blocking=True) \
                if bert_embeds is not None else None

        # Run forward
        outputs = self.network(len_rep=len_rep,
                               word_rep=word_rep,
                               pos_rep=pos_rep,
                               ner_rep=ner_rep,
                               deprel_rep=deprel_rep,
                               head_rep=head_rep,
                               type_rep=type_rep,
                               subj_rep=subject_pos_rep,
                               obj_rep=object_pos_rep,
                               position_rep=position_rep,
                               bert_embeds=bert_embeds,
                               adj_mask=adj_mask,
                               struct_relpos=struct_relpos)

        logits = outputs[0]
        loss = self.criterion(logits, labels) * ex_weights
        loss = loss.mean()

        if self.args.model_type == 'gcn':
            pooled_out = outputs[1]
            # l2 decay on all conv layers
            if self.args.conv_l2 > 0:
                loss += self.network.conv_l2() * self.args.conv_l2
            # l2 penalty on output representations
            if self.args.pooling_l2 > 0:
                loss += self.args.pooling_l2 * (pooled_out ** 2).sum(1).mean()
        elif self.args.model_type == 'gtn':
            pass

        self.optimizer.zero_grad()

        if self.args.fp16:
            global amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
        else:
            loss.backward()
            clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        return loss.item()

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
            src_raw: raw source (passage); required to replace `unk` term
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        len_rep = ex['len_rep']
        word_rep = ex['word_rep']
        position_rep = ex['position_rep']
        pos_rep = ex['pos_rep']
        ner_rep = ex['ner_rep']
        deprel_rep = ex['deprel_rep']
        head_rep = ex['head_rep']
        type_rep = ex['type_rep']
        labels = ex['labels']
        subject_pos_rep = ex['subject_pos_rep']
        object_pos_rep = ex['object_pos_rep']
        bert_embeds = ex['bert_embeds']
        adj_mask = ex['adj_mask']
        struct_relpos = ex['struct_relpos_rep']

        if self.use_cuda:
            len_rep = len_rep.cuda(non_blocking=True)
            word_rep = word_rep.cuda(non_blocking=True)
            pos_rep = pos_rep.cuda(non_blocking=True)
            ner_rep = ner_rep.cuda(non_blocking=True)
            deprel_rep = deprel_rep.cuda(non_blocking=True)
            head_rep = head_rep.cuda(non_blocking=True)
            type_rep = type_rep.cuda(non_blocking=True)
            subject_pos_rep = subject_pos_rep.cuda(non_blocking=True)
            object_pos_rep = object_pos_rep.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            struct_relpos = struct_relpos.cuda(non_blocking=True) \
                if struct_relpos is not None else None
            position_rep = position_rep.cuda(non_blocking=True) \
                if position_rep is not None else None
            bert_embeds = bert_embeds.cuda(non_blocking=True) \
                if bert_embeds is not None else None

        logits, _ = self.network(len_rep=len_rep,
                                 word_rep=word_rep,
                                 pos_rep=pos_rep,
                                 ner_rep=ner_rep,
                                 deprel_rep=deprel_rep,
                                 head_rep=head_rep,
                                 type_rep=type_rep,
                                 subj_rep=subject_pos_rep,
                                 obj_rep=object_pos_rep,
                                 position_rep=position_rep,
                                 bert_embeds=bert_embeds,
                                 adj_mask=adj_mask,
                                 struct_relpos=struct_relpos)

        loss = self.criterion(logits, labels)
        probs = f.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        return {
            'loss': loss,
            'probs': probs,
            'predictions': predictions,
        }

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'pos_dict': self.pos_dict,
            'ner_dict': self.ner_dict,
            'type_dict': self.type_dict,
            'deprel_dict': self.deprel_dict,
            'label_dict': self.label_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'pos_dict': self.pos_dict,
            'ner_dict': self.ner_dict,
            'type_dict': self.type_dict,
            'deprel_dict': self.deprel_dict,
            'label_dict': self.label_dict,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        model = EventRelationExtractor(
            args,
            state_dict=saved_params['state_dict']
        )
        model.word_dict = saved_params['word_dict']
        model.pos_dict = saved_params['pos_dict']
        model.ner_dict = saved_params['ner_dict']
        model.type_dict = saved_params['type_dict']
        model.deprel_dict = saved_params['deprel_dict']
        model.label_dict = saved_params['label_dict']
        return model

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        model = EventRelationExtractor(
            saved_params['args'],
            state_dict=saved_params['state_dict']
        )
        model.word_dict = saved_params['word_dict']
        model.pos_dict = saved_params['pos_dict']
        model.ner_dict = saved_params['ner_dict']
        model.type_dict = saved_params['type_dict']
        model.deprel_dict = saved_params['deprel_dict']
        model.label_dict = saved_params['label_dict']
        model.updates = saved_params['updates']
        model.init_optimizer(saved_params['optimizer'], use_gpu)
        return model, saved_params['epoch']

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()
        self.criterion = self.criterion.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()
        self.criterion = self.criterion.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
