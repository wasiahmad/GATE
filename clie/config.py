""" Implementation of all available options """
from __future__ import print_function

"""Model architecture/optimization options for IE model architecture."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type',
    'word_dim',
    'use_word',
    'pos_dim',
    'ner_dim',
    'deprel_dim',
    'type_dim',
    'position_dim',
    'comb_emb_type',
    'rnn_type',
    'rnn_hid',
    'gcn_hid',
    'tran_hid',
    'gcn_layers',
    'rnn_layers',
    'tran_layers',
    'use_all_enc_layers',
    'bidirection',
    'max_relative_pos',
    'struct_position',
    'use_neg_dist',
    'embed_graph',
    'd_ff',
    'd_k',
    'd_v',
    'num_head',
    'trans_drop',
    'mlp_layers',
    'no_adj',
    'prune_k',
    'pool_type',
    'knn_size',
    'max_tree_dist',
    'use_sent_rep'
}

BERT_CONFIG = {
    'attention_probs_dropout_prob',
    'hidden_act',
    'hidden_dropout_prob',
    'hidden_size',
    'initializer_range',
    'intermediate_size',
    'max_position_embeddings',
    'num_attention_heads',
    'num_hidden_layers',
    'type_vocab_size',
    'vocab_size'
}

DATA_OPTIONS = {
    'language',
    'max_src_len',
    'dataset_weights',
    'alpha'
}

ADVANCED_OPTIONS = {
    'bert_model',
    'use_bert',
    'freeze_bert',
    'bert_weight_file',
    'bert_config_file',
    'bert_vocab_file',
    'parallel',
    'batch_size',
    'num_epochs'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings',
    'optimizer',
    'learning_rate',
    'min_lr',
    'momentum',
    'weight_decay',
    'word_dropout',
    'dropout_rnn',
    'dropout',
    'dropout_emb',
    'dropout_gcn',
    'cuda',
    'max_grad_norm',
    'lr_decay',
    'decay_epoch',
    'gradient_accumulation_steps',
    'warmup_steps',
    'warmup_epochs',
    'fp16',
    'fp16_opt_level',
    'loss_scale',
    'conv_l2',
    'pooling_l2'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Data options
    data = parser.add_argument_group('Data parameters')
    data.add_argument('--max_src_len', type=int, default=100,
                      help='Maximum allowed length for the source sequence')
    data.add_argument('--language', type=str, nargs='+', default=['en'],
                      help='Language of the experimental data')

    # Model architecture
    model = parser.add_argument_group('Keyphrase Generator')
    model.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gtn'], help='Model architecture type')
    model.add_argument('--comb_emb_type', type=str, default='concat',
                       choices=['concat', 'add'],
                       help='The way to combine different embeddings')
    model.add_argument('--word_dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--pos_dim', type=int, default=0,
                       help='Part-of-speech embedding size')
    model.add_argument('--ner_dim', type=int, default=0,
                       help='NER tag embedding size')
    model.add_argument('--deprel_dim', type=int, default=0,
                       help='Dependency tag embedding size')
    model.add_argument('--type_dim', type=int, default=0,
                       help='Entity type embedding size')
    model.add_argument('--position_dim', type=int, default=0,
                       help='Absolute position embedding size')
    model.add_argument('--rnn_type', type=str, default='LSTM',
                       help='RNN type: LSTM, GRU')
    model.add_argument('--rnn_hid', type=int, default=200,
                       help='Hidden size of RNN units')
    model.add_argument('--gcn_hid', type=int, default=200,
                       help='Hidden size of GCN units')
    model.add_argument('--tran_hid', type=int, default=512,
                       help='Hidden size of Transformer units')
    model.add_argument('--bidirection', type='bool', default=True,
                       help='use bidirectional recurrent unit')
    model.add_argument('--use_word', type='bool', default=True,
                       help='use word embeddings')
    model.add_argument('--gcn_layers', type=int, default=1,
                       help='Number of layers in GCN')
    model.add_argument('--rnn_layers', type=int, default=1,
                       help='Number of layers in RNN encoder')
    model.add_argument('--tran_layers', type=int, default=3,
                       help='Number of layers in Transformer encoder')
    model.add_argument('--mlp_layers', type=int, default=2,
                       help='Number of MLP layers')
    model.add_argument('--use_all_enc_layers', type='bool', default=False,
                       help='Use a weighted average of all encoder layers\' '
                            'representation as the contextual representation')
    model.add_argument('--no_adj', type='bool', default=False,
                       help='Do not use adjancency information')
    model.add_argument('--prune_k', type=int, default=0,
                       help='Prune dependency paths')
    model.add_argument('--pool_type', type=str, default='max',
                       help='Pooling type')
    model.add_argument('--use_sent_rep', type='bool', default=True,
                       help='Use sentence rep. as features for classifier')

    # Transformer specific params
    model.add_argument('--embed_graph', type=int, default=0,
                       help='Embed graph information in multi-head attention')
    model.add_argument('--max_tree_dist', nargs='+', type=int, default=1,
                       help='Maximum distance to consider while constructing the adjacency matrix')
    model.add_argument('--max_relative_pos', nargs='+', type=int,
                       default=0, help='Max value for relative position representations')
    model.add_argument('--use_neg_dist', type='bool', default=True,
                       help='Use negative Max value for relative position representations')
    model.add_argument('--struct_position', type='bool', default=False,
                       help='Use structural positional embeddings')
    model.add_argument('--d_ff', type=int, default=2048,
                       help='Number of units in position-wise FFNN')
    model.add_argument('--d_k', type=int, default=64,
                       help='Hidden size of heads in multi-head attention')
    model.add_argument('--d_v', type=int, default=64,
                       help='Hidden size of heads in multi-head attention')
    model.add_argument('--num_head', type=int, default=8,
                       help='Number of heads in Multi-Head Attention')
    model.add_argument('--trans_drop', type=float, default=0.2,
                       help='Dropout for transformer')

    advanced = parser.add_argument_group('Advanced Optional Params')

    # Optimization details
    optim = parser.add_argument_group('Neural QA Reader Optimization')

    optim.add_argument('--word_dropout', type=float, default=0.04,
                       help='The rate at which randomly set a word to <UNK>')
    optim.add_argument('--dropout_emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout_rnn', type=float, default=0.2,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout_gcn', type=float, default=0.2,
                       help='Dropout rate for GCN')
    optim.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout for NN layers')
    optim.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for the optimizer')
    optim.add_argument('--min_lr', type=float, default=10e-6,
                       help='Minimum allowed learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='Decay ratio for learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5,
                        help='Decay learning rate after this epoch')
    optim.add_argument('--max_grad_norm', type=float, default=5.0,
                       help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix_embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps for gradient accumulation')
    optim.add_argument('--warmup_steps', type=int, default=0,
                       help='Percentage of warmup proportion')
    optim.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of of warmup steps')
    optim.add_argument('--conv_l2', type=float, default=0,
                       help='L2-weight decay on conv layers only')
    optim.add_argument('--pooling_l2', type=float, default=0,
                       help='L2-penalty for all pooling output')

    bert = parser.add_argument_group('Bert Configurations')
    bert.add_argument('--bert_model', type=str, default='',
                      help='Model name of the BERT')
    bert.add_argument('--use_bert', type='bool', default=False,
                      help='Use bert as a contextualized encoder')
    bert.add_argument('--freeze_bert', type='bool', default=True,
                      help='Do not train the BERT parameters')
    bert.add_argument('--attention_probs_dropout_prob', type=float, default=0.1,
                      help='Dropout rate for word embeddings')
    bert.add_argument('--hidden_act', type=str, default='gelu',
                      help='Hidden activation function')
    bert.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                      help='Dropout for hidden layers')
    bert.add_argument('--hidden_size', type=int, default=768,
                      help='Hidden size of sublayers')
    bert.add_argument('--initializer_range', type=float, default=0.02,
                      help='Initializer range for weight initialization')
    bert.add_argument('--intermediate_size', type=int, default=3072,
                      help='Intermediate size of position-wise feed-forward layer')
    bert.add_argument('--max_position_embeddings', type=int, default=512,
                      help='Maximum length of the input')
    bert.add_argument('--num_attention_heads', type=int, default=12,
                      help='Number of attention heads for multi-head attention')
    bert.add_argument('--num_hidden_layers', type=int, default=12,
                      help='Number of hidden layers in encoder and decoder')
    bert.add_argument('--type_vocab_size', type=int, default=2,
                      help='Size of the BERT type vocabulary')
    bert.add_argument('--vocab_size', type=int, default=30522,
                      help='Size of the BERT vocabulary')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER, ADVANCED_OPTIONS, DATA_OPTIONS, BERT_CONFIG

    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER | ADVANCED_OPTIONS \
                    | DATA_OPTIONS | BERT_CONFIG

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if (k in MODEL_OPTIMIZER):
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))

    return argparse.Namespace(**old_args)


def add_new_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global ADVANCED_OPTIONS
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k not in old_args:
            if (k in ADVANCED_OPTIONS):
                logger.info('Adding arg %s: %s' % (k, new_args[k]))
                old_args[k] = new_args[k]

    return argparse.Namespace(**old_args)
