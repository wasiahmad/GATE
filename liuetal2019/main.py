# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import clie.config as config
import liuetal2019.utils as util

from collections import OrderedDict
from tqdm import tqdm
from clie.utils.timer import AverageMeter, Timer
from clie.inputters import constant
from clie.utils import scorer

from liuetal2019.model import CL_TRANS_GCN

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--fp16', type='bool', default=False,
                         help="Whether to use 16-bit float precision instead of 32-bit")
    runtime.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model_dir', type=str, default='/tmp/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, required=True,
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, required=True,
                       help='Directory of training/validation data')
    files.add_argument('--bert_feats', type=str, default='',
                       help='Directory of BERT features')
    files.add_argument('--bert_dir', type=str, default='',
                       help='Directory of bert files')
    files.add_argument('--vocab_file', type=str, default='vocab.txt',
                       help='Preprocessed vocab file')
    files.add_argument('--embed_dir', type=str, default='',
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding_file', type=str, default='',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--tgt_language', type=str, default='',
                            choices=['ar', 'en', 'zh'],
                            help='Target language name')
    preprocess.add_argument('--knn_size', type=int, default=3,
                            help='K\'s value for nearest neighbors')
    preprocess.add_argument('--alpha', type=float, default=0.7,
                            help='Coefficient for the KNN loss')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = []
    args.valid_file = []
    args.knn_train_files = []
    args.knn_dev_files = []
    for lang in args.language:
        if not args.only_test:
            train_file = os.path.join(args.data_dir,
                                      '{}/train.json'.format(constant.LANG_MAP[lang]))
            if not os.path.isfile(train_file):
                raise IOError('No such file: %s' % train_file)
            args.train_file.append(train_file)

            knn_file = os.path.join(args.embed_dir,
                                    '{}_{}_knn.txt'.format(lang, args.tgt_language))
            if not os.path.isfile(knn_file):
                raise IOError('No such file: %s' % knn_file)
            args.knn_train_files.append(knn_file)

        tgt_file = 'test' if args.only_test else 'dev'
        valid_file = os.path.join(args.data_dir, '{}/{}.json'.format(
            constant.LANG_MAP[lang], tgt_file))
        if not os.path.isfile(valid_file):
            raise IOError('No such file: %s' % valid_file)
        args.valid_file.append(valid_file)

        if not args.only_test:
            knn_file = os.path.join(args.embed_dir,
                                    '{}_{}_knn.txt'.format(lang, args.tgt_language))
            if not os.path.isfile(knn_file):
                raise IOError('No such file: %s' % knn_file)
            args.knn_dev_files.append(knn_file)
        else:
            args.knn_dev_files.append(None)

    if not args.only_test:
        args.vocab_file = os.path.join(args.embed_dir, args.vocab_file)
        if not os.path.isfile(args.vocab_file):
            raise IOError('No such file: %s' % args.vocab_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.log_file = os.path.join(args.model_dir, args.model_name + '%s.txt' % suffix)
    args.pred_file = os.path.join(args.model_dir, args.model_name + '%s.json' % suffix)
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)
        with open(args.embedding_file, encoding='utf-8') as f:
            # if first line is of form count/dim.
            line = f.readline().rstrip().split(' ')
            dim = int(line[1]) if len(line) == 2 \
                else len(line) - 1
        args.word_dim = dim if args.use_word else 0

    return args


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    cl_loss = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)
    pbar.set_description("%s" % 'Epoch = %d [loss = x.xx]' % global_stats['epoch'])

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']

        loss = model.update(ex)
        cl_loss.update(loss, bsz)

        log_info = 'Epoch = %d [loss = %.2f]' % (global_stats['epoch'], cl_loss.avg)
        pbar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | loss = %.2f | Time for epoch = %.2f (s)' %
                (global_stats['epoch'], cl_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------

def draw_confusion_matrix(cm, labels, filename):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                annot[i, j] = '%d' % c
            elif c == 0:
                annot[i, j] = ''
            else:
                # annot[i, j] = '%.1f%%\n%d' % (p, c)
                annot[i, j] = '%d' % c

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(12, 12))
    midpoint = (cm.values.max() - cm.values.min()) / 2
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, center=midpoint,
                linewidths=0.01, cmap="YlGnBu", cbar=False)
    plt.tight_layout()
    plt.savefig(filename)


def validate(args, data_loader, model, global_stats, mode='valid'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    results = []
    total_example = 0

    with torch.no_grad():
        pbar = tqdm(data_loader)
        for ex in pbar:
            output = model.predict(ex)
            gold_labels = ex['labels'].tolist()
            for idx in range(len(gold_labels)):
                results.append(OrderedDict([
                    ('id', ex['ids'][idx]),
                    ('subject', ex['subject'][idx]),
                    ('object', ex['object'][idx]),
                    ('pred', model.label_dict[output['predictions'][idx]]),
                    ('gold', model.label_dict[gold_labels[idx]])
                ]))
            pbar.set_description("%s" % 'Epoch = %d [validating ... ]' %
                                 global_stats['epoch'])
            total_example += ex['batch_size']

    scorer_out = scorer.score(results, verbose=True)
    logger.info('Validation: precision = %.2f | recall = %.2f | f1 = %.2f |'
                ' examples = %d | %s time = %.2f (s) ' %
                (scorer_out['precision'] * 100, scorer_out['recall'] * 100,
                 scorer_out['f1'] * 100, total_example, mode, eval_time.time()))
    logger.info('\n' + scorer_out['verbose_out'])

    with open(args.pred_file, 'w') as fw:
        for item in results:
            fw.write(json.dumps(item) + '\n')

    if mode == 'test':
        cm_filename = os.path.join(
            args.model_dir, args.model_name + '_%s.png' % ('_'.join(args.language)))
        draw_confusion_matrix(cm=scorer_out['confusion_matrix'],
                              labels=scorer_out['labels'],
                              filename=cm_filename)

    return {
        'precision': scorer_out['precision'],
        'recall': scorer_out['recall'],
        'f1': scorer_out['f1']
    }


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def supervise_training(args, model, start_epoch, train_loader, dev_loader):
    stats = {
        'timer': Timer(),
        'epoch': start_epoch,
        'best_valid': 0,
        'no_improvement': 0
    }

    for epoch in range(start_epoch, args.num_epochs + 1):
        stats['epoch'] = epoch
        train(args, train_loader, model, stats)
        result = validate(args, dev_loader, model, stats)
        valid_metric_perf = float(result['{}'.format(args.valid_metric)])

        # Save best valid
        if valid_metric_perf > stats['best_valid']:
            logger.info('Best valid: %s = %.4f (epoch %d, %d updates)' %
                        (args.valid_metric, valid_metric_perf,
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = valid_metric_perf
            stats['no_improvement'] = 0
        else:
            stats['no_improvement'] += 1
            if stats['no_improvement'] >= args.early_stop:
                break


def main(args):
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = CL_TRANS_GCN.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = CL_TRANS_GCN.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = CL_TRANS_GCN.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = CL_TRANS_GCN.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                # Build a dictionary from the data questions + words (train/dev splits)
                logger.info('-' * 100)
                # Initialize model
                model = CL_TRANS_GCN(config.get_model_args(args),
                                     vocab_file=args.vocab_file)

                # Load pretrained embeddings for words in dictionary
                if model.args.use_word and args.embedding_file:
                    model.load_embeddings(args.embedding_file)

            # Set up optimizer
            model.init_optimizer()
            logger.info('Trainable #parameters [total] {}'.format(
                human_format(model.network.count_parameters())))
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    if not args.only_test and args.use_bert:
        model.activate_fp16()

    # multi-gpu training (should be after apex fp16 initialization)
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files [language - {}]'.format(
        ', '.join(args.language)))

    train_exs = []
    dataset_weights = dict()
    if not args.only_test:
        for train_file, knn_file, lang in \
                zip(args.train_file, args.knn_train_files, args.language):
            exs = util.load_data(train_file,
                                 src_lang=lang,
                                 tgt_lang=args.tgt_language,
                                 knn_file=knn_file,
                                 knn_size=args.knn_size,
                                 max_examples=args.max_examples)
            dataset_weights[constant.LANG_MAP[lang]] = len(exs)
            train_exs.extend(exs)

        logger.info('Num train examples = %d' % len(train_exs))
        args.num_train_batch = len(train_exs) // args.batch_size
        for lang_id in dataset_weights.keys():
            weight = (1.0 * dataset_weights[lang_id]) / len(train_exs)
            dataset_weights[lang_id] = round(weight, 2)
        logger.info('Dataset weights = %s' % str(dataset_weights))

    model.args.dataset_weights = dataset_weights

    dev_exs = []
    for valid_file, knn_file, lang in \
            zip(args.valid_file, args.knn_dev_files, args.language):
        exs = util.load_data(valid_file,
                             src_lang=lang,
                             tgt_lang=None,
                             knn_file=knn_file,
                             knn_size=args.knn_size,
                             max_examples=args.max_examples)
        dev_exs.extend(exs)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    train_loader = None
    if not args.only_test:
        train_dataset = util.ACE05Dataset(train_exs, model, evaluation=False)
        if args.sort_by_len:
            train_sampler = util.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=util.batchify,
            pin_memory=args.cuda
        )

    dev_dataset = util.ACE05Dataset(dev_exs, model, evaluation=True)
    dev_sampler = util.SortedBatchSampler(dev_dataset.lengths(),
                                          args.batch_size,
                                          shuffle=False)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=util.batchify,
        pin_memory=args.cuda
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    # if not args.only_test:
    #     logger.info('-' * 100)
    #     logger.info('CONFIG:\n%s' %
    #                 json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate(args, dev_loader, model, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        supervise_training(args, model, start_epoch, train_loader, dev_loader)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Cross-lingual Information Extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.device_count = torch.cuda.device_count()
    args.parallel = args.device_count > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
