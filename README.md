# GATE: Graph Attention Transformer Encoder
Official implementation of our AAAI 2021 paper on Cross-lingual Relation and Event Extraction. [[arxiv](https://arxiv.org/abs/2010.03009)]

### Note.
- We perform evaluation using the [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) dataset (3 languages - English, Arabic, and Chinese).
- We perform zero-shot relation extraction and event-argument role labeling.
- We consider both single-source and multi-source transfer setting.
- We implement three baseline methods for evaluation since their implementation is not publicly available. 
    - `CL_Trans_GCN` [[Liu et al., 2019](https://www.aclweb.org/anthology/D19-1068/)]
    - `CL_GCN` [[Subburathinam et al., 2019](https://www.aclweb.org/anthology/D19-1030/)]
    - `CL_RNN` [[Ni and Florian, 2019](https://www.aclweb.org/anthology/D19-1038/)]

### Training/Testing

To train and test a specific model, go to the [scripts](https://github.com/wasiahmad/GATE/tree/main/scripts) folder and run the bash files under the model directory.
For example, to train and test our GATE model, do the following.

```
$ cd  scripts/gate
$ bash relation.sh gpu_id model_name
```

Here, `model_name` is a string that will be used to name a directory under `tmp/` directory.

Once training/testing is finished, inside the `tmp/model_name/` directory, 30 files will appear. The filenames are formatted as follows, where, "src" and "tgt" are from ['en', 'ar', 'zh'].

- {src}_model_name.mdl
  - [Model file containing the parameters of the best model.]()
- {src}_model_name.mdl.checkpoint
  - [A model checkpoint, in case if we need to restart the experiment.]()
- {src}_model_name.txt
  - [Log file for training.]()
- {src}_model_name.json
  - [Contains the predictions and gold references.]()
- {src}\_model_name\_{tgt}_test.txt
  - [Log file for testing.]()
- {src}\_model_name\_{tgt}_test.json 
  - [Similar to {src}_model_name.json, but for testing.]()
- {src}\_model_name\_{tgt}.png
  - [Plot of the confusion matrix created during testing.]()


There is a python [script](https://github.com/wasiahmad/GATE/blob/main/scripts/preparer.py) that will read the log files to report the final results in the console as follows.

```
+---------+-------------------+-------------------+-------------------+
|         |      English      |       Arabic      |      Chinese      |
+---------+-------------------+-------------------+-------------------+
| English | 64.18/66.74/65.44 | 60.87/36.77/45.84 | 61.89/47.71/53.88 |
| Arabic  | 40.31/51.14/45.08 | 68.77/72.53/70.60 | 50.07/48.11/49.07 |
| Chinese | 45.01/48.75/46.80 | 59.54/46.67/52.32 | 71.55/78.98/75.08 |
+---------+-------------------+-------------------+-------------------+
```


#### Running experiments on CPU/GPU/Multi-GPU

- If `gpu_id` is set to -1, CPU will be used.
- If `gpu_id` is set to one specific number, only one GPU will be used.
- If `gpu_id` is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.


### Acknowledgement

We borrowed and modified code from [DrQA](https://github.com/facebookresearch/DrQA), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), and [Transformers](https://github.com/huggingface/transformers). We expresse our gratitdue for the authors of these repositeries.


### Citation

```
@inproceedings{ahmad2020gate,
    author = {Ahmad, Wasi Uddin and Peng, Nanyun and Chang, Kai-Wei},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    title = {GATE: Graph Attention Transformer Encoder for Cross-lingual Relation and Event Extraction},
    year = {2021}
}
```
