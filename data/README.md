# Dataset

We preprocessed ACE05 dataset following [Cross-lingual Structure Transfer for Relation and Event Extraction](https://www.aclweb.org/anthology/D19-1030.pdf).

We expect the data inside `ace_event` and `ace_relation` directories. 
The directory structure would be identifical. 
We show the structure of `ace_event` directory as follows.

```
ace_event
  |-Arabic
  |  |-train.json
  |  |-test.json
  |  |-dev.json
  |-Chinese
  |  |-test.json
  |  |-dev.json
  |  |-train.json
  |-English
  |  |-dev.json
  |  |-train.json
  |  |-test.json
```

Once the data is ready, run the [setup.sh](https://github.com/wasiahmad/GATE/blob/main/data/setup.sh) script.
