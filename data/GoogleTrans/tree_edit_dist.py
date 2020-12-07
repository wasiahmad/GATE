import math
import numpy as np
from conllu import parse
from collections import OrderedDict
from apted import APTED
from apted.helpers import Tree as aptedTree
from matplotlib import pyplot as plt
from tqdm import tqdm


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.idx = None
        self.token = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def print(self, level):
        for i in range(1, level):
            print('|----', end='')
        print(self.token)
        for i in range(self.num_children):
            self.children[i].print(level + 1)

    def size(self):
        if getattr(self, '_size', False):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def height(self):
        if getattr(self, '_height', False):
            return self._height
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_height = self.children[i].height()
                if child_height > count:
                    count = child_height
            count += 1
        self._height = count
        return self._height

    def depth(self):
        if getattr(self, '_depth', False):
            return self._depth
        count = 0
        if self.parent:
            count += self.parent.depth()
            count += 1
        self._depth = count
        return self._depth

    def delete(self):
        for i in range(self.num_children):
            self.parent.add_child(self.children[i])
            self.children[i].parent = self.parent
        index = None
        for i in range(self.parent.num_children):
            if self.parent.children[i].idx == self.idx:
                index = i
                break
        self.parent.children.pop(index)
        self.parent.num_children -= 1

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def head_to_tree(head, tokens):
    """
    Convert a sequence of head indexes into a tree object.
    """
    root = None
    nodes = [Tree() for _ in head]
    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].token = tokens[i]
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    return root, nodes


def treeToString(root: Tree, string: list):
    # base case
    if root is None:
        return

    # push the root data as character
    string.append(str(root.token))

    for i in range(root.num_children):
        string.append('{')
        treeToString(root.children[i], string)
        string.append('}')


def load_conllu(conllu_file):
    conllu_data = dict()
    with open(conllu_file, 'r', encoding='utf-8') as content_file:
        content = content_file.read()
        sentences = parse(content)
        for idx, sentence in enumerate(sentences):
            tokens, upos, head, deprel = [], [], [], []
            for widx, word in enumerate(sentence):
                if isinstance(word['id'], tuple):
                    # multi-word token, e.g., word['id'] = (4, '-', 5)
                    assert len(word['id']) == 3
                else:
                    tokens.append(word['form'])
                    upos.append(word['upostag'])
                    head.append(word['head'])
                    deprel.append(word['deprel'])

            sent_obj = OrderedDict([
                ('id', sentence.metadata['sent_id']),
                ('text', sentence.metadata['text']),
                ('word', tokens),
                ('upos', upos),
                ('head', head),
                ('deprel', deprel)
            ])
            conllu_data[sent_obj['id']] = sent_obj

    return conllu_data


def compute_edit_distance(src_file, para_file):
    src_data = load_conllu(src_file)
    para_data = load_conllu(para_file)
    assert len(src_data) == len(para_data)

    edit_distances = []
    for key in tqdm(src_data.keys(), total=len(src_data)):
        src_sent = src_data[key]
        para_sent = para_data[key]
        src_root, _ = head_to_tree(src_sent['head'], src_sent['upos'])
        para_root, _ = head_to_tree(para_sent['head'], para_sent['upos'])
        src_tree_to_string = []
        treeToString(src_root, src_tree_to_string)
        src_tree_to_string = ['{'] + src_tree_to_string + ['}']
        src_tree_to_string = ''.join(src_tree_to_string)
        para_tree_to_string = []
        treeToString(para_root, para_tree_to_string)
        para_tree_to_string = ['{'] + para_tree_to_string + ['}']
        para_tree_to_string = ''.join(para_tree_to_string)
        # print(src_tree_to_string)
        # print(para_tree_to_string)
        apted = APTED(aptedTree.from_text(src_tree_to_string),
                      aptedTree.from_text(para_tree_to_string))
        ted = apted.compute_edit_distance()
        edit_distances.append(ted)
        # mapping = apted.compute_edit_mapping()
        # print(mapping)

    return edit_distances


if __name__ == '__main__':
    edit_distances = compute_edit_distance('ace_event/Chinese/source_sent.conllu',
                                           'ace_event/Chinese/parallel_sent.conllu')
    bins = np.linspace(math.ceil(min(edit_distances)),
                       math.floor(max(edit_distances)),
                       20)  # fixed number of bins

    fig = plt.figure(figsize=(8, 6))
    plt.xlim([min(edit_distances) - 5, max(edit_distances) + 5])
    plt.hist(edit_distances, bins=bins, facecolor='blue',
             edgecolor='black', alpha=0.2, linewidth=1.0)
    plt.title('Tree Edit Distances')
    plt.xlabel('Distances (20 evenly spaced bins)')
    plt.ylabel('Count')
    fig.tight_layout()
    fig.savefig('edit_distances.png')
