import numpy
from clie.objects.tree import Tree, head_to_tree, \
    tree_to_adj, adj_mat_to_dist_mat


class Sentence(object):
    """
    Sentence containing words, pos-tags, ner-tags, dependency labels along with
    head word annotation, trigger and argument indices and their type.
    Note. Each sentence contains only one trigger and one argument.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._language = None
        self._tgt_lang = None
        self._words = []
        self._knn_words = []
        self._bert_vectors = None
        self._pos = []
        self._ner = []
        self._deprel = []
        self._head = []
        self._subject = []  # start and end index
        self._subj_type = None
        self._object = []  # start and end index
        self._obj_type = None
        self._relation = None
        self._adj_mat = None
        self._dist_mat = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, param: str) -> None:
        self._language = param

    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang

    @tgt_lang.setter
    def tgt_lang(self, param: str) -> None:
        self._tgt_lang = param

    @property
    def words(self) -> list:
        return self._words

    @words.setter
    def words(self, param: list) -> None:
        assert isinstance(param, list)
        self._words = param

    @property
    def knn_words(self) -> list:
        return self._knn_words

    @knn_words.setter
    def knn_words(self, param: list) -> None:
        assert isinstance(param, list)
        k_values = [len(p) for p in param]
        assert all(k == k_values[0] for k in k_values)
        self._knn_words = param

    @property
    def bert_vectors(self) -> numpy.ndarray:
        return self._bert_vectors

    @bert_vectors.setter
    def bert_vectors(self, param: numpy.ndarray) -> None:
        assert isinstance(param, numpy.ndarray)
        assert param.shape[0] == len(self.words)
        self._bert_vectors = param

    @property
    def pos(self) -> list:
        return self._pos

    @pos.setter
    def pos(self, param: list) -> None:
        assert isinstance(param, list)
        self._pos = param

    @property
    def ner(self) -> list:
        return self._ner

    @ner.setter
    def ner(self, param: list) -> None:
        assert isinstance(param, list)
        self._ner = param

    @property
    def deprel(self) -> list:
        return self._deprel

    @deprel.setter
    def deprel(self, param: list) -> None:
        assert isinstance(param, list)
        self._deprel = param

    @property
    def head(self) -> list:
        return self._head

    @head.setter
    def head(self, param: list) -> None:
        assert isinstance(param, list)
        self._head = param

    @property
    def subject(self) -> tuple:
        return self._subject[0], self._subject[1]

    @property
    def subj_text(self) -> str:
        return ' '.join(self.words[self._subject[0]: self._subject[1] + 1])

    @subject.setter
    def subject(self, param: list) -> None:
        assert isinstance(param, list) and len(param) == 2
        self._subject = param

    @property
    def subj_type(self) -> str:
        return self._subj_type

    @subj_type.setter
    def subj_type(self, param: str) -> None:
        self._subj_type = param

    @property
    def object(self) -> tuple:
        return self._object[0], self._object[1]

    @property
    def obj_text(self) -> str:
        return ' '.join(self.words[self._object[0]: self._object[1] + 1])

    @object.setter
    def object(self, param: list) -> None:
        assert isinstance(param, list) and len(param) == 2
        self._object = param

    @property
    def obj_type(self) -> str:
        return self._obj_type

    @obj_type.setter
    def obj_type(self, param: str) -> None:
        self._obj_type = param

    @property
    def relation(self) -> str:
        return self._relation

    @relation.setter
    def relation(self, param: str) -> None:
        self._relation = param

    @property
    def subj_position(self) -> list:
        """
        self._subject[0] = 5, self._subject[1] = 6, len(self.words) = 10
        returns: [-5, -4, -3, -2, -1, 0, 0, 1, 2, 3]
        """
        return list(range(-self._subject[0], 0)) + \
               [0] * (self._subject[1] - self._subject[0] + 1) + \
               list(range(1, len(self.words) - self._subject[1]))

    @property
    def obj_position(self) -> list:
        return list(range(-self._object[0], 0)) + \
               [0] * (self._object[1] - self._object[0] + 1) + \
               list(range(1, len(self.words) - self._object[1]))

    @property
    def struct_position(self) -> tuple:
        head = [int(h) for h in self.head]
        nodes = [Tree() for _ in head]
        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
        return root, [node.depth() for node in nodes]

    def adj_mat(self, prune=-1, directed=False, self_loop=False):
        if self._adj_mat is None:
            tree = head_to_tree(self.head, self.words, len(self),
                                prune, self.subj_position, self.obj_position)
            self._adj_mat = tree_to_adj(len(self), tree, directed=directed,
                                        self_loop=self_loop)
        return self._adj_mat

    def dist_mat(self, prune=-1, directed=False, self_loop=False):
        if self._dist_mat is None:
            adj_mat = self.adj_mat(prune, directed, self_loop)
            self._dist_mat = adj_mat_to_dist_mat(adj_mat, directed=True, self_loop=False)

        return self._dist_mat

    def __len__(self):
        return len(self.words)
