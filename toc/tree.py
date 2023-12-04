import dsp
from dsp.utils import deduplicate
from .retrieval.combine import rerank
from .utils import make_str_disambig


class ToC:
    def __init__(self, root):
        self.root = root
        self.n_nodes = 0
        self.valid_qas = []
        self.valid_nodes = []
        self.slt_psgs = []
        self.leaf_depth = 1
        self.leaf_nodes = [self.root]
        
    def add_node(self, qa, depth):
        new_ins = self.root.ins.copy(question=qa['question'])
        new_ins.question = qa['question']
        self.n_nodes += 1
        self.leaf_nodes += [Node(new_ins, depth)]
        self.valid_nodes += [Node(new_ins, depth)]
        self.valid_qas += [qa]
        
    def add_nodes(self, qas, depth):
        if depth > self.leaf_depth:
            self.leaf_depth = depth
        
        for qa in qas:
            self.add_node(qa, depth)
        
    def _get_tree(self, n_out_nodes):
        n_outs = min(len(self.valid_qas), n_out_nodes)
        str_disambigs = make_str_disambig(self.valid_qas[:n_outs], ambigQA=False)
        dic_example = {'question': self.root.ins.question,
                       'id'      : self.root.ins.id,
                       'disambig': str_disambigs,
                       }
        
        return dsp.Example(**dic_example)


class Node:
    def __init__(self, ins, depth=1):
        self.ins = ins # dsp.Example
        self.depth = depth
    
    
def select_passages(disambigs, tree, reranker, n_passages=5):
    psgs = deduplicate(tree.valid_psgs)
    
    idxs_psgs = []
    for disambig in disambigs:
        for idx_psg, passage in enumerate(psgs):
            if dsp.passage_has_answers(passage, disambig['answer'].split(";")) and \
                idx_psg not in idxs_psgs:
                idxs_psgs.append(idx_psg)
                break            
    idxs_slt = min(n_passages, len(idxs_psgs))
    slt_passages = [psgs[idx_psg] for idx_psg in idxs_psgs[:idxs_slt]]
    
    if len(idxs_psgs) < n_passages:
        # """randomly sample passages and append them"""    
        rest = set(range(len(psgs))) - set(idxs_psgs)
        rest_psgs = [psgs[idx_rest] for idx_rest in rest]
        slt_passages += rest_psgs
        
    slt_passages = rerank(reranker, tree.root.ins.question, slt_passages, n_passages)
    
    return slt_passages