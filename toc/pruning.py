import dsp
from dsp.utils import deduplicate

from .templates.load import get_ver_prompt
from .retrieval.combine import rerank

def check_unique(valid_qas, cur_qa):
    is_unique = True
    for prev_qa in valid_qas:
        if cur_qa['question'] == prev_qa['question'] and \
            prev_qa['answer'] in cur_qa['answer']:
            is_unique = False
            break
    
    return is_unique

def get_evidence(tree, cur_qa, reranker):
    passages = deduplicate(tree.slt_psgs)
    pos_passages = [passage for passage in passages \
                    if dsp.passage_has_answers(passage, cur_qa['answer'])]
    if len(pos_passages) == 0:
        pos_passages = passages
    evidence = rerank(reranker, cur_qa['question'], pos_passages, 1)
    
    return evidence

def verify_with_evidence(lm, tree, cur_qa, reranker):

    evidence = get_evidence(tree, cur_qa, reranker)
    prompt = get_ver_prompt(evidence[0], tree.root.ins.question, cur_qa)
    
    completion = lm(prompt)
    
    return completion