import os
import json
import dsp
import argparse
from tqdm import tqdm

from toc import (
    rerank,
    parse_disambig, make_str_disambig,
    ToC, Node,
    retrieve_passages,
    get_rac_template,
    check_unique, verify_with_evidence,
)
from utils import save_results


def get_dataset(args):
    data_path = os.path.join(args.data_dir, args.data_name) 
    data = json.load(open(data_path))
    
    qa_pairs = {}
    for split in ['train', 'dev']:
        n = 0
        qa_pairs[split] = []
        for idx, (id, ins) in enumerate(data[split].items()):
            question = ins['ambiguous_question']
            answers = [anns['long_answer'] for anns in ins['annotations']]
            
            entry = {'question' : question, 
                     'answer'   : answers,
                     'id'       : id,
            }
            
            str_disambigs = make_str_disambig(ins['qa_pairs']) if split == 'train' else ""
            entry.update({'disambig' : str_disambigs})
            
            qa_pairs[split] += [entry]        
            n+=1
    
    train = [dsp.Example(**kw_example) for kw_example in qa_pairs['train']]
    dev   = [dsp.Example(**kw_example) for kw_example in qa_pairs['dev']]

    return train, dev, data


def get_example(args, train, ins, passages, 
                reranker=None, consolidation=False):
    
    question = ins.question
    with dsp.settings.context(vectorizer=dsp.SentenceTransformersVectorizer()):
        knn_func = dsp.knn(train)
        n_dynamic = args.n_shot
        demos = knn_func(ins, n_dynamic)
        demos.reverse()

    dic_example = {'question': question,
                   'demos': demos,
                   }
    
    if args.top_k_reranked > 0 and reranker is not None: # when using reranker
        top_k_reranked = min(args.top_k_reranked, len(passages))
        passages = rerank(reranker, question, passages, top_k_reranked)
        dic_example.update({'context' : passages})
    if consolidation:
        dic_example.update({'disambig' : ins.disambig})
        
    example = dsp.Example(**dic_example)
    
    return example


@dsp.transformation
def QD_predict(example: dsp.Example, qd_template, 
               sc=False, temperature=0.0):
    example, completions = dsp.generate(qd_template, temperature=temperature)(example, stage='qa')
    
    kw_out = { 
            'answer': completions.answer,
            'disambig': completions.disambig,
    }
    
    out_example = example.copy(**kw_out)
    
    return out_example, completions

def remove_demos(demos, target_ids):
    new_demos = []
    for demo in demos:
        if demo.id not in target_ids:
            new_demos += [demo]
    return new_demos

def remove_dup_demos(demos, lst_disambigs):
    answers = []
    for disambig in lst_disambigs:
        answers += [disambig['answer']]
    
    target_ids = []
    for demo in demos:
        demo_disambigs = parse_disambig(demo.disambig)
        for da in demo_disambigs:
            if dsp.metrics.F1(da['answer'], answers) > 0.8:
                target_ids += [demo.id]
    
    demos = remove_demos(demos, target_ids)
    
    return demos

def remove_dup_psgs(passages, contexts, lst_disambigs):
    answers = []
    for disambig in lst_disambigs:
        answers += [disambig['answer']]
    
    target_idxs = []
    for idx, passage in enumerate(passages):
        if passage in contexts and \
            dsp.passage_has_answers(passage, answers):
                target_idxs += [idx]
    
    passages = [passage for idx, passage in enumerate(passages) if idx not in target_idxs]
    
    return passages

def get_argparser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument("--data_name", default="ASQA.json", type=str, help="The input data name.")
    parser.add_argument("--bing_path", default=None, type=str, help="The bing data path.")
    parser.add_argument("--prefix", default='', type=str, help="The prefix of output files.")
    parser.add_argument("--model_type", default='text-davinci-003', type=str, help="The GPT model type.")
    parser.add_argument("--openai_key",  default='', type=str, required=True, help="The openai key.")
    parser.add_argument("--colbert_url", default='', type=str,required=True, help= "The colbert server url.")
    parser.add_argument("--temperature", default=0.7, type=float, help="The temperature for generation.")
    parser.add_argument("--n_shot", default=5, type=int, help="The number of few-shot examples for in-context learning.")
    parser.add_argument("--n_dev", default=-1, type=int, help="The number of dev examples to run.")
    parser.add_argument("--max_nodes", default=10, type=int, help="The maximum number of nodes in a tree.")
    parser.add_argument("--max_depth", default=3, type=int, help="The maximum depth of a tree.")
    parser.add_argument("--max_trials", default=3, type=int, help="The maximum number of restarts.")
    parser.add_argument("--top_k_docs", default=100, type=int, help="The maximum number of retrieved documents.")
    parser.add_argument("--top_k_reranked", default=5, type=int, help="The maximum number of reranked documents.")
    parser.add_argument("--save_steps", default="", type=str, help="you can save intermediate results.")
    parser.add_argument("--verify", default=False, action='store_true',)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where predictions will be written.",
    )
    return parser

def main():    
    parser = get_argparser()
    args = parser.parse_args()
    
    ## Set DSP configuration
    lm = dsp.GPT3(model=args.model_type, api_key=args.openai_key)
    rm = dsp.ColBERTv2(url=args.colbert_server)
    kw_config = {'lm' : lm, 'rm' : rm}

    if args.top_k_reranked > 0:
        kw_config['reranker'] = dsp.SentenceTransformersCrossEncoder()

    dsp.settings.configure(**kw_config)
    
    train, dev, data = get_dataset(args)
    rac_template = get_rac_template()
    
    kw_args_ex = {}
    if args.top_k_reranked > 0:
        kw_args_ex['reranker'] = dsp.settings.reranker
    
    if args.bing_path is not None:
        bing_results = json.load(open(args.bing_path))
        assert len(bing_results) == len(dev)

    if args.n_dev < 0:
        n_dev = len(dev)
    else:
        n_dev = min(args.n_dev, len(dev))
        
    os.makedirs(args.output_dir, exist_ok=True)
    save_steps = [int(save_step) for save_step in args.save_steps.split(",")] if args.save_steps != "" else []
    
    preds = [] ; outputs = []
    lst_err = []
    bing_passages = None
    for idx, ambig_ins in enumerate(tqdm(dev[:n_dev])):
        cur_demos = train.copy()
        if (idx + 1) % 10 == 0:
            print(f"{str(idx +1)} steps")
        
        if args.bing_path is not None:
            bing_passages = bing_results[idx]
        
        all_passages = retrieve_passages(args, ambig_ins, bing_passages=bing_passages)
        cur_passages = all_passages.copy()
        toc = ToC(root=Node(ambig_ins))
        do_pruning = args.verify == True
        n_restarts = 0 ; n_expansions = 0
        
        while n_restarts < args.max_trials and \
            toc.n_nodes < args.max_nodes and \
            n_expansions <= 15:
            
            n_expansions += 1
            if toc.leaf_nodes == []:
                toc.leaf_nodes = [toc.root]
                toc.leaf_nodes += toc.valid_nodes
                n_restarts += 1

            cur_node = toc.leaf_nodes.pop(0)
            cur_ins = cur_node.ins
            if cur_node.depth > args.max_depth:
                continue
            
            qd_example = get_example(args, cur_demos, cur_ins, cur_passages, **kw_args_ex)
            toc.slt_psgs += qd_example.context
            qd_result, qd_completions = QD_predict(qd_example, rac_template, sc=False, temperature=args.temperature)
            try:
                lst_disambigs = parse_disambig(qd_result.disambig)
                if lst_disambigs == []:
                    lst_disambigs = parse_disambig(qd_result.answer.split("\nAnswer:")[0])
            except:
                lst_err += [[idx, toc, qd_result.disambig]]
            
            if args.verify:
                cur_demos = remove_dup_demos(cur_demos, lst_disambigs)
                cur_passages = remove_dup_psgs(cur_passages, qd_example.context, lst_disambigs)
                
                if do_pruning:
                    valid_disambigs = []
                    for disambig in lst_disambigs:
                        if check_unique(toc.valid_qas, disambig):
                            ver_completion = verify_with_evidence(dsp.settings.lm, 
                                                                toc, 
                                                                disambig, 
                                                                dsp.settings.reranker,
                                                                all_passages)
                            if "True" in ver_completion[0]:
                                valid_disambigs += [disambig]
                    lst_disambigs = valid_disambigs.copy()
            
            if len(lst_disambigs) > 0:
                toc.add_nodes(lst_disambigs, depth=cur_node.depth+1)
                continue
            
            if do_pruning:
                if n_restarts >= args.max_trials or n_expansions >= 10:    
                    n_restarts = 0
                    do_pruning = False
                    continue
        
        tree_ins = toc._get_tree(args.max_nodes)
        
        kw_args_ex.update({'consolidation': True})
        ac_example = get_example(args, cur_demos, tree_ins, all_passages, **kw_args_ex)
        
        ac_result, ac_completions = QD_predict(ac_example, rac_template, sc=False)
        
        preds   += [ac_result.answer]
        outputs += [ac_completions.data[0]]
        
        if (idx + 1) in save_steps:
            with open(os.path.join(args.output_dir, f"output_{str(idx+1)}.json"), 'w') as f:
                json.dump(outputs, f, indent=4)
            print(f"Saved output_{idx}.json!")
    
    lm.inspect_history(n=1)
    
    save_results(args, data, preds, outputs)

if __name__ == "__main__":
    main()
    