import dsp

def retrieve_passages(args, ins, bing_passages=None):
    question = ins.question
    passages = dsp.retrieve(question, k=args.top_k_docs)
        
    if bing_passages is not None:
        passages += bing_passages
    
    return passages