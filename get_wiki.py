import os
import argparse
import json

from toc import get_document, get_passages

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--data_name", default="output.json", type=str, help="The name of the input data file.")
    parser.add_argument("--top_k", default=100, type=int, help="The number of top-k documents to retrieve.")
    parser.add_argument("--save_step", default=100, type=int, help="The number of steps to save the output.")
    parser.add_argument("--debug", default=False, action="store_true", help="Whether to run in debug mode.")
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True, help="The output directory where the output files will be written.",)
    
    args = parser.parse_args()
    
    data = json.load(open(os.path.join(args.data_dir, args.data_name)))
    
    output_dir = os.path.join(args.data_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_passages = []
    n_fails = 0
    max_psgs = args.top_k
    for idx, doc in enumerate(data):
        if (idx + 1) % 10 == 0:
            print(f"Processing {idx}th data...")
        passages = []
        for d in doc:
            if "en.wikipedia.org" in d['link']:
                url = d['link']
                ret = get_document(url)
                if type(ret) == Exception: 
                    n_fails += 1
                    continue
                
                psgs = get_passages(ret)
                passages += psgs
                if len(passages) > max_psgs: break
        
        all_passages += [passages]
        
        if (idx + 1) % args.save_step == 0:
            with open(os.path.join(output_dir, f"passages_{str(idx+1)}.json"), 'w') as f:
                json.dump(all_passages, f, indent=4)
            print(f"Saved output_{idx}.json!")
            if args.debug: break
        
    
    with open(os.path.join(output_dir, f"passages.json"), 'w') as f:
        json.dump(all_passages, f, indent=4)
        
        
if __name__ == "__main__":
    main()