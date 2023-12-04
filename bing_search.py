import os
import argparse
import json
import time

from langchain.utilities import BingSearchAPIWrapper

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="", type=str, required=True, help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--data_name", default="ASQA.json", type=str, help="The name of the input data file.")
    parser.add_argument("--top_k", default=50, type=int, help="The number of top-k documents to retrieve.")
    parser.add_argument("--save_step", default=100, type=int, help="The number of steps to save the output.")
    parser.add_argument("--time", default=-1, type=float, help="The time to sleep between requests.")
    parser.add_argument("--debug", default=False, action="store_true", help="Whether to run in debug mode.")
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the output files will be written.")
    
    args = parser.parse_args()
    
    data = json.load(open(os.path.join(args.data_dir, args.data_name)))
    
    search = BingSearchAPIWrapper()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    docs = []
    for idx, (id, ins) in enumerate(data['dev'].items()):
        if idx % 10 == 0:
            print(f"Processing {idx}th data...")
        
        question = ins['ambiguous_question']
        prefix = "site:en.wikipedia.org"
        doc = search.results(prefix + " " + question, args.top_k)
        
        docs += [doc]
        
        if (idx + 1) % args.save_step == 0:
            with open(os.path.join(args.output_dir, f"output_{str(idx+1)}.json"), 'w') as f:
                json.dump(docs, f, indent=4)
            print(f"Saved output_{idx}.json!")
            if args.debug: break
        
        if args.time > 0:
            time.sleep(args.time)
    
    with open(os.path.join(args.output_dir, f"output.json"), 'w') as f:
        json.dump(docs, f, indent=4)
        
        
if __name__ == "__main__":
    main()