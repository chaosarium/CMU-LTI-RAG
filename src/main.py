import argparse
import datetime
import json
from tqdm import tqdm
from lib.pipeline import RAGPipeline
from lib.preprocess import TextDataset, PdfDataset, ComposedDataset, splitter_choices
from lib.retrieval import BM25Retriever, RAGatouilleRetriever, VectorRetriever
from lib.rerank import NoopReranker
from lib.generate import GeneratorBase, Llama13BGenerator, Llama70BGenerator, Mixtral8x7BGenerator, Gemma7BGenerator, TogetherGeneratorBase, format_documents
from lib.evaluation import load_annotated_qas, eval_exact_match, eval_f1, eval_rouge, eval_recall, eval_precision, eval_precision_recall
import pandas as pd
import time
import os
from pprint import pprint
import pickle

test_set_groups = {
    "toy": [
        '../annotations/toyQAs-on-papers.json',
    ],
    "courses": [
        '../annotations/coursesQAs.json',
    ],
    "human": [
        '../annotations/karina-annotated.json',
        '../annotations/leon-annotated.json',
    ],
    "template": [
        '../annotations/directoryQAs.json',
        '../annotations/coursesQAs.json',
        '../annotations/papersQAs.json',
    ],
}

def gt_args():
    parser = argparse.ArgumentParser(description='Run RAG system')
    parser.add_argument('--mode', type=str, choices=['index', 'retrieve', 'repl', 'evaluate'], default='repl')
    parser.add_argument('--sleeptime', type=float, default=1.2, help='time to sleep between requests to LLM API')
    parser.add_argument('--in_mem_index', action="store_true", default=True, help="whether to store the index in memory, if false... we don't know what happens")
    parser.add_argument('--dataset', type=str, choices=['txt', 'papers', 'directory', 'courses'], default='txt', help='which subset of raw data to retrieve from; txt is the full collection')
    parser.add_argument('--testset', type=str, choices=list(test_set_groups.keys()), default='toy', help='which test set to use')
    parser.add_argument('--retriever', type=str, choices=['bm25', 'ragatouille', 'vec'], default='bm25', help='which retriever to use')
    parser.add_argument('--reranker', type=str, choices=['noop'], default='noop', help="which reranker to use; apparently we didn't implement any useful rerankers")
    parser.add_argument('--generator', type=str, choices=['llama13b', 'llama70b', 'mixtral8x7b', 'gemma7b'], default='llama13b')
    parser.add_argument('--k', type=int, default=4, help='how many documents to consider for each query')
    parser.add_argument('--hypothetical', action="store_true", default=False, help='whether to use hypothetical questions')
    parser.add_argument('--chunkaugmentation', action="store_true", default=False, help='whether to augment text chunks by prepending metadata to each chunk')
    return parser.parse_args()

def mk_dataset(args):
    match args.dataset:
        case 'full':
            raise NotImplementedError
        case 'txt':
            dataset = TextDataset(
                data_dir='../collection',
                text_splitter = splitter_choices['recursive_char_text_splitter'],
                metadata_augment=args.chunkaugmentation
            )
        case 'papers' | 'directory' | 'courses':
            dataset = TextDataset(
                data_dir=f'../collection/{args.dataset}',
                text_splitter = splitter_choices['recursive_char_text_splitter'],
                metadata_augment=args.chunkaugmentation
            )
        case _:
            raise ValueError
    dataset.print_summary()
    return dataset

def mk_retriever(args, dataset, do_index: bool):
    print(args)
    print(dataset)
    match args.retriever:
        case 'bm25':
            return BM25Retriever(dataset=dataset)
        case 'ragatouille':
            return RAGatouilleRetriever(dataset=dataset, dataset_identifier=args.dataset, do_index=do_index, in_memory=args.in_mem_index)
        case 'vec':
            return VectorRetriever(dataset=dataset, model_name = "intfloat/e5-large-v2")
        case _:
            raise ValueError
    
def mk_reranker(args):
    match args.reranker:
        case 'noop':
            return NoopReranker()
        case _:
            raise ValueError

def mk_generator(args):
    match args.generator:
        case 'llama13b':
            return Llama13BGenerator()
        case 'llama70b':
            return Llama70BGenerator()
        case 'gemma7b':
            return Gemma7BGenerator()
        case 'mixtral8x7b':
            return Mixtral8x7BGenerator()
        case _:
            raise ValueError

def qa_repl(pipeline: RAGPipeline):
    while True:
        Q = input("Enter question: ")
        docs = pipeline.retrieval_pass(Q)['retrieved']
        A_hat, model_output, generation_prompt = generator.answer_with_context(Q, docs)
        print(f"Answer: {A_hat}")

def do_evaluation(generator: TogetherGeneratorBase, args):
    retrieval_identifier = mk_retrieval_experiment_identifier(args).replace(args.generator, "llama13b") # augmented data source always use llama13b
    evaluation_identifier = mk_eval_experiment_identifier(args)
    
    with open(f'../experiment/{retrieval_identifier}/retrieval_acc.pkl', 'rb') as file:
        retrieval_acc = pickle.load(file)
        
    res_acc = []
    for i, entry in tqdm(enumerate(retrieval_acc), desc="evaluating...", total=len(retrieval_acc)):
     
        Q, A = entry['Q'], entry['A']
        
        if args.testset == 'template' and i % 3 != 0:
            continue
   
        if args.k == 0:
            for i in range(10):
                try:
                    time.sleep(args.sleeptime)
                    A_hat, model_output, generation_prompt = generator.no_context_answer(Q)
                    break
                except BaseException as e:
                    print(f'failed to evaluate, {e}, rate limiting?')
                    time.sleep(args.sleeptime)
                    if i == 9:
                        A_hat, model_output, generation_prompt = 'failed', 'failed', 'failed'
                        retry = False
                        print(f'GIVE UP on entry {entry}. need to retry manually!')
                        break
        else:
            try_k = args.k
            retry = True
            strike = 0
            while retry:
                docs = entry['reranked'][:min(try_k, len(entry['reranked']))] # expect enough docs cached
                try:
                    time.sleep(args.sleeptime)
                    A_hat, model_output, generation_prompt = generator.answer_with_context(Q, docs)
                    break
                except BaseException as e:
                    print(f'failed to evaluate when k = {try_k} context window issue?, {e}')
                    time.sleep(args.sleeptime)
                    strike += 1
                    if strike > 2:
                        try_k -= 1
                    if try_k == 0:
                        A_hat, model_output, generation_prompt = 'failed', 'failed', 'failed'
                        retry = False
                        print(f'GIVE UP on entry {entry}. need to retry manually!')
                        break
                

        em = eval_exact_match(A_hat, [A])
        f1 = eval_f1(A_hat, [A])
        recall = eval_recall(A_hat, [A])
        precision = eval_precision(A_hat, [A])
        
        entry_res = {
            "em": em,
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "A_hat": A_hat,
            **entry,
            "model_output": model_output,
            "generation_prompt": generation_prompt,
        }
        if 'Q_aug' not in entry_res:
            entry_res['Q_aug'] = '-'
        res_acc.append(entry_res)
    
    # summary stats and file IO by copilot
    output_dir = f'../experiment/evaluation/{evaluation_identifier}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 0 > save arguments
    args_file = f'{output_dir}/args.txt'
    with open(args_file, 'w') as f:
        f.write(str(args))
    
    # 1 > save predictions
    with open(f'{output_dir}/res_acc.pkl', 'wb') as f:
        pickle.dump(res_acc, f)

    df = pd.DataFrame(res_acc)

    df.to_csv(f'{output_dir}/predictions.csv')
    df.to_json(f'{output_dir}/predictions.json', indent=2, orient='records')
    df.to_html(f'{output_dir}/predictions.html')
    df.style.background_gradient(subset=['em', 'f1', 'recall', 'precision'], cmap='RdYlGn', vmin=0, vmax=1).to_html(f'{output_dir}/predictions_s.html')
    df[["em", "f1", "recall", "precision", 'Q', "A_hat", 'A']].style.background_gradient(subset=['em', 'f1', 'recall', 'precision'], cmap='RdYlGn', vmin=0, vmax=1).to_html(f'{output_dir}/predictions_s_cleaner.html')
    
    # 2 > make mean calculations
    summary_stats = df[['em', 'f1', 'recall', 'precision']].mean().to_frame().T.rename(columns={'em': 'Mean Exact Match', 'f1': 'Mean F1 Score', 'recall': 'Mean Recall', 'precision': 'Mean Precision'})
    summary_stats.to_csv(f'{output_dir}/stats.csv', index=False)
    summary_stats.to_json(f'{output_dir}/stats.json', indent=2, orient='records')
    
    # 3 > bye
    print("done evaluating!")
    print("results dumped to", output_dir)
    print(summary_stats)
    
    return

def do_retrieval(pipeline: RAGPipeline, args):
    testset: list[dict] = load_annotated_qas(test_set_groups[args.testset], args.hypothetical)
    retrieval_acc = []
    for entry in tqdm(testset, desc="retrieving..."):
        if args.hypothetical:
            Q, A = entry['Q_aug'], entry['A']
        else:
            Q, A = entry['Q'], entry['A']
        retrieval_res = pipeline.retrieval_pass(Q)
        retrieval_acc.append({
            **entry,
            **retrieval_res,
        })
        
    output_dir = f'../experiment/{mk_retrieval_experiment_identifier(args)}'
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}/retrieval_acc.pkl', 'wb') as f:
        pickle.dump(retrieval_acc, f)
        
def mk_eval_experiment_identifier(args):
    return f"eval-{args.dataset}-{args.retriever}-{args.reranker}-{args.generator}-{args.testset}-hypothetical{args.hypothetical}{'-chunkaugmentation' if args.chunkaugmentation else ''}-k{args.k}"
        
def mk_retrieval_experiment_identifier(args):
    return f"{args.dataset}-{args.retriever}-{args.reranker}-{args.generator}-{args.testset}-hypothetical{args.hypothetical}{'-chunkaugmentation' if args.chunkaugmentation else ''}"

def mk_run_identifier(args): # deprecated
    timestamp = f"{datetime.datetime.now()}"
    return f"{timestamp}-{args.dataset}-{args.retriever}-{args.reranker}-{args.generator}-{args.testset}"
    
if __name__ == '__main__':
    args = gt_args()
    reranker = mk_reranker(args)
    generator = mk_generator(args)

    match args.mode:
        case 'index':
            dataset = mk_dataset(args)
            retriever = mk_retriever(args, dataset, True)
        case 'repl':
            dataset = mk_dataset(args)
            retriever = mk_retriever(args, dataset, False)
            pipeline = RAGPipeline(retriever, reranker, generator, args.k)
            qa_repl(pipeline)            
        case 'evaluate':
            do_evaluation(generator, args)
        case 'retrieve':
            dataset = mk_dataset(args)
            retriever = mk_retriever(args, dataset, False)
            pipeline = RAGPipeline(retriever, reranker, generator, args.k)
            do_retrieval(pipeline, args)