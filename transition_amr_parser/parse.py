# AMR parsing given a sentence and a model
import time
from datetime import timedelta
import os
import signal
import socket
import argparse
from collections import Counter, defaultdict
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
from transition_amr_parser.utils import yellow_font
from transition_amr_parser.io import (
    writer,
    read_sentences,
)
import transition_amr_parser.utils as utils
from transition_amr_parser.utils import print_log
import math

from transition_amr_parser.amr_parser import AMRParser
from transition_amr_parser.roberta_utils import extract_features_aligned_to_words_batched

# is_url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser')
    # Multiple input parameters
    parser.add_argument(
        "--in-sentences",
        help="file space with carriare return separated sentences",
        type=str
    )
    parser.add_argument(
        "--in-actions",
        help="file space with carriage return separated sentences",
        type=str
    )
    parser.add_argument(
        "--out-amr",
        help="parsing model",
        type=str
    )
    parser.add_argument(
        "--in-model",
        help="parsing model",
        type=str
    )
    parser.add_argument(
        "--model-config-path",
        help="Path to configuration of the model",
        type=str
    )
    # state machine rules
    parser.add_argument(
        "--action-rules-from-stats",
        help="Use oracle statistics to restrict possible actions",
        type=str
    )
    # Visualization arguments
    parser.add_argument(
        "--verbose",
        help="verbose mode",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--step-by-step",
        help="pause after each action",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--pause-time",
        help="time waited after each step, default is manual",
        type=int
    )
    parser.add_argument(
        "--clear-print",
        help="clear command line before each print",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--offset",
        help="start at given sentence number (starts at zero)",
        type=int
    )
    parser.add_argument(
        "--random-up-to",
        help="sample randomly from a max number",
        type=int
    )
    parser.add_argument(
        "--no-whitespace-in-actions",
        action='store_true',
        help="Assume whitespaces normalized to _ in PRED"
    )
    parser.add_argument(
        "--num-cores",
        default=1,
        help="number of cores to run on",
        type=int
    )
    parser.add_argument(
        "--use-gpu",
        help="Use GPU if true",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--add-root-token",
        help="Add root token (<ROOT>) at the end of the sentence.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size to compute roberta embeddings",
        default=12,
        type=int
    )
    parser.add_argument(
        "--parser-chunk-size",
        help="number of sentences for which to compute the parse at a time",
        default=5000,
        type=int
    )

    args = parser.parse_args()

    # Argument pre-processing
    if args.random_up_to:
        args.offset = np.random.randint(args.random_up_to)

    # force verbose
    if not args.verbose:
        args.verbose = bool(args.step_by_step)

    # Sanity checks
    assert args.in_sentences
    assert args.in_model
    assert args.action_rules_from_stats
    assert args.model_config_path

    return args


def reduce_counter(counts, reducer):
    """
    Returns a new counter from an existing one where keys have been mapped
    to in  many-to-one fashion and counts added
    """
    new_counts = Counter()
    for key, count in counts.items():
        new_key = reducer(key)
        new_counts[new_key] += count
    return new_counts
        
class DataHandler():
    def __init__(self, sentences, embeddings, cores):
        self.sentences_tuples = [(sent_idx, sentence) for sent_idx, sentence in enumerate(sentences)]
        self.cores = cores
        self.num_samples = int(math.ceil(len(self.sentences_tuples) * 1.0 / self.cores))
        self.embeddings = embeddings

    def get_sentences_for_rank(self, rank):
        return self.sentences_tuples[rank*self.num_samples:rank*self.num_samples+self.num_samples]

    def get_embeddings(self, sent_id):
        return self.embeddings[sent_id]

class Logger():

    def __init__(self, step_by_step=None, clear_print=None, pause_time=None,
                 verbose=False):

        self.step_by_step = step_by_step
        self.clear_print = clear_print
        self.pause_time = pause_time
        self.verbose = verbose or self.step_by_step

        if step_by_step:

            # Set traps for system signals to die graceful when Ctrl-C used

            def ordered_exit(signum, frame):
                """Mesage user when killing by signal"""
                print("\nStopped by user\n")
                exit(0)

            signal.signal(signal.SIGINT, ordered_exit)
            signal.signal(signal.SIGTERM, ordered_exit)

    def update(self, sent_idx, state_machine):

        if self.verbose:
            if self.clear_print:
                # clean screen each time
                os.system('clear')
            print(f'sentence {sent_idx}\n')
            print(state_machine)
            # step by step mode
            if self.step_by_step:
                if self.pause_time:
                    time.sleep(self.pause_time)
                else:
                    input('Press any key to continue')

def get_embeddings(model, sentences, batch_size):
    embeddings = {}
    data = [(sent_id, sentence) for sent_id, sentence in enumerate(sentences)]
    for i in range(0, math.ceil(len(data)/batch_size)):
        batch = data[i*batch_size : i*batch_size+batch_size]
        batch_indices = [item[0] for item in batch]
        batch_sentences = [item[1] for item in batch]
        batch_embeddings = extract_features_aligned_to_words_batched(model, sentences=batch_sentences, use_all_layers=True, return_all_hiddens=True)
        for index, features in zip(batch_indices, batch_embeddings):
            data_features = []
            for tok in features:
                if str(tok) not in ['<s>', '</s>']:
                    data_features.append(tok.vector)
            data_features = torch.stack(data_features).detach().cpu().numpy()
            embeddings[index] = data_features
    return embeddings

def worker(rank, model, handler, results, master_addr, master_port, cores):
    if cores > 1:
        torch.set_num_threads(1)
    print_log("Global: ","Threads :" + str(torch.get_num_threads()))
    print_log("Dist: ","starting rank:" + str(rank))
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    if cores > 1:
        dist.init_process_group(world_size=cores, backend='gloo', rank=rank)
        dist_module = torch.nn.parallel.DistributedDataParallelCPU(model)
        dist_model = dist_module.module
    else:
        dist_model = model
    
    data = handler.get_sentences_for_rank(rank)
    print_log("Data", f'Length of data is {len(data)} for rank {rank} ')
    for _, item in tqdm(enumerate(data)):
        sent_id = item[0]
        sentence = item[1]
        tokens = sentence.split()
        sent_rep = utils.vectorize_words(dist_model, tokens, training=False, gpu=dist_model.use_gpu)
        bert_emb = handler.get_embeddings(sent_id)
        amr = dist_model.parse_sentence(tokens, sent_rep, bert_emb)
        results[sent_id] = amr

    print("Dist: ","Finished worker for rank: " + str(rank))

def parse(sentences, parser, args):

    cores = args.num_cores

    start = time.time()
    embeddings = get_embeddings(parser.roberta, sentences, args.batch_size)
    end = time.time()
    print_log('parser', f'Time taken to get embeddings: {timedelta(seconds=float(end-start))}')

    # Make sure we got all the embeddings
    assert (len(embeddings) == len(sentences))

    # Create the distributed data handler
    handler = DataHandler(sentences, embeddings, cores)

    manager = multiprocessing.Manager()
    results = manager.dict()

    master_addr = socket.gethostname()
    master_port = '64646'

    arguments = (parser.model, handler, results, master_addr, master_port, cores)
    start = time.time()

    # Spawn multiple processes if user wants to run on multiple cores
    if cores > 1:
        mp.spawn(worker, nprocs=cores, args=arguments)
    else:
        worker(*(tuple([0]) + arguments))
    end = time.time()
    print_log('parser', f'Time taken to parse chunk: {timedelta(seconds=float(end-start))}')

    # Make sure we have processed all the sentences in the chunk
    assert (len(sentences) == len(results))

    amrs = []
    for i in range(0, len(sentences)):
        amrs.append(results[i])
    
    return amrs

def main():

    # Argument handling
    args = argument_parser()

    # Get num of cores to run on
    cores = args.num_cores

    # Get data
    sentences = read_sentences(args.in_sentences, add_root_token=args.add_root_token)

    # Initialize logger/printer
    logger = Logger(
        step_by_step=args.step_by_step,
        clear_print=args.clear_print,
        pause_time=args.pause_time,
        verbose=args.verbose
    )

    if args.num_cores > 1:
        model_use_gpu = False
    else:
        model_use_gpu = args.use_gpu

    # Load the parser
    parser = AMRParser(
                model_path=args.in_model,
                oracle_stats_path=args.action_rules_from_stats,
                config_path=args.model_config_path,
                model_use_gpu=model_use_gpu,
                roberta_use_gpu=args.use_gpu,
                logger=logger)

    amrs = []
    num_chunks = math.ceil(len(sentences)/args.parser_chunk_size)
    print_log('parser', f'Total number of chunks: {num_chunks}')

    start = time.time()
    for i in range(0,num_chunks):
        print_log('parser', f'Parsing chunk number: {i+1}')
        current_chunk = sentences[i*args.parser_chunk_size: i*args.parser_chunk_size+args.parser_chunk_size]
        current_amrs = parse(current_chunk, parser, args)
        amrs.extend(current_amrs)
    
    end = time.time()
    print_log('parser', f'Total time taken to parse sentences: {timedelta(seconds=float(end-start))}')

    # Make sure we have processed all the sentences
    assert (len(sentences)==len(amrs))

    if args.out_amr:
        # Get output AMR writer
        amr_write = writer(args.out_amr)
        
        # store output AMR
        for amr in amrs:
            amr_write(amr.toJAMRString())
        
        # close output AMR writer
        amr_write()
