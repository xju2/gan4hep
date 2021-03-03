
import time
import os
import itertools
import re
import numpy as np
from typing import Optional
from multiprocessing import Pool
from functools import partial

import tensorflow as tf
from graph_nets import utils_tf


n_node_features = 6
n_max_nodes = 2 # maximum number of out-going particles

class HerwigHadrons(object):
    def __init__(self, with_padding=False, n_graphs_per_evt=1):
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_padding = with_padding
        self.n_files_saved = 0
        self.graphs = []
        self.n_graphs_per_evt = n_graphs_per_evt
        self.n_evts = 0

    def read(filename):
        with open(filename, 'r') as f:
            for line in f:
                yield line[:-1]

    def _num_evts(self, filename: str) -> int:
        """
        return total number of events in the filename
        """
        return sum([1 for _ in self.read(filename)])

    def make_graph(event, debug=False):
        
        particles = event[:-2].split(';')
        
        is_light = True
        array = []
        node_dict = {}
        node_idx = 0
        root_uid = 0
        for ipar, par in enumerate(particles):
            items = par.split(',')
            if ipar == 0:
                is_light = items[0] == 'L'
                items = items[1:]

            if len(items) < 2:
                continue
                
            uid = int(re.search('([0-9]+)', items[0]).group(0))
            if ipar == 0:
                root_uid = uid

            node_dict[uid] = node_idx
            node_idx += 1
            pdgid = [int(items[1])]
            children, idx = ([int(x) for x in re.search('[0-9]+ [0-9]*', items[2]).group(0).strip().split(' ')],3) if '[' in items[2] else ([-1, -1], 2)
            # WARN: assuming the number of decay products could be 0, 1 and 2
            if len(children) == 1:
                children += [-1]
            try:
                momentum = [float(x) for x in items[idx:]]
            except ValueError:
                print(particles)
                print("missing 4vec info, skipped")
                return [(None, None)]
            all_info = [uid] + children + pdgid + momentum
            array.append(all_info)

        # rows: particles, 
        # columns: uid, child1, child2, pdgid, 4-momentum
        array = np.array(array)
        nodes = array[:, 4:].astype(np.float32)
        n_nodes = nodes.shape[0] - 1

        if n_nodes > n_max_nodes:
            print("cluster decays to more than {} nodes".format(n_max_nodes))
            return [(None, None)]

        if n_nodes < n_max_nodes:
            nodes = np.concatenate([nodes, np.zeros([n_max_nodes-n_nodes, nodes.shape[1]], dtype=np.float32)], axis=0)

        n_nodes = nodes.shape[0]
        senders = np.concatenate([array[array[:, 1] > 0, 0].astype(np.int32), array[array[:, 2] > 0, 0].astype(np.int32)])
        receivers = np.concatenate([array[array[:, 1] > 0, 1].astype(np.int32), array[array[:, 2] > 0, 2].astype(np.int32)])

        zero = np.array([0], dtype=np.float32)
        
        # # convert node id to [0, xxx]
        # # remove root id in the edges
        # senders = np.array([node_dict[i] for i in senders], dtype=np.int32)
        # receivers = np.array([node_dict[j] for i,j in zip(senders,receivers)], dtype=np.int32)
        # n_edges = senders.shape[0]
        # edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)


        # use fully connected graph
        all_edges = list(itertools.combinations(range(n_nodes), 2))
        senders = np.array([x[0] for x in all_edges])
        receivers = np.array([x[1] for x in all_edges])
        all_senders = np.concatenate([senders, receivers], axis=0)
        all_receivers = np.concatenate([receivers, senders], axis=0)
        n_edges = len(all_edges*2)
        edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)

        input_datadict = {
            "n_node": 1,
            "n_edge": 1,
            "nodes": nodes[0:1, :],
            "edges": np.expand_dims(np.array([0.], dtype=np.float32), axis=1),
            "senders":zero,
            "receivers": zero,
            "globals": zero,
        }
        target_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": all_senders,
            "receivers": all_receivers,
            # "globals": np.array([1]*(n_nodes-1)+[0]*(max_nodes-n_nodes+1), dtype=np.float32)
            "globals": zero,
        }
        # print("input: ", input_datadict)
        # print("target:", target_datadict)
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        # padding the graph if number of nodes is less than max-nodes??
        # not sure it is nessary..

        return [(input_graph, target_graph)]

    def subprocess(self, ijob, n_evts_per_record, filename, outname, debug):
        outname = "{}_{}.tfrec".format(outname, ijob)
        if os.path.exists(outname):
            print(outname,"is there. skip...")
            return 0, n_evts_per_record

        ievt = -1
        ifailed = 0
        all_graphs = []
        start_entry = ijob * n_evts_per_record
        for event in self.read(filename):
            ievt += 1
            if ievt < start_entry:
                continue
            
            gen_graphs = self.make_graph(event, debug)
            if gen_graphs[0][0] is None:
                ifailed += 1
                continue

            all_graphs += gen_graphs
            if ievt == start_entry + n_evts_per_record - 1:
                break
        
        isaved = len(all_graphs)
        ex_input, ex_target = all_graphs[0]
        input_dtype, input_shape = graph.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        target_dtype, target_shape = graph.dtype_shape_from_graphs_tuple(
            ex_target, with_padding=self.with_padding)
        def generator():
            for G in all_graphs:
                yield (G[0], G[1])

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(input_dtype, target_dtype),
            output_shapes=(input_shape, target_shape),
            args=None)

        writer = tf.io.TFRecordWriter(outname)
        for data in dataset:
            example = graph.serialize_graph(*data)
            writer.write(example)
        writer.close()
        return ifailed, isaved
        

    def process(self, filename, outname, n_evts_per_record, debug, max_evts, num_workers=1, **kwargs):
        now = time.time()

        all_evts = self._num_evts(filename)
        all_evts = max_evts if max_evts > 0 and all_evts > max_evts else all_evts

        n_files = all_evts // n_evts_per_record
        if all_evts%n_evts_per_record > 0:
            n_files += 1

        print("In total {:,} events, write to {:,} files with {:,} workers".format(all_evts, n_files, num_workers))
        with Pool(num_workers) as p:
            process_fnc = partial(self.subprocess,
                        n_evts_per_record=n_evts_per_record,
                        filename=filename,
                        outname=outname,
                        debug=debug)
            res = p.map(process_fnc, list(range(n_files)))

        ifailed = sum([x[0] for x in res])
        isaved = sum([x[1] for x in res])
            
        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            isaved, read_time/60.))
        print("{:,} events failed in being converted to graph".format(ifailed))