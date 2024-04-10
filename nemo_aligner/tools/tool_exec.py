from typing import List, Dict
from queue import Queue
from dataclasses import dataclass, field
import logging

import numpy as np

@dataclass
class GenerationItem:
    data: Dict
    id: int
    completed: bool = False
    tool_data: Dict = field(default_factory=lambda:{})

class GenerationPipelineStage:
    def __init__(self, synchronous: bool = True):
        self.synchronous = synchronous
        self.work_queue = Queue()
        self.out_queue = Queue()
    
    def run_batch(self):
        pass
    
    def enqueue(self, inputs: List[GenerationItem]):
        """
        Checks if this stage should run the given inputs. 
        Returns inputs this stage does not apply to, runs on rest
        """
        pass
    
    def get_complete(self):
        pass
    
    def has_work(self):
        return self.work_queue.qsize() > 0
    
    def get_stop_words(self):
        pass
    
    
class GenerationPipelineRunner:
    def __init__(self, stages: List[GenerationPipelineStage]):
        """
        stages will be run in order
        """
        self.stages = stages

    def run_pipe(self, inputs: List):
        """
        inputs: List of arbitrary dictionaries
        """
        num_inputs = len(inputs)
        num_stages = len(self.stages)
        completed = []
        sync_stages = filter(lambda x: self.stages[x].synchronous, list(range(num_stages))) # get sync stage idxs
        current_sync_stage = 0
        current_stage = 0

        def _get_next_sync_stage(self, current):
            return sync_stages[(sync_stages.index(current) + 1) % len(sync_stages)]

        # Enqueue all inputs to stage 0
        self.stages[0].enqueue([GenerationItem(data=input, id=idx) for idx, input in enumerate(inputs)])

        while len(completed) < num_inputs:
            # Check all out queues to advance them to the next appropriate stage
            for stage_idx in range(len(self.stages)):
                stage_complete = self.stages[stage_idx].get_complete()
                completed += list(filter(lambda x: x.completed, stage_complete))
                logging.warning(f'completed {len(completed)}')

                to_queue = list(filter(lambda x: not x.completed, stage_complete))
                for offset in range(num_stages):
                    logging.warning(f'to_queue {len(to_queue)}')
                    to_queue = self.stages[(stage_idx + offset + 1) % num_stages].enqueue(to_queue)
                    completed += list(filter(lambda x: x.completed, to_queue))
                    to_queue = list(filter(lambda x: not x.completed, to_queue))

                logging.warning(f'post queuing completed {len(completed)}, to_queue {len(to_queue)}')
                
                assert len(to_queue) == 0, f"ERROR: there are {len(to_queue)} items that no \
                                            stage can work on and are still marked incomplete"
            
            # for _ in range(len(sync_stages)):
                # if not self.stages[current_sync_stage].has_work():
                    # current_sync_stage = _get_next_sync_stage(current_sync_stage)
                # else:
                    # self.stages[current_sync_stage].run_batch()
            #         break
            for s in range(num_stages):
                if self.stages[(current_stage + s) % num_stages].has_work():
                    logging.warning(f"stage {current_stage + s} has work")
                    self.stages[(current_stage + s) % num_stages].run_batch()
                    # current_stage = (current_stage + s) % num_stages
                    break
                else:
                    logging.warning(f"stage {current_stage + s} has no work")

        completed.sort(key=lambda x: x.id)
        return [c.data for c in completed]

def merge_stop_words(stop_word_lists_processed: List[np.array]):
    """
    Takes in a list of batch-size 1 (no leading axis) stop-word
    tensors preprocessed for TRT-LLM and merges them together.
    Assumes the ids are not padded.
    """
    flat_ids = []
    lens = []
    for stop_tensor in stop_word_lists_processed:
        item_flat_ids = stop_tensor[0]
        item_offsets = stop_tensor[1]
        
        flat_ids += item_flat_ids.tolist()
        
        # need to undo the cumsum for merging
        prev_offset = 0
        for offset in item_offsets:
            if offset != -1:
                lens.append(offset - prev_offset)
                prev_offset = offset

    flat_ids = np.array(flat_ids)
    # redo cumsum
    offsets = np.cumsum(np.array(lens))

    # redo pad 
    offsets = np.pad(offsets, (0, len(flat_ids) - len(offsets)), constant_values = -1)
    
    stop_words = np.array([flat_ids, offsets], dtype="int32")
    return stop_words
        
            

