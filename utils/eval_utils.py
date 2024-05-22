import time
import numpy as np
import pandas as pd

class Runner:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round

    def run(self, landscape, starting_sequence, model, explorer):
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        self.update_results(0, [starting_sequence], [starting_fitness])
        
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()

            model.train(self.sequence_buffer, self.fitness_buffer)
            sequences, model_scores = explorer.propose_sequences(self.results)
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)

            round_running_time = time.time()-round_start_time
            self.update_results(round, sequences, true_scores, round_running_time)
    
    def update_results(self, round, sequences, true_scores, running_time=0.0):
        self.results = self.results.append(
            pd.DataFrame({
                "round": round,
                "sequence": sequences,
                "measured_score": true_scores
            })
        )
        print('round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec)'.format(round, self.results['measured_score'].max(), running_time))
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['measured_score'].to_numpy()
    
class RunnerBinarized:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round

    def run(self, landscape, starting_sequence, model, explorer):
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        self.update_results(0, [starting_sequence], [starting_fitness], [0])
        
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()

            if round > 1:
                model.train(self.sequence_buffer, self.fitness_buffer)
            

            # TODO: Currently we measure each sequence _once_ and so only get a label of 1 or 0
            #       However there should in reality be some coverage and some noise, such that
            #       better sequences have a higher probability of being in the 1 class and 
            #       worse have a higher probability of being in the 0 class.


            sequences, model_scores = explorer.propose_sequences(self.results)
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)
            measured_scores = np.zeros(len(true_scores), dtype=int)
            threshold = np.percentile(true_scores, 80)
            measured_scores[true_scores>=threshold] = 1

            round_running_time = time.time()-round_start_time
            self.update_results(round, sequences, true_scores, measured_scores, round_running_time)
    
    def update_results(self, round, sequences, true_scores, measured_scores, running_time=0.0):
        self.results = self.results.append(
            pd.DataFrame({
                "round": round,
                "sequence": sequences,
                "measured_score" : measured_scores,
                "true_score": true_scores
            })
        )
        print('round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec)'.format(round, self.results['true_score'].max(), running_time))
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['measured_score'].to_numpy()