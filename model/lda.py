import torch
import torch.nn as nn
import numpy as np
from . import register_model
from utils.seq_utils import sequences_to_tensor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

@register_model("lda")
class LDAModel():
    
    def __init__(self, args, alphabet, starting_sequence, **kwargs):
        self.alphabet = alphabet
        self.lda = LinearDiscriminantAnalysis()
        self.is_fitted = False

    def train(self, sequences, labels):
        mutation_sets = sequences_to_tensor(sequences, self.alphabet).reshape(len(sequences), -1)
        self.lda.fit(mutation_sets, labels)
        self.is_fitted = True

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]
        
        mutation_sets = sequences_to_tensor(sequences, self.alphabet).reshape(len(sequences), -1)

        if self.is_fitted:
            predictions = self.lda.predict(mutation_sets)
        else:
            predictions = np.zeros(len(sequences))
            print("Warning: LDA model is not fitted yet. Returning zero predictions.")

        return predictions
