#%%
import numpy as np
from landscape import get_landscape, task_collection, landscape_collection
from algorithm import get_algorithm, algorithm_collection
from model import get_model, model_collection
from model.ensemble import ensemble_rules
from utils.os_utils import get_arg_parser
from utils.eval_utils import Runner

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

def get_args():
    parser = get_arg_parser()
    
    parser.add_argument('--device', help='device', type=str, default='cuda')

    # landscape arguments
    parser.add_argument('--task', help='fitness landscape', type=str, default='avGFP', choices=task_collection.keys())
    parser.add_argument('--oracle_model', help='oracle model of fitness landscape', type=str, default='tape', choices=landscape_collection.keys())

    # for jupyter
    parser.add_argument('--f', help='ipykernel hack', type=str, default='')

    args = parser.parse_args()
    return args

#%%
# if __name__=='__main__':
args = get_args()

landscape, alphabet, starting_sequence = get_landscape(args)

# Create a single site saturation mutagenesis landscape by mutating each position in the starting sequence to each of the 20 amino acids in the alphabet
def single_site_saturation_mutagenesis(alphabet, starting_sequence):
    mutants = []
    for i in range(len(starting_sequence)):
        for aa in alphabet:
            mutant = starting_sequence[:i] + aa + starting_sequence[i+1:]
            mutants.append(mutant)
    return mutants

single_mutants = single_site_saturation_mutagenesis(alphabet, starting_sequence)
single_mutant_fitness = landscape.get_fitness(single_mutants)
single_mutant_fitness = np.array(single_mutant_fitness)

#%%
plt.figure()
sns.histplot(single_mutant_fitness[single_mutant_fitness > 1.5], bins=20)
plt.vlines(landscape.get_fitness([starting_sequence]), 0, 10, color='red', label='Starting sequence')
plt.xlabel('Fitness')
plt.show()

#%%
# Take the top 10% of mutants and perform double site saturation mutagenesis
top_10_percent = np.percentile(single_mutant_fitness, 90)
top_10_percent_mutants = [mutant for mutant, fit in zip(single_mutants, single_mutant_fitness) if fit > top_10_percent]

double_mutants = []
for mutant in single_mutants:
    double_mutants.extend(single_site_saturation_mutagenesis(alphabet, mutant))

print("Total number of double mutants:", len(double_mutants))

# %%
# Take a random sample of the double mutants
sample_size = 10000
sampled_double_mutants = np.random.choice(double_mutants, sample_size, replace=False)

double_mutant_fitness = landscape.get_fitness(sampled_double_mutants)
double_mutant_fitness = np.array(double_mutant_fitness)

# %%
plt.figure()
sns.histplot(single_mutant_fitness[single_mutant_fitness > 1.5], bins=20, label='Single mutants')
sns.histplot(double_mutant_fitness[double_mutant_fitness > 1.5], bins=20, label='Double mutants')
plt.vlines(landscape.get_fitness([starting_sequence]), 0, 10, color='red', label='Starting sequence')
plt.xlabel('Fitness')
plt.legend()
plt.show()

# %%
print(max(single_mutant_fitness))
print(max(double_mutant_fitness))

#%%
from typing import List

def run_experiment(sequences: List[str], true_fitnesses: np.ndarray):

    idxs = np.arange(len(sequences))

    # Randomly oversample by some factor
    # Note this does not _guarantee_ that every sequence is sampled even once
    # so there may be some sequences that are not in the final dataset
    oversample_factor = 10
    idxs = np.random.choice(idxs, np.floor(oversample_factor * len(idxs)).astype(int), replace=True)

    # Add biological noise to the true fitness
    # TODO: Could make the noise depend on the true fitness, e.g. higher noise for higher fitness
    biological_noise = np.random.normal(0, 0.1, len(idxs))
    noisy_fitnesses = true_fitnesses[idxs] + biological_noise

    # Add FACS measurement noise
    FACS_noise = np.random.normal(0, 0.1, len(idxs))
    FACS_measurements = noisy_fitnesses + FACS_noise

    # Add Mother Machine measurement noise - get to measure multiple times
    MM_measurements = []
    for i in range(5):
        MM_noise = np.random.normal(0, 0.1, len(idxs))
        MM_measurements.extend(noisy_fitnesses + MM_noise)

    return idxs, FACS_measurements, MM_measurements

# %%
plt.figure()
idxs, FACS_data, MM_data = run_experiment(single_mutants, single_mutant_fitness)
plt.scatter(FACS_data, np.array(MM_data).reshape(5, -1).mean(axis=0))
plt.xlabel('FACS')
plt.ylabel('Mother Machine (averaged)')
plt.show()

# %%
plt.figure()
for i in range(5):
    plt.scatter(FACS_data, np.array(MM_data).reshape(5, -1)[i])
plt.xlabel('FACS')
plt.ylabel('Mother Machine')
plt.show()
