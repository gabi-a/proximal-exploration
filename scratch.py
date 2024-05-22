#%%
import torch
import torch.nn as nn

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
# # Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()


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
sns.histplot(single_mutant_fitness, bins=20)
plt.vlines(landscape.get_fitness([starting_sequence]), 0, 10, color='red', label='Starting sequence')
plt.xlabel('Fitness')
plt.ylim(0, 10)
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

#%%
# Train an LDA one-hot model to predict new sequences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

def one_hot_encode(alphabet, sequences):
    X = np.array([list(map(alphabet.index, seq)) for seq in sequences])
    X = np.eye(len(alphabet))[X].reshape(len(sequences), -1)
    return X

def one_hot_decode(alphabet, encoded):
    X = encoded.reshape(len(encoded), len(starting_sequence), len(alphabet))
    X = np.array([''.join([alphabet[np.argmax(aa)] for aa in x]) for x in X])
    return X

#%%
X = one_hot_encode(alphabet, single_mutants)
y = np.array(single_mutant_fitness) > np.percentile(single_mutant_fitness, 90)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)

print(balanced_accuracy_score(y_test, lda.predict(X_test)))

#%%
# Design new sequence using the weights of the LDA model
# designed_variant = ''.join([alphabet[i] for i in lda.coef_.reshape(len(starting_sequence), len(alphabet)).argmax(axis=1)])
# print(landscape.get_fitness([designed_variant]))

params = lda.coef_.copy()[0]

# Set the coefs of _no mutation_ to be much lower to favour making mutations
# Found by trial and error that 0.01 gives the best designed_variant score
params = params.reshape(len(starting_sequence), len(alphabet))
for i in range(len(starting_sequence)):
    params[i, alphabet.index(starting_sequence[i])] *= 0.01
params = params.reshape(1, -1)

designed_variant = one_hot_decode(alphabet, params)[0]
print(landscape.get_fitness([designed_variant]))

#%%
muts = lambda designed_variant: [f'{aa1}{i}{aa2}' for aa1, i, aa2 in zip(starting_sequence, range(len(starting_sequence)), designed_variant) if aa1 != aa2]
print(f"Made {len(muts(designed_variant))} mutations.")

# %%
# Again but using double mutant data
all_mutants = single_mutants + list(sampled_double_mutants)
all_fitnesses = np.concatenate([single_mutant_fitness, double_mutant_fitness])
X = one_hot_encode(alphabet, all_mutants)

y = np.array(all_fitnesses) > np.percentile(all_fitnesses, 90)
print("WARNING WARNING WARNING")
print("Currently using only the top 10% of fitnesses as the positive class")
print("WARNING WARNING WARNING")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from imblearn.over_sampling import RandomOverSampler

X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)

print(balanced_accuracy_score(y_test, lda.predict(X_test)))

#%%
params = lda.coef_.copy()[0]

# Now it seems we have to make the coefs of _no mutation_ higher 
# to favour NOT making mutations??
params = params.reshape(len(starting_sequence), len(alphabet))
for i in range(len(starting_sequence)):
    params[i, alphabet.index(starting_sequence[i])] += 70
params = params.reshape(1, -1)

designed_variant = one_hot_decode(alphabet, params)[0]

print('Initial', landscape.get_fitness([starting_sequence])[0])
print('Designed', landscape.get_fitness([designed_variant])[0])

print(f"Made {len(muts(designed_variant))} mutations.")

# NOTES
# So far, this approach is not great
# because the best screened single mutant has a fitness of >3.5
# but the designed variants have fitness ~1.5 (which is greater than the starting sequence!)

#%%

# get the sequence of the best single mutant
best_single_mutant = single_mutants[np.argmax(single_mutant_fitness)]

print(best_single_mutant)
print(landscape.get_fitness([best_single_mutant]))

X_starting = one_hot_encode(alphabet, [starting_sequence])
X_best = one_hot_encode(alphabet, [best_single_mutant])

print(lda.coef_ @ X_starting.T + lda.intercept_)
print(lda.coef_ @ X_best.T + lda.intercept_)

#%%
def continuos_fitness_prediction(lda, sequence):
    X = one_hot_encode(alphabet, [sequence])
    return lda.coef_ @ X.T + lda.intercept_

#%%
# Simulated annealing starting from the initial sequence
# and using the LDA model to predict the fitness of new sequences

def mutate_sequence(sequence):
    # Perform a single random mutation
    idx_mutate = np.random.randint(len(sequence))
    new_aa = alphabet[np.random.randint(0, len(alphabet))]
    new_sequence = sequence[:idx_mutate] + new_aa + sequence[idx_mutate+1:]
    return new_sequence

def mutate_sequence_conservative(sequence, starting_sequence=starting_sequence):
    _starting_sequence = list(starting_sequence)
    _sequence = list(sequence)
    idxs = [i for i in range(len(_starting_sequence)) if _starting_sequence[i] != _sequence[i]]
    if len(idxs) == 0:
        idx_mutate = np.random.randint(len(sequence))
    else:
        if np.random.rand() < 0.75:
            idx_mutate = np.random.choice(idxs)
        else:
            idx_mutate = np.random.randint(len(sequence))
            print("New mutation")
    new_aa = alphabet[np.random.randint(0, len(alphabet))]
    new_sequence = sequence[:idx_mutate] + new_aa + sequence[idx_mutate+1:]
    return new_sequence

def mutate_sequence_orthadox(sequence, starting_sequence=starting_sequence):
    _starting_sequence = list(starting_sequence)
    _sequence = list(sequence)
    idxs = [i for i in range(len(_starting_sequence)) if _starting_sequence[i] != _sequence[i]]
    if len(idxs) == 0:
        return mutate_sequence(sequence)
    elif len(idxs) < 3 and np.random.rand() < 0.01:
        return mutate_sequence(sequence)
    else:
        existing_mutation = np.random.choice(idxs)
        new_aa = alphabet[np.random.randint(0, len(alphabet))]
        probs = np.exp(-((np.arange(len(sequence)) - existing_mutation) ** 2) / 2)
        probs /= probs.sum()
        mutation_idx = np.random.choice(len(sequence), p=probs)
        new_sequence = starting_sequence[:mutation_idx] + new_aa + starting_sequence[mutation_idx+1:]
        return new_sequence
    
#%%
## This doesn't improve things
#
# import blosum as bl
# matrix = bl.BLOSUM(62)

# def mutate_sequence_blosum62(sequence):
#     # Perform a single random mutation using the BLOSUM62 matrix for the probabilities
#     idx_mutate = np.random.randint(len(sequence))
    
#     # Get the probabilities of each amino acid mutation
#     probs = np.exp([matrix[sequence[idx_mutate]][k] for k in alphabet])
#     probs /= probs.sum()
#     new_aa = np.random.choice(list(alphabet), p=probs)

#     new_sequence = sequence[:idx_mutate] + new_aa + sequence[idx_mutate+1:]
#     return new_sequence

#%%
def calculate_acceptance_probability(current_fitness, candidate_fitness, temperature):
    # Calculate the acceptance probability based on the fitness difference and temperature
    if candidate_fitness > current_fitness:
        return 1
    else:
        return np.exp((candidate_fitness - current_fitness) / temperature)

def run_simulated_annealing(
    lda,
    starting_sequence,
    temperature=1.0,
    cooling_rate=0.01,
    num_iterations=1000,
    mutate_fn=mutate_sequence
):
    current_sequence = starting_sequence
    current_fitness = continuos_fitness_prediction(lda, current_sequence).squeeze()
    best_sequence = current_sequence
    best_fitness = current_fitness
    
    for i in range(num_iterations):
        
        # if i % 100 == 0:
        #     print(f"Current fitness (oracle): {landscape.get_fitness([current_sequence])[0]:.3f}")
        #     print(f"Current fitness (LDA): {current_fitness:.3f}")
        
        # Generate a new candidate sequence by making a random mutation
        candidate_sequence = mutate_fn(current_sequence)
        candidate_fitness = continuos_fitness_prediction(lda, candidate_sequence).squeeze()
    
        # Calculate the acceptance probability based on the fitness difference
        acceptance_probability = calculate_acceptance_probability(current_fitness, candidate_fitness, temperature)
        
        # Accept the candidate sequence with a certain probability
        if acceptance_probability >= np.random.rand():
            # print('ap', acceptance_probability)
            # print('temp', temperature)
            # print('cf', current_fitness)
            # print('cs', candidate_fitness)
            current_sequence = candidate_sequence
            current_fitness = candidate_fitness
        
        # Update the best sequence and fitness if necessary
        if candidate_fitness > best_fitness:
            best_sequence = candidate_sequence
            best_fitness = candidate_fitness
        
        # Cool down the temperature
        temperature *= (1 - cooling_rate)
    
    return best_sequence, best_fitness

# Call the simulated annealing function with the necessary parameters
np.random.seed(0)

# Pick one of the sequences in the top 10% of fitness to start with
# start_from = starting_sequence


##
## Result of running the code below is quite interesting:
## Essentially the model can most of the time improve upon
## sequences that havea lowish fitness that is near the threshold value.
## BUT 
## It usually makes _worse_ sequences from ones that are very good!! 
## This is somewhat addressed by mutate_sequence_orthadox which
## only allows up to 2 total mutations from the starting sequence
##

#%%
from tqdm import tqdm

designed_sequences = []

dups = 0

for start_from in tqdm([starting_sequence] + list(np.array(all_mutants)[y])):

    best_sequence, best_fitness = run_simulated_annealing(
        lda, 
        starting_sequence=start_from, 
        temperature=1.0, 
        cooling_rate=0.01, 
        num_iterations=1000,
        mutate_fn=mutate_sequence_orthadox)
    
    if best_sequence in list(np.array(all_mutants)[y]):
        dups += 1
        continue

    if continuos_fitness_prediction(lda, best_sequence) > continuos_fitness_prediction(lda, start_from):
        designed_sequences.append(best_sequence)

    # print(muts(best_sequence))
    # print(f"{landscape.get_fitness([start_from])[0]:.3f} -> {landscape.get_fitness([best_sequence])[0]:.3f}")
    # print()

    # print("Original fitness:", landscape.get_fitness([starting_sequence])[0])
    # print("Starting fitness:", landscape.get_fitness([start_from])[0])
    # print("Best sequence:", best_sequence)
    # print("Best fitness (oracle):", landscape.get_fitness([best_sequence])[0])
    # print("Best fitness:", best_fitness)

designed_fitnesses = landscape.get_fitness(designed_sequences)

#%%
starting_sequences = [starting_sequence] + list(np.array(all_mutants)[y])
starting_fitnesses = landscape.get_fitness(starting_sequences)

# %%

print(np.mean(starting_fitnesses))
print(np.mean(designed_fitnesses))

plt.figure()
bins = np.linspace(0, 5, 100)
sns.histplot(single_mutant_fitness, bins=bins, label='Single mutants', stat='density', alpha=0.5)
sns.histplot(double_mutant_fitness, bins=bins, label='Double mutants', stat='density', alpha=0.5)
sns.histplot(starting_fitnesses, bins=bins, label='Starting sequences', stat='density', alpha=0.5)
sns.histplot(designed_fitnesses, bins=bins, label='Designed sequences', stat='density', alpha=0.5)
plt.ylim(0, 0.5)
plt.xlim(1, 4)
plt.xlabel('Fitness')
plt.legend()
plt.show()
# %%
