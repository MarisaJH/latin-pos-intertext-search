import pickle
from statistics import mean
import glob
import sys
model_type = 'constraints=target-morph-bert'
indices = [
    (0,330),
    (330,660),
    (660,944)
]
filenames = []

for index in indices:
    file = glob.glob(f'./temp/*-results-start-{index[0]}-end-{index[1]}-{model_type}.p')[0]
    filenames.append(file)

results_ = []
for filename in filenames:
    print(f'Loading {filename}')
    with open(filename, 'rb') as f:
        results_.extend(pickle.load(f))
print('Results length:', len(results_))

# %%
# Get ranks

ranks = [len(result[1]) for result in results_ if len(result[1]) != 0]

# %%
# Compute recall & precision at k; computer MRR

ks = [1, 3, 5, 10, 25, 50, 75, 100, 250]

def recall_at_k(ranks, k):
    n = len([rank for rank in ranks if rank <= k])
    d = len(ranks)
    recall = n/d
    return recall

def precision_at_k(ranks, k):
    n = len([rank for rank in ranks if rank <= k])
    d = sum([rank if rank<=k else k for rank in ranks])
    precision = n/d
    return precision

def mrr(ranks):
    return mean([1/item for item in ranks])

print(f'MRR: {mrr(ranks)}')
print()

print(f'Checking the following values for k {ks}\n')
for k in ks:
    print(f'\tRecall at k={k}: {recall_at_k(ranks, k)}')
    print(f'\tPrecision at k={k}: {precision_at_k(ranks, k)}')
    print()