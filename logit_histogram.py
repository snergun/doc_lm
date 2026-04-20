import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
model_path = "/work/hdd/bcyi/eergun/doc_lm/trainedmodel/ptb/additional_finetuned"
k= 1024
n_samples = 1000

results_path = os.path.join(model_path, "results") 
full_logits = np.load(os.path.join(results_path, "validation_full_prob.npy"), mmap_mode="r")

top_k_logits = []
for logits in tqdm(full_logits):
    top_k_logits.append(np.partition(logits, -k)[-k:])
top_k_logits = np.array(top_k_logits)
# Get histogram of top-k logits
histogram, bin_edges = np.histogram(top_k_logits, bins=50)
# Plot histogram and save it
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor='black')
plt.title(f'Histogram of Top-{k} Logits')
plt.xlabel('Logit Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig(os.path.join(results_path, f"top_{k}_logits_histogram.png"))
plt.show()

# Get histogram of all logits with sampling
sampled_logits = full_logits[:n_samples].copy()
histogram_all, bin_edges_all = np.histogram(sampled_logits.flatten(), bins=50)
# Plot histogram and save it
plt.figure(figsize=(10, 6))
plt.bar(bin_edges_all[:-1], histogram_all, width=np.diff(bin_edges_all), edgecolor='black')
plt.title('Histogram of All Logits (Sampled)')
plt.xlabel('Logit Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig(os.path.join(results_path, "all_logits_histogram_sampled.png"))
plt.show()