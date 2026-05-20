
import pandas as pd
from itertools import combinations
import math

INPUT = "coded_ideas.csv"

df = pd.read_csv(INPUT)

required = ["prompt_id", "provider", "model", "run_id", "canonical_idea", "cluster_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["prompt_id"] = df["prompt_id"].astype(str)
df["provider"] = df["provider"].astype(str)
df["model"] = df["model"].astype(str)
df["cluster_id"] = df["cluster_id"].astype(str)
df["canonical_idea"] = df["canonical_idea"].astype(str)

def entropy(labels):
    counts = labels.value_counts()
    total = counts.sum()
    h = 0
    for count in counts:
        p = count / total
        h -= p * math.log2(p)
    return h

# ----------------------------
# 1. Intra-model repetition
# ----------------------------

intra_results = []

for (prompt_id, model), group in df.groupby(["prompt_id", "model"]):
    total = len(group)
    unique_clusters = group["cluster_id"].nunique()
    repetition_rate = 1 - unique_clusters / total if total else 0
    top_cluster_share = group["cluster_id"].value_counts(normalize=True).iloc[0]

    top_cluster = group["cluster_id"].value_counts().index[0]
    top_cluster_count = group["cluster_id"].value_counts().iloc[0]

    intra_results.append({
        "prompt_id": prompt_id,
        "model": model,
        "total_ideas": total,
        "unique_clusters": unique_clusters,
        "intra_model_repetition_rate": repetition_rate,
        "top_cluster": top_cluster,
        "top_cluster_count": top_cluster_count,
        "top_cluster_share": top_cluster_share
    })

intra_df = pd.DataFrame(intra_results)
intra_df.to_csv("intra_model_repetition.csv", index=False)

# ----------------------------
# 2. Inter-model similarity
# ----------------------------

inter_results = []

for prompt_id, group in df.groupby("prompt_id"):
    models = sorted(group["model"].unique())

    for model_a, model_b in combinations(models, 2):
        clusters_a = set(group[group["model"] == model_a]["cluster_id"])
        clusters_b = set(group[group["model"] == model_b]["cluster_id"])

        intersection = clusters_a & clusters_b
        union = clusters_a | clusters_b

        jaccard = len(intersection) / len(union) if union else 0

        inter_results.append({
            "prompt_id": prompt_id,
            "model_a": model_a,
            "model_b": model_b,
            "shared_clusters": len(intersection),
            "total_clusters_pair": len(union),
            "jaccard_similarity": jaccard,
            "shared_cluster_names": "; ".join(sorted(intersection))
        })

inter_df = pd.DataFrame(inter_results)
inter_df.to_csv("inter_model_similarity.csv", index=False)

# ----------------------------
# 3. Prompt-level diversity
# ----------------------------

prompt_results = []

for prompt_id, group in df.groupby("prompt_id"):
    total = len(group)
    unique_clusters = group["cluster_id"].nunique()
    h = entropy(group["cluster_id"])
    max_h = math.log2(unique_clusters) if unique_clusters > 1 else 0
    normalized_entropy = h / max_h if max_h > 0 else 0

    top_cluster_counts = group["cluster_id"].value_counts()
    top_cluster = top_cluster_counts.index[0]
    top_cluster_count = top_cluster_counts.iloc[0]
    top_cluster_share = top_cluster_count / total if total else 0

    top_clusters = top_cluster_counts.head(5).to_dict()

    prompt_results.append({
        "prompt_id": prompt_id,
        "total_ideas": total,
        "unique_clusters": unique_clusters,
        "diversity_ratio": unique_clusters / total if total else 0,
        "idea_entropy": h,
        "normalized_idea_entropy": normalized_entropy,
        "top_cluster": top_cluster,
        "top_cluster_count": top_cluster_count,
        "top_cluster_share": top_cluster_share,
        "top_5_clusters": str(top_clusters)
    })

prompt_df = pd.DataFrame(prompt_results)
prompt_df.to_csv("prompt_level_diversity.csv", index=False)

# ----------------------------
# 4. Overall summary numbers
# ----------------------------

total_ideas = len(df)
total_prompts = df["prompt_id"].nunique()
total_models = df["model"].nunique()
total_clusters = df.groupby("prompt_id")["cluster_id"].nunique().sum()

overall_diversity_ratio = total_clusters / total_ideas if total_ideas else 0
overall_intra_repetition = intra_df["intra_model_repetition_rate"].mean()
overall_top_cluster_share = prompt_df["top_cluster_share"].mean()
overall_inter_similarity = inter_df["jaccard_similarity"].mean()

highest_intra = intra_df.sort_values("intra_model_repetition_rate", ascending=False).iloc[0]
lowest_diversity_prompt = prompt_df.sort_values("diversity_ratio", ascending=True).iloc[0]
highest_inter_pair = inter_df.sort_values("jaccard_similarity", ascending=False).iloc[0]

# Find common clusters across providers per prompt
provider_overlap_rows = []

for prompt_id, group in df.groupby("prompt_id"):
    provider_clusters = {}
    for provider, pgroup in group.groupby("provider"):
        provider_clusters[provider] = set(pgroup["cluster_id"])

    if len(provider_clusters) >= 2:
        common = set.intersection(*provider_clusters.values())
        union = set.union(*provider_clusters.values())
        overlap_rate = len(common) / len(union) if union else 0

        provider_overlap_rows.append({
            "prompt_id": prompt_id,
            "providers": ", ".join(sorted(provider_clusters.keys())),
            "common_clusters_across_all_providers": len(common),
            "total_provider_union_clusters": len(union),
            "provider_overlap_rate": overlap_rate,
            "common_cluster_names": "; ".join(sorted(common))
        })

provider_overlap_df = pd.DataFrame(provider_overlap_rows)
provider_overlap_df.to_csv("provider_overlap_summary.csv", index=False)

if not provider_overlap_df.empty:
    strongest_provider_overlap = provider_overlap_df.sort_values("provider_overlap_rate", ascending=False).iloc[0]
else:
    strongest_provider_overlap = None

# ----------------------------
# 5. Print terminal summaries
# ----------------------------

print("\n" + "=" * 90)
print("OVERALL DATASET SUMMARY")
print("=" * 90)
print(f"Total generated ideas analyzed: {total_ideas}")
print(f"Total prompts: {total_prompts}")
print(f"Total models: {total_models}")
print(f"Total prompt-level idea clusters: {total_clusters}")
print(f"Overall diversity ratio: {overall_diversity_ratio:.3f}")
print(f"Average intra-model repetition rate: {overall_intra_repetition:.3f}")
print(f"Average inter-model Jaccard similarity: {overall_inter_similarity:.3f}")
print(f"Average top-cluster share per prompt: {overall_top_cluster_share:.3f}")

print("\n" + "=" * 90)
print("MOST REPETITIVE MODEL/PROMPT CASE")
print("=" * 90)
print(
    f"{highest_intra['model']} produced {int(highest_intra['total_ideas'])} ideas "
    f"for the {highest_intra['prompt_id']} prompt, but they collapsed into only "
    f"{int(highest_intra['unique_clusters'])} distinct clusters."
)
print(f"Intra-model repetition rate: {highest_intra['intra_model_repetition_rate']:.3f}")
print(
    f"Most common cluster: {highest_intra['top_cluster']} "
    f"({int(highest_intra['top_cluster_count'])}/{int(highest_intra['total_ideas'])}, "
    f"{highest_intra['top_cluster_share']:.1%})"
)

print("\n" + "=" * 90)
print("LOWEST PROMPT-LEVEL DIVERSITY")
print("=" * 90)
print(
    f"The {lowest_diversity_prompt['prompt_id']} prompt produced "
    f"{int(lowest_diversity_prompt['total_ideas'])} total ideas but only "
    f"{int(lowest_diversity_prompt['unique_clusters'])} broad clusters."
)
print(f"Diversity ratio: {lowest_diversity_prompt['diversity_ratio']:.3f}")
print(
    f"Most common cluster: {lowest_diversity_prompt['top_cluster']} "
    f"({int(lowest_diversity_prompt['top_cluster_count'])}/{int(lowest_diversity_prompt['total_ideas'])}, "
    f"{lowest_diversity_prompt['top_cluster_share']:.1%})"
)

print("\n" + "=" * 90)
print("HIGHEST INTER-MODEL SIMILARITY PAIR")
print("=" * 90)
print(
    f"For the {highest_inter_pair['prompt_id']} prompt, "
    f"{highest_inter_pair['model_a']} and {highest_inter_pair['model_b']} shared "
    f"{int(highest_inter_pair['shared_clusters'])} clusters out of "
    f"{int(highest_inter_pair['total_clusters_pair'])} total clusters."
)
print(f"Jaccard similarity: {highest_inter_pair['jaccard_similarity']:.3f}")
print(f"Shared clusters: {highest_inter_pair['shared_cluster_names']}")

if strongest_provider_overlap is not None:
    print("\n" + "=" * 90)
    print("STRONGEST PROVIDER-LEVEL OVERLAP")
    print("=" * 90)
    print(
        f"For the {strongest_provider_overlap['prompt_id']} prompt, providers "
        f"({strongest_provider_overlap['providers']}) shared "
        f"{int(strongest_provider_overlap['common_clusters_across_all_providers'])} clusters "
        f"out of {int(strongest_provider_overlap['total_provider_union_clusters'])} total provider-level clusters."
    )
    print(f"Provider overlap rate: {strongest_provider_overlap['provider_overlap_rate']:.3f}")
    print(f"Common clusters: {strongest_provider_overlap['common_cluster_names']}")

print("\n" + "=" * 90)
print("PROMPT-LEVEL SUMMARY TABLE")
print("=" * 90)
print(
    prompt_df[
        [
            "prompt_id",
            "total_ideas",
            "unique_clusters",
            "diversity_ratio",
            "top_cluster",
            "top_cluster_share"
        ]
    ]
    .sort_values("diversity_ratio")
    .to_string(index=False)
)

print("\n" + "=" * 90)
print("COPY-PASTE PAPER SENTENCES")
print("=" * 90)

print(
    f"Intra-model repetition was clear. For instance, {highest_intra['model']} produced "
    f"{int(highest_intra['total_ideas'])} ideas for the {highest_intra['prompt_id']} prompt, "
    f"but they collapsed into only {int(highest_intra['unique_clusters'])} distinct clusters. "
    f"Its intra-model repetition rate was {highest_intra['intra_model_repetition_rate']:.3f}, "
    f"while the average intra-model repetition rate across all model-prompt pairs was "
    f"{overall_intra_repetition:.3f}. This means that a large portion of separate outputs "
    f"repeated similar, narrow claims."
)

print()

print(
    f"Inter-model homogeneity was also visible. For example, for the "
    f"{highest_inter_pair['prompt_id']} prompt, {highest_inter_pair['model_a']} and "
    f"{highest_inter_pair['model_b']} shared {int(highest_inter_pair['shared_clusters'])} "
    f"clusters out of {int(highest_inter_pair['total_clusters_pair'])} total clusters, "
    f"for a Jaccard similarity score of {highest_inter_pair['jaccard_similarity']:.3f}. "
    f"Across all model pairs and prompts, the average inter-model similarity score was "
    f"{overall_inter_similarity:.3f}."
)

print()

print(
    f"At the prompt level, the strongest convergence appeared in the "
    f"{lowest_diversity_prompt['prompt_id']} prompt, where {int(lowest_diversity_prompt['total_ideas'])} "
    f"generated thesis angles collapsed into only {int(lowest_diversity_prompt['unique_clusters'])} "
    f"broad idea clusters. This produced a diversity ratio of "
    f"{lowest_diversity_prompt['diversity_ratio']:.3f}, suggesting that many outputs differed "
    f"in wording while repeating the same general argumentative moves."
)

print()
print("\nSaved files:")
print("- intra_model_repetition.csv")
print("- inter_model_similarity.csv")
print("- prompt_level_diversity.csv")
print("- provider_overlap_summary.csv")
