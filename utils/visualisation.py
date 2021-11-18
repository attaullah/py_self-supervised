import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE


def plot_t_sne(test_feat, labels, n_components=2, legend=None):

    tsne = TSNE(n_components=n_components, init='random', random_state=0, perplexity=100)
    embedding = tsne.fit_transform(test_feat)

    plt.figure(figsize=(16, 9))
    if legend:
        labels = legend
    ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels,
                         palette=sns.color_palette("Set1", len(np.unique(labels))))
    plt.legend(fontsize=12)

    return embedding, ax


def plot_umap(test_feat, labels, legend=True, sort=True, ret_umap=False):

    reducer = umap.UMAP()
    reducer.fit(test_feat)
    embedding = reducer.transform(test_feat)

    plt.figure(figsize=(16, 9))
    nc = len(np.unique(labels))
    if sort:
        # sort labels and embeddings
        sort_idx = np.argsort(labels)
        s_labels = labels[sort_idx]
        s_embedding = embedding[sort_idx]
    else:
        s_labels = labels
        s_embedding = embedding
    ax = sns.scatterplot(x=s_embedding[:, 0], y=s_embedding[:, 1], hue=s_labels, palette=sns.color_palette("Set1", nc),
                         legend=legend)
    plt.legend(fontsize=18, loc="lower right")
    plt.axis('off')
    if ret_umap:
        return reducer, embedding, ax
    return embedding, ax
