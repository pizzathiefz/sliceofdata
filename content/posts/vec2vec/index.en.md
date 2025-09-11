---
title: "vec2vec: Bridging Embeddings from Different Spaces"
date: 2025-05-26T20:38:11+09:00
draft: false
author: pizzathief
description: A brief summary of Harnessing the Universal Geometry of Embeddings (2025)
summary: A brief summary of Harnessing the Universal Geometry of Embeddings (2025)
keywords:
  - vec2vec
  - embedding
tags:
  - AI-ML
categories:
  - data
slug: vec2vec
ShowReadingTime: true
ShowShareButtons: true
ShowPostNavLinks: false
ShowBreadCrumbs: false
ShowCodeCopyButtons: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
disableScrollToTop: false
hidemeta: false
hideSummary: false
showtoc: true
tocopen: false
UseHugoToc: true
disableShare: false
searchHidden: false
robotsNoIndex: false
comments: true
weight: 10
math: true
---

In deep learning, embeddings have been central for quite some time. Since models ultimately work with numbers, the first step is always to turn raw data—whether it's an image or text—into a compact, dense vector that captures its underlying meaning. These embeddings are what make it possible, from the earliest neural networks to today's LLMs, to measure similarity and perform meaning-driven computations.

{{< figure src="/posts/vec2vec/embedding-vector.png" width="500" caption="[image source](https://www.pinecone.io/learn/vector-embeddings/)" align="center">}}


There's a phrase that often comes up when talking about embeddings: "The embeddings aren't in the same space," or "They're not aligned." What this means is that embeddings trained by different models live in different vector spaces. For example, even if we take the exact same word, like _dog_, the vector produced by Model A and the vector produced by Model B will almost certainly be different. Because of that, you can't simply take Model A's _dog_ vector and Model B's _cat_ vector, measure their cosine similarity, and then draw conclusions about their relationship. The two sets of embeddings don't share the same semantic framework—almost like two people speaking entirely different languages. That's why the general rule of thumb is: **embeddings from different spaces shouldn't be directly compared or mixed together without some alignment process.**

<br>

## The Platonic Representation Hypothesis

[The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) offers an intriguing perspective on this topic. It proposes that when trained on data of sufficient scale, representation models—regardless of their differing architectures—will ultimately converge toward learning the same underlying representations.


{{< figure src="/posts/vec2vec/platonic-representation-hypothesis.png" width="400" caption="the essence and the shadow" align="center">}}


It sounds a little out of place to mention Plato here, but there's a reason. That's because of his well-known Theory of Forms (or _Ideas_). The idea is that there exists an unchanging essence—the "Form"—while what we perceive in the real world is just an imperfect depiction of it, like prisoners in a dark cave seeing only the shadows of objects cast on the wall. 

You can think of **Z** in the figure as representing that underlying reality (the essential features). Images and text are just two different ways of describing it. Models trained on images or text are, in effect, observing projections of this reality **Z** and extracting information from those projections.

The core claim of the paper is that since there's always a single underlying truth (an essence, or statistical property) behind the data, the bigger the dataset and model, and the more capable the model becomes at downstream tasks, the more different embeddings will tend to converge toward the same representation. As indirect evidence, the authors also present alignment metrics showing this convergence across embeddings trained in very different ways.

So why does this convergence happen—not just "because the essence is the same," but more concretely? The paper points to three main reasons:

- **Task Generality**: As datasets grow larger and tasks become more general (and harder), the constraints increase. That means the space of valid solutions—the kinds of representations that can actually satisfy those constraints—shrinks. In other words, no matter where you start, you'll be funneled toward roughly the same target space.
- **Model Capacity**: Bigger models are better at finding optimal solutions. With enough capacity, a capable model is more likely to converge toward that shared representation.
- **Simplicity Bias**: Even with the same inputs and outputs, models could in theory learn very different internal representations. But deep networks are inherently biased toward simpler fits. That makes them more likely to settle into the overlapping "simple" solution space, rather than diverging into more complex alternatives.

<br>

## vec2vec

The paper [_Harnessing the Universal Geometry of Embeddings_](https://arxiv.org/html/2505.12540v2) takes the Platonic representation idea a step further. It shows that two embeddings trained by completely different models can actually be aligned — without knowing the encoders and without any paired supervision—purely in an unsupervised way. In other words, it demonstrates a kind of universal "embedding translator" that works across models. They call this approach **vec2vec**.


{{< figure src="/posts/vec2vec/vec2vec-architecture.png" width="580" align="center">}}


### Architecture

The structure of **vec2vec** is shown in the diagram above: it's broken into three components—an input adapter (A), an output adapter (B), and a shared backbone network (T). The goal is to connect embeddings from two different models. Imagine we have embeddings from _Model 1_ and _Model 2_, but we don't know the encoder of one side. The key assumption is that each embedding can be mapped into some **universal latent representation** (the "essential" representation in Platonic terms).

- **Input Adapter (A)**: Maps embeddings from each individual space into the universal latent representation. 
- **Shared Backbone Network (T)**: Operates inside the latent space to refine and align embeddings so they line up across sources/domains.
- **Output Adapter (B)**: Maps the universal latent representation back into the embedding space of each model.

Technically, they implement this with an MLP using residual connections, layer normalization, and SiLU nonlinearities. Combining A, T, and B yields four possible functions (∘ is function composition):

- **Transformation functions**
	- F1 = B2 ∘ T ∘ A1: Map from Model 1's space to Model 2's space    
    - F2 = B1 ∘ T ∘ A2: Map from Model 2's space to Model 1's space
- **Reconstruction functions**
    - R1 = B1 ∘ T ∘ A1: Take Model 1's embedding, go through latent space, and reconstruct it in Model 1's space
    - R2 = B2 ∘ T ∘ A2: Same, but for Model 2

Now, why is **T** needed at all—couldn't A and B just handle the mapping? The point is that A merely aligns each model's embeddings into a shared latent space, but that by itself doesn't guarantee the embeddings from different domains will sit nicely together. **T** acts as a universal MLP inside the latent space, enforcing that embeddings from multiple sources (models, domains) are properly aligned and comparable at a deeper, shared level.


### Training

The goal of **vec2vec** is to make embeddings—trained in different ways on the same text—converge toward nearly identical representations in a shared universal latent space. To achieve this, the model is trained with several complementary objectives (loss functions):

- **Reconstruction Loss**: An embedding that passes through the latent space and back to its original space should reconstruct faithfully.
- **Cycle Consistency Loss**: An embedding that travels from its original space → latent space → another space → latent space → back to the original should still come back intact.
- **Vector Space Preservation Loss**: The pairwise distances between transformed embeddings should be close to the pairwise distances before transformation, preserving the relative geometry of the space.
- **Adversarial Loss**: As in a standard GAN setup, the transformed embeddings (via F and R as generators) are judged by a discriminator, encouraging their distribution to match the target space distribution.

The logic behind the first two losses is this: even if embeddings bounce around between two spaces and through the latent representation, their values should remain stable if the A–T–B adapters are doing their job correctly. It's like me saying something in Korean to a translator, that being rendered into French, then repeated back by a French speaker into the translator, and finally delivered back to me. The sentence I receive at the end should still be the one I originally spoke.

As for the GAN loss, ablation studies show that removing it significantly degrades performance. This suggests it plays a critical role in the process of generating new vectors during the mapping into latent space.


### Evaluation

{{< figure src="/posts/vec2vec/results.png" width="480"  align="center">}}

Put another way, the goal of **vec2vec translation** is: given a source embedding $u_i = M_1(d_i)$ (where we don't know what model $M_1$ actually is), generate a translated vector $F(u_i)$ that is as close as possible to the target embedding $v_i = M_2(d_i)$ of the same document under a different model.
 
The evaluation metrics are:
- **Mean Cosine Similarity**: How similar $F(u_i)$ is to the true target embedding $v_i$. Ideal value = **1.0**.
- **Top-1 Accuracy**: The proportion of cases where $F(u_i)$ correctly identifies $v_i$ as the closest embedding (by cosine similarity) among the candidate set $M_2(d_j)$. Ideal value = **1.0**.
- **Mean Rank**: The average rank position of the true target embedding $v_i$ when comparing $F(u_i)$ against the candidate set $M_2(d_j)$. Ideal value = **1st**.

In the experiments, vec2vec was trained on the **Natural Questions (NQ)** dataset. It not only performed well on in-distribution evaluation (using the same dataset) but also generalized effectively to completely different domains—like tweets and medical records—showing strong robustness.


<br>

## Embeddings aren't immune to security risks

Within industry practice, embeddings are often considered less sensitive than raw customer data. For instance, a record like "Customer A's name is OOO, she's a 28-year-old woman living in Seoul, and she purchased products A, B, and C…"  is obviously highly sensitive personal information. Everyone agrees that if such data leaks, it's a major problem.

By contrast, if you train a model and extract an embedding that represents a customer's purchase behavior, what you see is just something like (0.324, -0.4253, 0.988, …  across256 dimensions). At first glance, it feels impossible for anyone who steals this to know what it actually means. That's why many assume embeddings don't need the same degree of protection.

But in the final section of the **vec2vec** paper, the authors show that even if you only know the modality and language of embeddings trained by an unknown model, vec2vec can be used to **recover nontrivial amounts of information from them**. In other words, embeddings aren't as opaque or "safe" as they look—they can leak meaningful signals if not properly secured.


{{< figure src="/posts/vec2vec/prompt.png" width="480" align="center">}}

{{< figure src="/posts/vec2vec/inversion.png" width="480"  align="center">}}


Here's how the email experiment worked: they took email text, passed it through an unknown model $M_1$ to get embeddings, then used **vec2vec** to convert those embeddings into the space of a known model $M_2$. From there, they attempted zero-shot inversion—reconstructing the original email text from the transformed embeddings. Finally, they fed both the original and reconstructed emails into GPT-4o, asking it a simple yes/no question: "Was information from the original email leaked?"

The results were striking. For certain model pairs, up to **80% of the document's content** could be recovered from the transformed embeddings. The reconstructions weren't perfect, but they still contained sensitive details: names, dates, promotions, financial data, service outages, even lunch orders.
This shows two things:
1. **vec2vec can map between embedding spaces effectively**, not just preserving geometric structure but also retaining **semantic content**.
2. The assumption that embeddings are "just hundreds of meaningless numbers" is risky—because with the right tools, those numbers can expose far more information than expected.


<br>