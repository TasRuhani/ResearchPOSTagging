# Comparative Research Report: POS Tagging Architectural Evolution

This report provides a detailed comparison between the previous POS tagging approaches and the newly introduced **Hybrid Semi-Supervised POS Tagging Framework for Low-Resource Languages** (Assamese). This document serves as a presentation-ready summary of the architectural changes, evaluation results, and structural improvements for your team.

---

## 1. Architectural Overview: Old vs. New

The previous pipeline relied on a combination of custom vocabulary clustering and heuristic rules to generate weak labels for training a basic linear model. The new architecture transitions to a **fully integrated, 3-phase hybrid deep learning pipeline** incorporating contextual embeddings, sequence-level constraints, tri-training consensus verification, and active learning.

### Side-by-Side Comparison

| Component / Phase | Old Architecture ([try1.ipynb](file:///g:/PROJECTS_DEMO/Research/try1.ipynb) / [semi3refine.ipynb](file:///g:/PROJECTS_DEMO/Research/semi3refine.ipynb)) | New Architecture ([Hybrid_SemiSupervised_POS_Tagging.ipynb](file:///g:/PROJECTS_DEMO/Research/Hybrid_SemiSupervised_POS_Tagging.ipynb)) |
| :--- | :--- | :--- |
| **Pipeline Workflow** | **Disjoint, heuristic prototype**: Pre-computed Google Drive files and static lookup lexicons used to train a standalone baseline classifier. | **Integrated 3-Phase hybrid loop**: Programmatic induction (Phase 1) followed by neural sequence refinement & filtering (Phase 2), leading to self-training and dataset augmentation (Phase 3). |
| **POS Induction & Clustering** | Custom-written Brown clustering algorithm from scratch. It attempted vocabulary-wide merges ($O(V^3)$ search complexity), causing it to stall (stuck at iteration 1 with zero likelihood gain due to sparsity). | **Scalable vector-reduction approach**: Counts left/right word contexts and subword orthographic features. Standardized and reduced via **Truncated SVD** (50 dense features) before clustering via **Agglomerative Clustering** (`linkage='ward'`). |
| **Weak Label Generation** | Static heuristic **Labeling Functions (LFs)**: Lookup tables for pronouns, conjunctions, prepositions, and suffix rules (e.g., checking if word ends in `ী` or `ীয়া`). Highly sensitive to word list constraints. | **Dynamic mapping**: Clusters automatically mapped to canonical POS tags by majority vote on the training seed (20% split). The entire unsupervised split (60%) is labeled dynamically using these mapping vectors. |
| **Sequence Labeling Model** | Word-shape and context features processed by `DictVectorizer` and `Logistic Regression` with optional bigram-based `Viterbi decoding` post-hoc. No deep representations. | State-of-the-art **Deep Neural Sequence Labeler**: Fuses pretrained **XLM-Roberta** contextual embeddings with **Character-level CNNs** (morphology capture), modeling context via a **BiLSTM** layer. |
| **Sequence Modeling Layer** | Token-level class probabilities decoded post-hoc. No joint transition constraint learning. | **Conditional Random Field (CRF)** layer integrated directly as the sequence loss and transition modeling layer. |
| **Confidence & Filtering** | Static probability threshold filtering on Logistic Regression output. | **Calibrated sequence-level confidence** based on length-scaled CRF log-likelihood: $P(y \vert x) = \exp(\log P(y \vert x) / N)$. |
| **Consensus & Verification** | No cross-validation or agreement filters. The classifier relies entirely on the quality of the LFs. | **Tri-Training Agreement Filter**: Token predictions from a generative HMM, a discriminative Logistic Regression, and the deep BiLSTM-CRF are cross-verified. Only consensus labels are accepted. |
| **Active Learning Loop** | None. Single-model execution. | **Adaptive Active Learning**: Sequences with sequence confidence below the $0.75$ threshold are routed to an Active Learning pool (gold label simulation). |
| **Self-Training** | None. Model is evaluated directly on a single run. | **Augmented Retraining**: Re-trains the neural network on the seed set combined with accepted pseudo-labels and Active Learning annotations, utilizing early stopping (patience=5) based on validation Macro F1. |

---

## 2. Deep Dive: Key Architectural Improvements

### A. Phase 1: Scalable Unsupervised POS Induction
* **Old Limitation**: Attempting to cluster raw vocabulary words directly using a custom bigram/trigram transition probability matrix from scratch. The $O(V^3)$ search complexity meant that with $12,000+$ unique vocabulary words, finding optimal cluster merges was computationally prohibitive and failed to scale (early stopping on Iteration 1).
* **New Design**: The new codebase bypasses vocabulary-wide search by extracting context counts and orthographic shapes, then using **SVD (Singular Value Decomposition)** to project high-dimensional context vectors into a 50-dimensional dense space. It then runs Agglomerative Clustering on the dense features. This results in stable, fast, and mathematically sound cluster mapping.

### B. Phase 2: Morphology-Aware Deep Neural sequence Labeling
* **Old Limitation**: Character-level morphology in Assamese (e.g., cases, prefixes, suffixes) was modeled using basic regex and suffix lookup LFs, while local word context was limited to immediate bigrams/trigrams.
* **New Design**: The new model uses a dual embedding framework:
  1. **Multilingual Contextual Embeddings (XLM-R)**: Captures complex sentence semantics and grammatical structures.
  2. **Character-level CNN (CharCNN)**: Employs 1D convolutions over character sequences with multiple window sizes (kernel sizes 3, 4, 5) and max-pooling to dynamically learn morphological cues (prefixes, suffixes, inflections) without manual dictionaries.
* **BiLSTM + CRF**: The BiLSTM captures bidirectional context, and the CRF layer models joint sequence-level transitions, enforcing syntax coherence (e.g., preventing invalid tag sequences).

### C. Phase 3: Tri-Training Agreement & Active Learning Loop
* **Old Limitation**: Weak labels generated by heuristic LFs were fed directly into the model without any filtering. Malformed annotations caused model drift and degraded test metrics.
* **New Design**: To prevent error propagation, the new pipeline uses a two-tier gate:
  1. **CRF Calibrated Confidence**: The CRF output calculates sequence-level probabilities. If confidence falls below $0.75$, the sequence is sent to Active Learning (simulating human query of true gold tags).
  2. **Tri-Training Filter**: For high-confidence sequences, a consensus is required between three structurally distinct models: a generative HMM, a discriminative Logistic Regression, and the deep BiLSTM-CRF. Only tokens where all three agree are accepted.

---

## 3. Empirical Performance and Improvements

Based on the empirical evaluations compiled in the submission report ([ResearchReport.pdf](file:///g:/PROJECTS_DEMO/Research/ResearchReport.pdf)):

### Overall Validation Performance
The new model achieves high sequence-level accuracy and macro-averaged metrics on the validation dataset:

* **POS Tagging Accuracy**: **88.46%**
* **Macro Precision**: **86.32%**
* **Macro Recall**: **85.99%**
* **Macro F1-Score**: **86.02%**
* **V-Measure (Clustering)**: **67.24%**
* **Average Log-Likelihood**: **-6.6788**

### Per-POS Tagging Breakdown
The deep tagger is highly robust across standard POS categories, particularly on frequent parts of speech:

| POS Category | Support | Validation Accuracy | Key Observations |
| :--- | :--- | :--- | :--- |
| **Pr** (Pronoun) | 286 | **95.80%** | Captures pronoun groupings with minimal error. |
| **CONJ** (Conjunction) | 203 | **95.07%** | Learns fixed-vocabulary conjunctions with high precision. |
| **N** (Noun) | 2193 | **89.47%** | Strong generalization on nouns despite high lexical vocabulary sparsity. |
| **Adj** (Adjective) | 761 | **86.73%** | Captures adjectives, utilizing the subword CharCNN filters for morphology. |
| **V** (Verb) | 568 | **84.68%** | Robust identification of verb endings in Assamese. |
| **PREP** (Preposition) | 86 | **81.40%** | Accurately maps suffix case-markers. |
| **Adv** (Adverb) | 80 | **68.75%** | Lower accuracy due to low support (80 samples) and high variability. |

*Note: Punctuation (`.`) and Interjections (`INTJ`) had 0 support in this validation split, showing 0.0% accuracy.*

---

## 4. Key Takeaways for the Team Presentation

1. **Overcoming Scalability Hurdles**: The team successfully replaced the slow, custom-written Brown clustering algorithm with an SVD-reduced Agglomerative Clustering pipeline, solving the vocabulary clustering bottleneck.
2. **State-of-the-Art Deep Model**: The addition of XLM-R and Character CNNs provides the model with strong semantic and morphological understanding, critical for Assamese.
3. **Smart Data Refinement (Self-Training + AL)**: Instead of manually expanding dictionary lexicons, the pipeline automatically refines weak labels via **Tri-Training Agreement** and routes low-confidence sequences to **Active Learning**, making optimal use of annotated data.
