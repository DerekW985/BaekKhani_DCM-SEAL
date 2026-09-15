# DCM-SEAL Codebase Study Guide & Implementation Roadmap

> **Target Audience:** Graduate students, mobility data scientists, and transportation researchers seeking to understand the architecture of **DCM-SEAL** (*Discrete Choice Model with Segmentation and Embedding-Augmented Learning*) and fork or adapt it for novel choice modeling applications.  
> **Accompanying Manuscript:** Baek & Khani (2026), *“A Discrete Choice Model with Segmentation and Embedding-Augmented Learning for Travel Behavior Analysis”* ([`BaekKhani2026_DCMSEAL.pdf`](file:///c:/git/temp/BaekKhani_DCM-SEAL/BaekKhani2026_DCMSEAL.pdf)).

---

## Overview & Conceptual Architecture

DCM-SEAL bridges classical econometric choice modeling and deep learning architectures. It extends the Latent Class Choice Model (LCCM) by integrating:
1. **A Nonlinear Segmentation Network**: A multi-layer perceptron (MLP) mapping individual sociodemographics to class assignment probabilities $P(k \mid s)$.
2. **Class-Specific Embedding Layers**: High-dimensional categorical context variables (e.g., trip purpose, luggage, time-of-day) mapped into alternative-specific utility adjustments with unit $L_2$ norm identification constraints.
3. **Interpretable Linear Utility Cores**: Traditional continuous policy attributes (e.g., travel time, monetary cost, headway) governed by economically sign-constrained parameters ($\beta_{\text{core}} \le 0$).

When reading the codebase, your objective is to understand how these theoretical components are translated into a vectorized, GPU-accelerated PyTorch Lightning pipeline.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                     Input Batch                         │
                     └───────┬──────────────────────┬───────────────────┬──────┘
                             │                      │                   │
             Sociodemographics (s)           Context Categorical (e)   Core Attributes (x)
                    (B, S)                         (B, E)                   (B, J, C)
                             │                              │                   │
              ┌──────────────┴──────────────┐               │                   │
              │  Segmentation Network (MLP) │               │                   │
              └──────────────┬──────────────┘               │                   │
                             │                              │                   │
                     Class Probabilities            Global Embeddings      Core Utility
                        P(k|s): (B, K)                 W_k[e]: (B,J,K)       (B, J, K)
                             │                              │                   │
                             │                       Positive Betas             │
                             │                    (Softplus): (K, E)            │
                             │                              │                   │
                             │                     Embedding Utility            │
                             │                         (B, J, K)                │
                             │                              │                   │
                             └──────────────┬───────────────┴───────────────────┘
                                            │
                                 Total Class Utility + ASC
                                        V_jk: (B, J, K)
                                            │
                                 Weighted Choice Utility
                                   V_j = Σ_k P(k|s) * V_jk
                                        (B, J)
                                            │
                                 Mask Unavailable Alts (-inf)
                                            │
                                 Softmax / Cross-Entropy Loss
```

---

## Logical Study Plan: Step-by-Step

Follow this sequence to build a progressive, rigorous understanding of the repository from data contracts to model definition, optimization, and post-estimation analysis.

### Step 1: Data Generating Process & Structural Contract
*Start here to see the ground truth econometric formulation and the exact input schema.*

1. **[`data/Synthesized/genSynthData.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/data/Synthesized/genSynthData.py)**
   - **What to study:**
     - The function [`synthDataGen`](file:///c:/git/temp/BaekKhani_DCM-SEAL/data/Synthesized/genSynthData.py#L13-L96) simulates a synthetic discrete choice market with 3 latent classes ($K=3$), 2 alternatives ($J=2$), core attributes ($x_1, x_2, x_3$), and embedding context levels ($e_1, e_2$).
     - Observe how true utility equations (Lines 77–80) are constructed: core linear effects, latent-class-specific modifiers, and discrete embedding shifts.
     - Note the target long-format data table structure: each record represents one alternative of a choice scenario, keyed by `chid` (choice situation ID), `alt` (alternative index), and `match` (1 if chosen, 0 otherwise).
2. **[`data/Synthesized/trueMRS.csv`](file:///c:/git/temp/BaekKhani_DCM-SEAL/data/Synthesized/trueMRS.csv)**
   - **What to study:**
     - Inspect the benchmark Marginal Rates of Substitution (MRS) for each latent class against the numeraire attribute $x_1$. This file serves as the ground truth benchmark for verifying parameter recovery.

---

### Step 2: Data Preprocessing & Pipeline Architecture
*Learn how raw CSV data is converted into memory-efficient, leak-free tensors.*

1. **[`src/data_processing.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/data_processing.py)**
   - **Key function:** [`load_and_preprocess_data`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/data_processing.py#L16-L155)
   - **What to study:**
     - **Mutual Exclusivity Validation (Lines 50–64):** Ensures that variables designated as `core_vars`, `embedding_vars`, and `segmentation_vars` do not overlap.
     - **Categorical Context Conversions (Lines 83–91):** How [`dfConv.csv`](file:///c:/git/temp/BaekKhani_DCM-SEAL/data/SwissMetro/dfConv.csv) is ingested to aggregate sparse categories prior to modeling.
     - **Embedding Indexing (Lines 94–109):** Uses `scikit-learn`’s `LabelEncoder` to convert embedding categories into contiguous integers $[0, Z-1]$ and discovers `embedding_dims`.
     - **Leak-Free Data Splitting (Lines 121–129):** Notice that train/test splitting is strictly partitioned by `chid` rather than raw rows, ensuring all alternatives of an individual choice situation remain in the same partition.
     - **Standardization (Lines 139–151):** `StandardScaler` is fitted *only* on the training subset and subsequently applied to the test subset.

---

### Step 3: High-Performance Data Loading (`PaddedChoiceDataset`)
*Understand the $O(1)$ batch tensor bank and vectorized gather.*

1. **[`src/data_loader.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/data_loader.py)**
   - **Key class:** [`PaddedChoiceDataset`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/data_loader.py#L14-L144)
   - **What to study:**
     - **Pre-computed Tensor Banks (`__init__`, Lines 68–104):** Rather than grouping pandas rows dynamically inside `__getitem__`, the dataset precomputes fixed-size memory banks:
       - `core_bank`: Tensor of shape $(N, J_{\max}, C)$
       - `mask_bank`: Boolean tensor $(N, J_{\max})$ indicating valid alternatives for each scenario
       - `choice_bank`: Tensor $(N,)$ storing the chosen alternative index
       - `seg_bank`: Tensor $(N, S)$ containing individual-level segmentation features
       - `x_emb_bank`: Tensor $(N, E)$ containing embedding indices
     - **Fast Mode & Vectorized Collation ([`collate_indices`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/data_loader.py#L137-L143)):** When `fast_mode=True`, `__getitem__` returns integer indices, and `collate_indices` performs a single vectorized `.index_select()`, bypassing Python loop overhead and eliminating DataLoader bottlenecks.

---

### Step 4: The Core DCM-SEAL Model Architecture
*The central mathematical engine of the repository.*

1. **[`src/dcm_seal_model.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/dcm_seal_model.py)**
   - **Key class:** [`DCM_SEAL`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/dcm_seal_model.py#L38-L591)
   - **Sub-components to read in detail:**
     - **Embedding Offsets (Lines 80–86):** Notice `self.emb_offsets = cumsum(...)`. Because multiple categorical variables share an underlying `nn.Embedding` matrix of size $(Z, J)$, category indices are shifted by offsets so they index into unique row ranges without collisions.
     - **Forward Pass ([`forward`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/dcm_seal_model.py#L174-L300)):**
       1. *Segmentation Probabilities:* `class_probs = F.softmax(self.segmentation_net(x_seg), dim=1)` yields shape $(B, K)$.
       2. *Core Utility:* Calculated via tensor contraction `torch.einsum('kc,bjc->bjk', core_betas, x_core)`. If variables are in `non_positive_core_vars`, non-positivity is enforced through `-F.softplus(core_betas)`.
       3. *Embedding Utility:* Categorical lookups produce $(B, E, J)$ representations, multiplied by positive class-specific weights $\beta_{\text{emb}} = \text{softplus}(\text{embedding\_betas})$ and summed over $E$.
       4. *Alternative Specific Constants (ASCs):* Added with reference category normalized to zero: `[0, asc]`.
       5. *Expected Utility & Masking:* Multiplied by class membership probabilities and summed over $K$ (`torch.sum(class_specific_utility * class_probs.unsqueeze(1), dim=2)`). Unavailable alternatives are masked with `-torch.inf` before passing to the softmax.
     - **Econometric Identification via Unit $L_2$ Projection ([`project_embedding_rows_unit_l2`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/dcm_seal_model.py#L302-L325)):**
       - Solves scale indeterminacy between embedding weights $W$ and magnitude coefficients $\beta_{\text{emb}}$.
       - Hooks into Lightning’s `on_fit_start` (Line 330) and `optimizer_step` (Line 334) to project rows of $W$ back onto the unit hypersphere: $\|W_{z, :}\|_2 = 1$.
     - **Differential Regularization ([`configure_optimizers`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/dcm_seal_model.py#L352-L393)):**
       - Segmentation network weights receive `weight_decay_segmentation` ($L_2$ penalty).
       - Embedding weights receive `weight_decay_embedding`.
       - Behavioral parameters ($\beta_{\text{core}}, \beta_{\text{emb}}, \text{ASC}$) have **zero weight decay**, preserving unbiased econometric estimates.
     - **Log-Likelihood & Goodness-of-Fit Hooks (Lines 478–487):** Tracks full sample log-likelihood ($LL_{\beta}$) and null log-likelihood under uniform choice over available alternatives ($LL_0$) to compute McFadden’s $\rho^2 = 1 - \frac{LL_{\beta}}{LL_0}$.

---

### Step 5: Experiment Execution & Standalone Runner
*Learn how to instantiate and train a single model instance.*

1. **[`src/run_model.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/run_model.py)**
   - **Key function:** [`run_model`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/run_model.py#L30-L148)
   - **What to study:**
     - How dynamic layer dimensions are calculated: `segmentation_net_dims = [total_seg_vars] + hidden_dims + [n_latent_classes]`.
     - Lightning `Trainer` setup with mixed-precision (`precision="16-mixed"` on CUDA), gradient clipping (`gradient_clip_val=1.0`), and checkpointing on minimum `val_loss`.
     - Inspect the test block at the bottom of the script (Lines 150–194) for an example configuration of a standalone single run.

---

### Step 6: Hyperparameter Search & Automated Exploration
*Understand how Optuna coordinates architecture search and training budgets.*

1. **[`main.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/main.py)**
   - **What to study:**
     - **Initial Settings Block (Lines 7–93):** Configuration dictionaries for datasets (`Synthesized` vs. `SwissMetro`).
     - **Dynamic Batch and Epoch Scheduling ([`objective`](file:///c:/git/temp/BaekKhani_DCM-SEAL/main.py#L99-L146)):**
       - Optuna samples `total_updates` and `updates_per_epoch`, calculating `batch_size = ceil(train_size / updates_per_epoch)` and `max_epochs = total_updates // updates_per_epoch`. This balances gradient stability with computational speed.
     - **Exploration Mode (`STUDYNAME='explore'`) (Lines 75–86):** Automatically generates combinatorial permutations of variables assigned to either the segmentation network or the embedding layers.
     - **SQLite Persistence & Artifact Pruning (Lines 149–208):** Saves Optuna trials in SQLite, renames top-ranking TensorBoard version logs, and removes inferior checkpoints.

---

### Step 7: Post-Search Econometric Analysis & Behavioral Metrics
*Convert trained neural weights back into standard transportation economics outputs.*

1. **[`src/reporting_estimates.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/reporting_estimates.py)**
   - **Key functions:**
     - [`extract_betas`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/reporting_estimates.py#L13-L86): Extracts constrained parameters ($\beta_{\text{core}}$, $\beta_{\text{emb}}$, $\text{ASC}$) into a clean DataFrame.
     - [`extract_embedding`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/reporting_estimates.py#L113-L149): Pulls the $(Z, J)$ weight matrix and maps integer rows back to human-readable category strings.
     - [`predict_membership`](file:///c:/git/temp/BaekKhani_DCM-SEAL/src/reporting_estimates.py#L151-L180): Evaluates the segmentation network on arbitrary DataFrames to output individual class membership probabilities.
2. **Post-Search Analysis Block in [`main.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/main.py#L213-L455)**
   - **Key functions:**
     - `resReport` (Lines 277–342): Computes Marginal Rates of Substitution ($\text{MRS} = \frac{\beta_i}{\beta_{\text{numeraire}}}$) across all alternatives and latent classes.
     - Hungarian Assignment (`match_rows`, Lines 261–275): Solves the label-switching problem in latent class models by matching estimated classes to ground truth via Mean Relative Absolute Error (MRAE).
     - Value of Time (VoT) reordering for empirical data (Lines 322–326): Automatically sorts classes by ascending VoT ($\frac{\beta_{\text{TT}}}{\beta_{\text{cost}}}$).

---

### Step 8: Empirical Application & Baseline Comparisons
*Examine the real-world benchmark workflow.*

1. **[`data/SwissMetro/preprocessSM.py`](file:///c:/git/temp/BaekKhani_DCM-SEAL/data/SwissMetro/preprocessSM.py)**
   - Reads the standard SwissMetro SP survey (`filteredSwissMetro.csv`) and reshapes it from wide format to long format, setting up choice availability masks and attribute scalings.
2. **[`mlogit/mlogit.R`](file:///c:/git/temp/BaekKhani_DCM-SEAL/mlogit/mlogit.R)**
   - Estimates classical Multinomial Logit (MNL) specifications (short MNL and full socio-demographic MNL) using R’s `mlogit` package. Use this as a baseline against which DCM-SEAL gains in $\rho^2$ and out-of-sample log-likelihood are measured.

---

## Actionable Forking Guide: Adapting to a New Dataset

To adapt DCM-SEAL to your own discrete choice problem, follow this structured checklist:

```
[Your Raw Data]
       │
       ▼
 1. Prepare Long-Format CSV (`dfIn.csv`):
    - Columns required: `chid`, `alt`, `match` (1/0), plus all features
    - If categorical levels require grouping, supply `dfConv.csv`
       │
       ▼
 2. Create Dataset Directory:
    - Place in `data/<YourDatasetName>/dfIn.csv`
       │
       ▼
 3. Define Experiment Configuration Dictionary:
    - core_vars: continuous policy variables (e.g., travel time, cost)
    - non_positive_core_vars: attributes where utility must decrease (cost, time)
    - embedding_vars: high-dimensional categorical contexts (purpose, weather, day-of-week)
    - segmentation_vars_categorical: demographic categories for class assignment (gender, age group)
    - segmentation_vars_continuous: demographic continuous variables (income, household size)
    - n_latent_classes (K) and n_alternatives (J)
       │
       ▼
 4. Run Smoke Test via `src/run_model.py`
       │
       ▼
 5. Launch Hyperparameter Tuning via `main.py`
       │
       ▼
 6. Run Post-Estimation Diagnostics:
    - Extract class memberships, VoT distributions, and embedding utility shifts
```

### Configuration Template

Below is a starter template for adding a custom dataset into `main.py`:

```python
case "MyTransitStudy":
    CONFIG = {
        # --- Data Specification ---
        "train_size": 15000,
        "test_size": 0.2,
        "random_state": 42,
        
        # --- Variables ---
        "core_vars": ["ivt", "ovt", "fare"],
        "non_positive_core_vars": ["ivt", "ovt", "fare"],  # Enforces beta <= 0 via -softplus
        "embedding_vars": ["trip_purpose", "weather", "time_of_day"],
        "segmentation_vars_categorical": ["car_ownership", "transit_pass", "worker_status"],
        "segmentation_vars_continuous": ["income_normalized", "age_normalized"],
        
        # --- Model Topology ---
        "n_latent_classes": 3,
        "n_alternatives": 4,          # Auto, Transit, Walk, Bike
        "choice_mode": "heterogeneous", # Alternative Specific Constants enabled
        "embedding_mode": "class-specific", # Separate embeddings per latent class
        
        # --- Architecture & Optimization Defaults ---
        "segmentation_hidden_dims": [64, 128, 64],
        "segmentation_dropout_rate": 0.1,
        "learning_rate": 0.005,
        "weight_decay_segmentation": 1e-2,
        "weight_decay_embedding": 1e-3,
        "batch_size": 512,
        "max_epochs": 100,
        "dataloader_fast_mode": True,
        "dataloader_num_workers": 0,
    }
```

---

## Key Caveats & Debugging Tips

1. **Latent Class Collapse (Class Saturation):**
   - If the segmentation network assigns all individuals deterministically to a single class (i.e., entropy approaches 0), increase `weight_decay_segmentation`, increase `segmentation_dropout_rate`, or reduce the learning rate. Inspect the `scaledentropy` metric in TensorBoard logs.
2. **Alternative Masking with Variable Choice Sets:**
   - Always ensure that `alt_position` and `mask_bank` reflect the true availability of alternatives per choice scenario. In `dcm_seal_model.py`, unavailable alternatives are set to `-torch.inf` so their choice probability is exactly 0.
3. **Reference Alternative Normalization:**
   - ASCs are parameterized with Alternative 0 as the reference alternative ($ASC_0 = 0$). Ensure that your choice coding adheres to zero-based indexing (`0, 1, ..., J-1`) across both `alt` and `choice`.
4. **Unit $L_2$ Embedding Norm vs. Scale:**
   - Do not remove `project_embedding_rows_unit_l2()`. Without row normalization on the embedding matrices, the product of embedding weights and embedding betas is overparameterized, causing numerical drift and uninterpretable parameter scales.
