# GAME Borzoi Human Predictor

Borzoi ([Linder *et al.* 2025](https://www.nature.com/articles/s41588-024-02053-6)) is a deep learning model that predicts cell-type and tissue-specific RNA-seq coverage directly from DNA sequence, enabling interpretation of genetic variants across multiple layers of gene regulation, including transcription, splicing, and polyadenylation. It is trained on human RNA-seq data from ENCODE (across diverse biosamples, including cell lines and adult tissues) and Genotype-Tissue Expression (GTEx) data. The training dataset also includes epigenomic datasets from the Enformer model, such as CAGE, DNase-seq, ATAC-seq, and ChIP-seq tracks.

**The Predictor:**

- Returns `point` and `track` predictions for the requested `(type, cell_type)` pairs.
- The Borzoi model itself outputs predictions across all **7,611 human tracks** for every inference; the Predictor extracts only the tracks needed for the request before any per-task aggregation happens (see [Section 2.2](#22-prediction-logic)).
- Averages across 4 trained model folds, each evaluated on both forward and reverse-complement strands.
- Predicts on the full sequence context and slices to `prediction_ranges` afterwards, preserving regulatory flanking context.
- Uses an optional Matcher service to suggest semantically related cell types or binding molecules when no exact match exists; degrades gracefully to "exact-match-only" mode when Matcher is not configured.

**Quick start**

```bash
apptainer run --nv --containall borzoi_human_predictor.sif HOST PORT [MATCHER_IP] [MATCHER_PORT]
```

## Important Links

- To learn more about the GAME Framework ([Main GAME Repository](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation), [preprint](https://www.biorxiv.org/content/10.1101/2025.07.04.663250v1.full))
- GAME Documentation: [ReadTheDocs](https://genomic-api-for-model-evaluation-documentation.readthedocs.io)
- Pre-built Borzoi container image: [Zenodo]([LINK HERE])
- To learn more about Borzoi: [Borzoi GitHub Repository](https://github.com/calico/borzoi)

---

**<ins>NOTE</ins>: We already have a pre-built container for Borzoi human model, as linked above. This section just walks through how we built the Predictor and how this workflow can be used to build on top of our work, which is the essence of GAME.**

## 1. Overview

This document outlines the structure of the API codebase for Borzoi and how it integrates with the containerized setup. The architecture is designed as a containerized microservice that communicates via HTTP REST endpoints.

This container for Predictor includes:

- Predictor script for sequence processing and error handling
- Integrated Borzoi model with its dependencies and the `borzoi-gpu` conda environment created using `borzoi_gpu_environment.yml`
- Baskerville and Borzoi repositories (as of 2025-02-25), which contain helper scripts and 4 replicate model weights
- Support scripts:
  - `borzoi_predict_codebase.py`, `borzoi_utils.py`, `model_validation.py` (model-specific)
  - `schema_validation.py`, `predictor_content_handler.py`, `error_checking_functions.py`, `config.py` (API scaffolding)
  - `predictor_help_message.json`

**NOTE:** This container ***requires a GPU*** for execution because the Borzoi model relies on TensorFlow's GPU-accelerated operations. Running on CPU may lead to excessive memory usage and thread allocation failures.

---

## 2. Understanding the API

### Predictor API (Server)

- **Purpose**: A Flask-based web server that listens for HTTP requests, validates inputs, runs the Borzoi model, and returns structured JSON responses.
- **Core Script**: `src/script_and_utils/borzoi_predictor_API.py`
- **Inference Orchestrator**: `borzoi_predict_codebase.py` decomposes the prediction pipeline into focused helpers &mdash; track index collection, fold-averaged prediction, short-sequence handling, long-sequence tiling, last-chunk handling, and per-task prediction assignment &mdash; rather than running everything in one monolithic function.

### 2.1 Key Features

1. **Versioned Predictor Name**:
    - Inside the container: `Borzoi_Human_YYYYMMDD-HHMMSS_TZ` (e.g. `Borzoi_Human_20251128-180629_PST`).
    - Outside the container (dev mode): `Borzoi_Human_dev`.
    - The build timestamp is read from `/.singularity.d/labels.json` at startup and embedded in every response (`predictor_name` field).

2. **Dynamic Path Handling**:
    - `config.py` uses `os.path.exists('/.singularity.d')` to switch between in-container and dev paths for `BORZOI_DIR` and `HELP_FILE`.

3. **HTTP Server Setup**:
    - Endpoints: `POST /predict`, `GET /help`, `GET /formats`.
    - Wire formats: `application/json` and `application/msgpack` for both request and response (negotiated via `Content-Type` and `Accept` headers).

4. **Request Validation**:
    - Validates mandatory top-level keys (`readout`, `prediction_tasks`, `sequences`) and per-task keys (`name`, `type`, `cell_type`, `species`).
    - Rejects unsupported request features at the API edge (see [Section 2.3](#23-supported-request-features)).
    - Returns standardized error responses (400, 422, 500) keyed by violation category.

5. **Predict-Once-Filter-Many**:
    - The Borzoi model outputs predictions across all 7,611 human tracks for every inference. The Predictor performs **a single forward pass per sequence chunk** (or per tile, for long sequences) and then extracts only the union of track indices needed across all requested tasks.
    - For a request with multiple tasks (e.g. `(expression, K562)` and `(accessibility, K562)`), tracks are deduplicated before extraction so that the same track is never selected twice. Per-task averaging happens on the already-filtered prediction array.

6. **Multi-Task Tolerance**:
    - Per-task failures (unmatched cell types, missing tracks, Matcher errors) are recorded inline as `{"error": ...}` entries in `task_predictions`. Other tasks in the same request are still processed and returned.
    - This differs from DREAM-RNN (due to Borzoi's multi-task training), which rejects the entire request if any task has an unsupported feature.

7. **Help Endpoint**:
    - Returns the contents of `predictor_help_message.json` when `/help` is queried.

### 2.2 Prediction Logic

The Borzoi model takes a 524,288 bp input and outputs predictions across 16,352 bins (32 bp per bin) over a 523,264 bp prediction window &mdash; the model crops 512 bp from each side of its input as a buffer. This wide context is what makes Borzoi accurate but also what makes the prediction pipeline non-trivial for sequences of arbitrary length.

```text
Borzoi architecture constants:
  SEQ_LEN = MODEL_INPUT_LEN  = 524,288 bp     (model input)
  prediction_window          = 523,264 bp     (output window: 16,352 bins x 32 bp)
  HALF_BUFFER                = 512 bp         (per-side buffer cropped by the model)
  BIN_SIZE                   = 32 bp
```

#### 2.2.1 Track Index Collection (runs once per request)

Before any sequence is fed to the model, the Predictor (filter_evaluator_request() function in borzoi_utils.py) walks through every `(type, cell_type)` pair in the request and resolves it to a set of track indices:

```text
For each (request_type, cell_type) in request_tasks:
  1. Look up matching rows in simplified_targets_df (filtered by Assay + Cell Type)
  2. If exact match: record indices, cell_type_actual = cell_type
  3. If no exact match AND Matcher configured:
       3.1. Query Matcher for a semantically related cell_type
            (or binding molecule, for binding_* requests)
       3.2. Re-lookup indices using the matched value
       3.3. cell_type_actual = matched_cell_type
  4. If no exact match AND Matcher not configured:
       Record {"error": ...} for this task; continue with other tasks
  5. Add this task's indices to unique_track_indices (a set)

Final unique_track_indices is the union of all tracks needed across all tasks
-- this is what powers predict-once-filter-many.
```

`type_actual` and `cell_type_actual` in the response report what was actually used.

#### 2.2.2 Sequence Handling (runs once per sequence)

The Predictor handles three logical cases depending on sequence length and whether `prediction_ranges` are provided:

```text
1. Short sequence path (len ≤ 523,264 bp), no prediction_ranges:
   1.1. dna_1hot centers the sequence in 524,288 bp
        (right-biased: left = total_padding // 2)
   1.2. Predict -> extract unique_track_indices -> average across 4 folds
        (each fold averages forward and reverse-complement strands internally)
   1.3. slice_prediction_tracks removes pure-N bins on both sides
   1.4. trim_upstream = bp of N-padding still present in the leftmost returned bin
        (allows the Evaluator to align bin coordinates to base coordinates)

2. Short sequence path (len ≤ 523,264 bp), with prediction_ranges:
   2.1. subset_sequence_for_ranges trims the input around the range with HALF_BUFFER
        of context flanking each side. Coordinates are remapped so range_start/end
        are relative to the subsetted sequence
   2.2. Predict + extract + fold-average as in 1.2
   2.3. slice_prediction_tracks_for_range slices to bins covering the range only
   2.4. trim_upstream = bp in the leftmost bin that fall before range_start
   *full biological context is preserved around the range*

3. Long sequence path (len > 523,264 bp):
   3.1. Prepend HALF_BUFFER (512 bp) of N-padding so bin 0 of the tiled output
        aligns with base 0 of the input
   3.2. Tile in a loop through the sequence with stride = prediction_window (no overlaps):
        Case a (full chunk):     run model directly, append all 16,352 bins
        Case b (tail > pred_win): pad downstream with Ns to MODEL_INPUT_LEN, predict,
                                  crop pure-N bins from the end, continue
        Case c (tail ≤ pred_win): pad downstream with Ns to MODEL_INPUT_LEN, predict,
                                  crop pure-N bins from the end, break
        (Cases b and c share the same prediction logic -- they only differ in whether
         the tiling loop continues or breaks afterwards)
   3.3. Each tile extracts unique_track_indices from the model output before being
        appended
   3.4. Concatenate all tile outputs into a single (total_bins, n_tracks) array
   3.5. If prediction_ranges:
          start_bin = floor(range_start / BIN_SIZE)
          end_bin   = ceil(range_end / BIN_SIZE)
          slice to [start_bin:end_bin]
          trim_upstream = range_start - start_bin * BIN_SIZE
        Else:
          trim_upstream = 0  (bin 0 already aligns with base 0)
```

#### 2.2.3 Per-Task Aggregation (runs once per (sequence, task) pair)

After bin-level predictions are assembled, per-task aggregation is applied (_assign_predictions_to_tasks() function in borzoi_predict_codebase.py):

```text
For each task with valid track_indices:
  1. Map task's track_indices into positions within unique_track_indices
     (e.g. if unique_track_indices = [1, 2, 3, 4] and the task wants [1, 3],
      pick positions [0, 2] from the prediction array)
  2. Average across the selected tracks -> single track per (sequence, task)
  3. If readout = point: average across all bins -> single scalar per sequence
     If readout = track: keep the per-bin track and report trim_upstream
```

The response includes explicit `aggregation` metadata indicating which axes were averaged (`bins`, `tracks`, `replicates`).

### 2.3 Supported Request Features

| Feature | Support | Notes |
|---|---|---|
| `readout = point` | ✅ | Bin-averaged single value per sequence |
| `readout = track` | ✅ | Returns full bin-resolution track |
| `readout = interaction_matrix` | ❌ | Rejected with `prediction_request_failed` (422) |
| `type = expression` | ✅ | RNA-seq tracks |
| `type = expression_mrna`, `expression_pol1`, `expression_pol3` | ✅ | RNA-seq tracks |
| `type = expression_pol2` | ✅ | CAGE tracks (falls back to RNA if no CAGE match) |
| `type = accessibility` | ✅ | Concatenates ATAC + DNASE tracks |
| `type = binding_[molecule]` | ✅ | ChIP-seq tracks for the specified molecule |
| `type = conformation_*` | ❌ | Not supported |
| `species = homo_sapiens` | ✅ | |
| Other species | ❌ | Rejected with `prediction_request_failed` (422) |
| `scale = linear` (default) | ✅ | Native model output |
| `scale = log` | ✅ | Output is `log2(x + 1)` |
| `cell_type` | ✅ | Exact match preferred; Matcher used as fallback |
| `prediction_ranges` | ✅ | Predicts with full context, slices to range |
| `upstream_seq` / `downstream_seq` | ✅ | Prepended/appended before prediction |

**Multi-task request behaviour**: Borzoi processes each task independently. A failure in one task (no matching tracks, Matcher unavailable, etc.) is reported inline and does not block others in the same request.

### 2.4 Error Handling

| Status | Error key | Triggers |
|---|---|---|
| 400 | `bad_prediction_request` | Malformed JSON/msgpack, missing mandatory keys, invalid `prediction_ranges` format, unsupported readout |
| 422 | `prediction_request_failed` | Sequence has invalid characters, empty sequence, unsupported species/type/scale, `prediction_ranges` out of bounds |
| 500 | `server_error` | Model load failure, unexpected internal error |
| 502 | `upstream_dependency_failed` | Fatal network failure of a required external service (e.g. Matcher unreachable). Distinct from per-task `prediction_request_failed` because the entire request cannot be served, not just one task |

---

## 3. Building and Running the Container

### 3.1 Model Availability

The model weights can be downloaded as .h5 files by running the `download_models.sh` to download all model replicates and annotations into the `/borzoi/examples/` folder. From the `borzoi_API_script_and_utils/`, run this command:

```bash
cd borzoi
./download_models.sh
```

**NOTE:** Downloading the model weights requires python modules like `pyfaidx`. For more details regarding the models, please refer to [Borzoi Github Repository](https://github.com/calico/borzoi?tab=readme-ov-file#model-availability).

The saved models will be arranged as such (based on HUMAN training data only):

```bash
borzoi_API_script_and_utils/borzoi/examples/saved_models
├── f3c0
│   └── train
│       └── model0_best.h5
├── f3c1
│   └── train
│       └── model0_best.h5
├── f3c2
│   └── train
│       └── model0_best.h5
└── f3c3
    └── train
        └── model0_best.h5
```

### 3.2 Build the container (SIF)

```bash
cd src
apptainer build borzoi_human_predictor.sif borzoi_human_predictor.def
```

### 3.3 Run the container

```bash
apptainer run --nv --containall borzoi_human_predictor.sif HOST PORT [MATCHER_IP] [MATCHER_PORT]
```

- `--nv`: enables NVIDIA GPU support (required).
- `--containall`: full runtime isolation (blocks host directory mounts, scrubs environment variables).
- Replace `HOST` with the server IP (e.g. from `hostname -I`) and `PORT` with an open port number above 1024.
- `MATCHER_IP` and `MATCHER_PORT` are optional. Without them, the predictor runs in "exact-match-only" mode and reports `matcher_version: "N/A"` in responses.

#### Example

```bash
apptainer run --nv --containall borzoi_human_predictor.sif 172.xx.xx.xx 5000
```

On startup you should see:

```text
Borzoi_Human_20251128-180629_PST Predictor is running on http://172.16.47.244:5000
Matcher service is NOT configured. Running in 'exact-match-only' mode.
```

---

## 4. Example Input and Output JSON

### 4.1 Input

```json
{
    "readout": "track",
    "prediction_tasks": [
        {
            "name": "k562_accessibility",
            "type": "accessibility",
            "cell_type": "K562",
            "species": "homo_sapiens",
            "scale": "linear"
        }
    ],
    "sequences": {
        "enhancer_region": "ACGT...<long sequence>...ACGT"
    },
    "prediction_ranges": {
        "enhancer_region": [10000, 11000]
    }
}
```

### 4.2 Output

```json
{
    "predictor_name": "Borzoi_Human_20251128-180629_PST",
    "matcher_version": "N/A",
    "bin_size": 32,
    "prediction_tasks": [
        {
            "name": "k562_accessibility",
            "type_requested": "accessibility",
            "type_actual": ["ATAC", "DNASE"],
            "cell_type_requested": "K562",
            "cell_type_actual": "K562",
            "species_requested": "homo_sapiens",
            "species_actual": "homo_sapiens",
            "scale_prediction_requested": "linear",
            "scale_prediction_actual": "linear",
            "aggregation": {"tracks": "mean", "replicates": "mean"},
            "trim_upstream": {"enhancer_region": 16},
            "predictions": {
                "enhancer_region": [0.123, 0.145, ..., 0.087]
            }
        }
    ]
}
```

The `predictor_name` matches the container build timestamp, allowing evaluation results to be traced back to a specific build. `trim_upstream` reports how many base pairs of the leftmost returned bin fall before `range_start` &mdash; the Evaluator uses this to align bin-level predictions to base-level coordinates. 

Note: For track requests, once the Evaluator expands Borzoi predictions using `bin_size`, they should be cropped using `trim_upstream` upstream and the remaining downstream excess should be cropped to achieve a 1-1 mapping of sequence to predictions. 

---

## 5. Additional Notes about the `%environment` Block in the Definition File

```bash
%environment
    # Prevent automatic binding of host directories
    export APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"
    # Prepend the borzoi-gpu conda environment's bin directory so its executables (like python3) are used
    export PATH="/opt/conda/envs/borzoi-gpu/bin:$PATH"
    # Set the library search path to prioritize libraries from the borzoi-gpu environment and CUDA libraries
    export LD_LIBRARY_PATH="/opt/conda/envs/borzoi-gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    # Define CUDA_HOME pointing to the conda environment    
    export CUDA_HOME="/opt/conda/envs/borzoi-gpu"
    # Configure TensorFlow to use the newer cuDNN frontend API, rather than the legacy API
    export TF_CUDNN_USE_FRONTEND=1
```

- `export APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"`:
*Why it is required:* By default, Apptainer automatically mounts host directories (like`/home` directory, `/tmp`, `/proc`, `/sys`, and `/dev`) into the container. This can inadvertently expose host data or cause conflicts.

For maximum security and reproducibility, it is highly recommended to run containers with the `--containall` flag.

This flag creates a strictly isolated environment, preventing the container from accessing host files (like `/home`, `/tmp`) or external environment variables. This practice is critical for reproducible runs as it ensures the container execution is not accidentally influenced by the host system.

#### Layered Isolation Approach

- Definition file (`.def`): The containers are built with `APPTAINER_NO_MOUNT` configurations to provide a safe default. This setting is intended to prevent common host directories from being automatically mounted, making the container inherently more isolated by design.
- Runtime (`--containall`): While defaults are useful, the `--containall` flag provides complete, enforced isolation at runtime. It blocks all unexpected host directories, scrubs environment variables, and creates separate IPC/PID namespaces. This is the best practice to guarantee a run is entirely isolated.

**Note:** Because `--containall` blocks access to host files by default, the `-B` (bind) flag must be used to explicitly allow the container to read inputs and write outputs.

### Additional Links for Reference

- [Apptainer Documentation](https://apptainer.org/docs/user/latest/)
- [HEP Software Foundation -- Introduction to Apptainer/Singularity](https://hsf-training.github.io/hsf-training-singularity-webpage/)
- [NSC: Using Apptainer on Berzelius](https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-apptainer/)
