# GAME Borzoi Human Predictor

Borzoi ([Linder *et al.* 2025](https://www.nature.com/articles/s41588-024-02053-6)) is a deep learning model that predicts cell-type and tissue-specific RNA-seq coverage directly from DNA sequence, enabling interpretation of genetic variants across multiple layers of gene regulation, including transcription, splicing, and polyadenylation. It is trained on human RNA-seq data from ENCODE (with 866 datasets across diverse biosamples, including cell lines and adult tissues) and Genotype-Tissue Expression (GTEx) data (with 2-3 replicates for each tissue, processed by the recount3 project). The training dataset also includes epigenomic datasets from the Enformer model, such as CAGE, DNase-seq, ATAC-seq, and ChIP-seq tracks.

- To learn more about the GAME Framework ([Main GAME Repository](https://github.com/de-Boer-Lab/Genomic-API-for-Model-Evaluation), [preprint](https://www.biorxiv.org/content/10.1101/2025.07.04.663250v1.full))
- GAME documentation is hosted on: [ReadTheDocs](https://genomic-api-for-model-evaluation-documentation.readthedocs.io) 
- Pre-built Borzoi container image: [Zenodo](https://zenodo.org/records/18182688)
- To learn more about DREAM-RNN: [DREAM-RNN K562](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022/tree/main/benchmarks/human)

---
---

## Configuring Definition File and Running Predictor Container for Borzoi

**<ins>NOTE</ins>: We already have a pre-built container for Borzoi human model, as linked above. This section just walks through how we built the Predictor and how this workflow can be used to build on top of our work, which is the essence of GAME.**

For details regarding:

1. Creating wrapper functions for the API JSON structure
2. Configuring and Running the API:

    - Configuring containers using definition files
    - Purpose of definition files
    - Why containers are used and to learn more about them
please checkout [GAME documentation](https://genomic-api-for-model-evaluation-documentation.readthedocs.io) or the [DREAM-RNN K562 Predictor repository](https://github.com/de-Boer-Lab/GAME-DREAM-RNN-human-k562-predictor).

## Overview for building on top of this work

This container for Predictor includes:

- Predictor script for sequence processing and error handling.
- Integrated Borzoi model with its dependencies and `borzoi-gpu` conda environment created using `borzoi_gpu_environment.yml`.
- Latest Baskerville and Borzoi repositories (as of 2025-02-25), which also contain helper scripts and 4 replicate model weights.
- Support scripts like:
  - `schema_validation.py`
  - `predictor_content_handler.py`
  - `error_checking_functions.py`
  - `predictor_help_message.json`
  - `borzoi_utils.py`
  - `model_validation.py`

**NOTE:** This container ***requires a GPU*** for execution because the Borzoi model relied on TensorFlow's GPU-accelerated operations. Running on CPU may lead to excessive memory usage and thread allocation failures.

### Model Availability

The model weights can be downloaded as .h5 files by running the `download_models.sh` to download all model replicates and annotations into the `/borzoi/examples/` folder. From the `borzoi_API_script_and_utils/`, run this command:

```bash
cd borzoi
./download_models.sh
```

**NOTE:** Downloading the model weights require python modules like `pyfaidx`. For more details regarding the models, please refer to [Borzoi Github Repository](https://github.com/calico/borzoi?tab=readme-ov-file#model-availability).

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

### Build the container (SIF)

```bash
apptainer build borzoi_human_predictor.sif predictor.def
```

### Run the container

```bash
apptainer run --nv --containall borzoi_human_predictor.sif HOST PORT
```

## Details

- The container receives data via a TCP socket and does not require mounted data directories.
- Replace `HOST` and `PORT` with the server and port configuration for the evaluator to connect to.
- The `--nv` flag sets up the environment of the container to use an NVIDIA GPU and CUDA libraries to run a CUDA-enabled application.
- The `--containall` flag ensures a clean, isolated environment for reproducible runs.

## Purpose

- Facilitates genomic model evaluation and prediction using the Borzoi model.
- It is designed to seamlessly integrate with other tools via API endpoints.

## Example Command

```bash
apptainer run --nv --containall borzoi_human_predictor.sif 172.16.47.xxx 5000
```

## Arguments

1. `HOST`: IP address or hostname of the Predictor server.
2. `PORT`: Port number the Predictor is listening on.

## Additional Notes about the `%environment` Block in the Definition File

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

This flag creates a strictly isolated environment, preventing the container from accessing host files (like `/home`, `/tmp`) or external environment variables. This practice is critical for reproducible run as it ensures the container execution is not accidentally influenced by the host system.

#### Layered Isolation Approach

- Definition file (`.def`): The containers are built with `APPTAINER_NO_MOUNT` configurations to provide a safe default. This setting is intended to prevent common host directories from being automatically mounted, making the container inherently more isolated by design.
- Runtime (`--containall`): While defaults are useful, the `--containall` flag provides complete, enforced isolation at runtime. It blocks all unexpected host directories, scrubs environment variables, and creates separate IPC/PID namespaces. This is the best practice to guarantee a run is entirely isolated.

**Note:** Because `--containall` blocks access to host files by default, the `-B` (bind) flag must be used to explicitly allow the container to read inputs and write outputs.

### Additional Links for Reference

- [Apptainer Documentation](https://apptainer.org/docs/user/latest/)
- [HEP Softwate Foundation -- Introduction to Apptainer/Singularity](https://hsf-training.github.io/hsf-training-singularity-webpage/)
- [NSC: Using Apptainer on Berzelius](https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-apptainer/)
