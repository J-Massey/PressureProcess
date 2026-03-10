PressureProcess
===============

This module performs pressure processing for two microphones. It ingests raw
mat files, computes calibration transfer functions, saves raw HDF5 products,
then applies those transfer functions to produce processed HDF5 outputs. It
also includes plotting sanity checks.

Install
-------
From the repository root:
```
python -m pip install -e .
```
or:
```
python -m pip install .
```

Tests and coverage
------------------
Install test tooling:
```
python -m pip install -e ".[dev]"
```

Run coverage:
```
pytest --cov=src --cov-config=.coveragerc --cov-report=term-missing --cov-report=xml
```

Docker
------
Build:
```
docker build -t pressureprocess .
```

Run (mount local `data/` and `figures/`):
```
docker run --rm \
  --memory=8g \
  --cpus=4 \
  --user "$(id -u):$(id -g)" \
  -e PRESSUREPROCESS_ROOT_DIR=data/phase1 \
  -e PRESSUREPROCESS_USE_TEX=auto \
  -e TORCH_NUM_THREADS=1 \
  -v "$(pwd)/data:/app/data:Z" \
  -v "$(pwd)/figures:/app/figures:Z" \
  pressureprocess
```

Notes:
- `PRESSUREPROCESS_ROOT_DIR` controls where the pipeline looks for inputs and writes outputs.
- The image installs CPU-only PyTorch and conservative thread defaults to reduce memory pressure on smaller hosts.
- `--user "$(id -u):$(id -g)"` avoids host/container ownership mismatches.
- `:Z` on bind mounts applies SELinux relabeling (needed on Fedora/RHEL/CentOS systems).
- `PRESSUREPROCESS_USE_TEX` controls LaTeX labels in plots: `auto` (default), `true`, `false`.
  In `auto`, LaTeX is used only if a `latex` binary is available.

Usage overview
--------------
1) Define the required file structure and raw variable names.
2) Set user parameters in `src/config_params.py` and/or `PRESSUREPROCESS_ROOT_DIR`.
3) Run the pipeline with `python -m src.run_pipeline`.

Required file structure
-----------------------
All paths are rooted at `ROOT_DIR` in `src/config_params.py`.

Raw wall pressure mat files
```
data/phase1/raw_wallp/
  close/
    0psig.mat
    50psig.mat
    100psig.mat
  far/
    0psig.mat
    50psig.mat
    100psig.mat
```

Raw calibration mat files for PH to NC
```
data/phase1/raw_calib/PH/
  calib_0psig_1.mat
  calib_0psig_2.mat
  calib_50psig_1.mat
  calib_50psig_2.mat
  calib_100psig_1.mat
  calib_100psig_2.mat
```

Raw calibration mat files for NC to NKD
```
data/phase1/raw_calib/NC/
  0psig/
    nkd-ns_nofacilitynoise.mat
  50psig/
    nkd-ns_nofacilitynoise.mat
  100psig/
    nkd-ns_nofacilitynoise.mat
```

Raw variable names expected in mat files
----------------------------------------
For wall pressure and freestream raw files, the loader expects:
```
channelData
```
with columns ordered as:
```
PH1, PH2, NC
```

For PH to NC calibration files, the loader expects:
```
channelData_WN
```
with columns ordered as:
```
PH1, PH2, NC, ...
```

For NC to NKD calibration files, the loader expects:
```
channelData
```
and for 100psig only:
```
channelData_nofacitynoise
```

Configure user parameters
-------------------------
Edit `src/config_params.py` to set:
 - ROOT_DIR (or set env var `PRESSUREPROCESS_ROOT_DIR`)
 - LABELS, PSIGS, U_TAU, U_TAU_REL_UNC, U_E, ANALOG_LP_FILTER, F_CUTS
 - SENSITIVITIES_V_PER_PA
 - SPACINGS
 - RUN_NC_CALIBS
 - INCLUDE_NC_CALIB_RAW

Path fields (`RAW_CAL_BASE`, `RAW_BASE`, `TF_BASE`, `PH_RAW_FILE`,
`PH_PROCESSED_FILE`, `NKD_RAW_FILE`, `NKD_PROCESSED_FILE`) are derived from
`ROOT_DIR`.
