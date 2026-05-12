# HyperWeak-Urban

This repository provides the dataset generation code for **HyperWeak-Urban**, a weak-supervision urban hyperspectral image dataset suite generated from fully supervised hyperspectral benchmarks.

Dataset download: [HyperWeak-Urban on Google Drive](https://drive.google.com/file/d/1P7JNigzLbPo2SHq3ad0ZuYBspsKKlfS7/view?usp=drive_link)

Raw and generated dataset archive: [HyperWeak-Urban on Zenodo](https://zenodo.org/records/20125456)

## Overview

The generation pipeline converts clean pixel-level ground-truth labels into three weak-supervision settings:

- **Incomplete labels:** sparse pixel-level labels sampled from clean labels.
- **Inaccurate labels:** noisy labels generated with symmetric, asymmetric, and boundary noise.
- **Inexact labels:** point-level and scribble-level labels.

Supported sub-datasets:

- `AeroRIT`
- `Chikusei`
- `Houston2013`
- `Houston2018`
- `Pavia_Centre`
- `Pavia_University`
- `Washington_DC`

## Repository Structure

```text
.
+-- configs/
|   +-- paths.yaml
+-- data/
|   +-- sub_dataset/       # Source dataset loaders
|   +-- utils/             # Path config, saving, label mapping, and MAT helpers
|   +-- weak_label_gen/    # Incomplete, inaccurate, and inexact generators
+-- process_all.py         # Main generation script
+-- requirements.txt
+-- README.md
```

## How To Use

There are two ways to use this project.

### 1. Download the Generated Weak-Supervision Dataset

If you only need the ready-to-use weak-supervision dataset, download it directly:

- [HyperWeak-Urban on Google Drive](https://drive.google.com/file/d/1P7JNigzLbPo2SHq3ad0ZuYBspsKKlfS7/view?usp=drive_link)
- [HyperWeak-Urban on Zenodo](https://zenodo.org/records/20125456)

After extraction, the dataset already contains `images/`, `clean_labels/`, `incomplete_labels/`, `inaccurate_labels/`, `inexact_labels/`, and metadata files for each supported sub-dataset.

### 2. Generate Weak Labels With Custom Parameters

Use this option if you want to change the weak-supervision generation parameters, such as incomplete-label sampling rates, inaccurate-label noise ratio, or inexact-label generation settings.

Follow these steps to regenerate the weak labels.

**Step 1: Download and extract the original datasets**

Download the original fully supervised source datasets from [HyperWeak-Urban on Zenodo](https://zenodo.org/records/20125456), then extract them into a dataset root directory.

The expected raw-data layout is:

```text
<dataset_root>/
+-- Raw_Data/
    +-- AeroRIT/
    +-- Chikusei/
    +-- Houston2013/
    +-- Houston2018/
    +-- Pavia_Centre/
    +-- Pavia_University/
    +-- Washington_DC/
```

**Step 2: Clone this repository**

```bash
git clone https://github.com/1804071544/HyperWeak-Urban-Dataset.git
cd HyperWeak-Urban-Dataset
```

**Step 3: Install dependencies**

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include `numpy`, `rasterio`, `scipy`, `scikit-image`, `pyyaml`, and `tqdm`.

**Step 4: Adjust weak-supervision parameters**

Default generation parameters are defined in `process_all.py`.

Incomplete-label parameters:

```python
DEFAULT_INCOMPLETE_CONFIG = {
    "sampling_rates": [0.01, 0.05, 0.10, 0.20, 0.50],
    "bg_value": 0,
    "seed": 42,
    "show_progress": True,
}
```

Inaccurate-label parameters:

```python
DEFAULT_INACCURATE_CONFIG = {
    "noise_ratio": 0.30,
    "noise_weights": [1, 1, 1],
    "max_shift": 15,
    "adaptive_shift_ratio": 0.25,
    "smooth_sigma": 10.0,
    "bg_value": 0,
    "seed": 42,
    "asym_chunk_size": 200000,
    "n_jobs": 1,
    "show_progress": True,
}
```

Inexact-label parameters:

```python
DEFAULT_INEXACT_CONFIG = {
    "bg_value": 0,
    "show_progress": True,
}
```

Edit these dictionaries before running `process_all.py` if you need a different weak-supervision setting.

**Step 5: Set the dataset root and run generation**

`dataset_root` is the directory that contains `Raw_Data/` and receives the generated weak-supervision outputs.

You can set it in either of two ways.

Option A: edit the default config file:

```text
configs/paths.yaml
```

Example:

```yaml
data:
  dataset_root: E:\Hyperspectral_Dataset_Raw_Data
```

Then run:

```bash
python process_all.py
```

Option B: keep `configs/paths.yaml` unchanged and override the root path from the command line:

```bash
python process_all.py --dataset-root "E:\Hyper_Weak"
```

To process selected datasets only, add `--datasets`:

```bash
python process_all.py --dataset-root "E:\Hyper_Weak" --datasets AeroRIT Washington_DC
```

If `--datasets` is omitted, all supported datasets are processed.

In both cases, the expected input and output layout is:

```text
<dataset_root>/
+-- Raw_Data/              # Original fully supervised datasets
|   +-- AeroRIT/
|   +-- Chikusei/
|   +-- ...
+-- AeroRIT/               # Generated weak-supervision dataset
+-- Chikusei/
+-- ...
```

## Output Structure

Each processed dataset is saved as:

```text
<dataset_root>/<dataset_name>/
+-- images/
|   +-- image.tif
|   +-- reflectance.tif
|   +-- radiance.tif
+-- clean_labels/
|   +-- labels.tif
|   +-- label_mapping.json
+-- incomplete_labels/
|   +-- incomplete_labels_1percent.tif
|   +-- incomplete_labels_5percent.tif
|   +-- incomplete_labels_10percent.tif
|   +-- incomplete_labels_20percent.tif
|   +-- incomplete_labels_50percent.tif
+-- inaccurate_labels/
|   +-- inaccurate_labels.tif
|   +-- symmetric_noise_labels.tif
|   +-- asymmetric_noise_labels.tif
|   +-- boundary_noise_labels.tif
|   +-- noise_type_mask.tif
+-- inexact_labels/
|   +-- point_labels.tif
|   +-- scribble_labels.tif
+-- incomplete_labels_metadata.json
+-- inaccurate_labels_metadata.json
+-- inexact_labels_metadata.json
```

The exact image files depend on the source dataset. For example, AeroRIT saves `reflectance.tif` and `radiance.tif`, while most other datasets save `image.tif`.

## Label Mapping

For each source dataset, the loader first checks whether this file exists:

```text
<dataset_root>/Raw_Data/<dataset_name>/label_mapping.json
```

If it exists, it is copied into:

```text
<dataset_root>/<dataset_name>/clean_labels/label_mapping.json
```

If it does not exist, the code automatically creates a numeric fallback mapping where class names are numeric strings such as `"0"`, `"1"`, and `"2"`.

## Metadata

The script writes one metadata file for each weak-supervision setting:

- `incomplete_labels_metadata.json`
- `inaccurate_labels_metadata.json`
- `inexact_labels_metadata.json`

Each metadata file records the dataset name, supervision type, generation parameters, label format, and weak-label output directory.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hyperweak_urban_2026,
  title  = {HyperWeak-Urban},
  author = {Your Name and Coauthors},
  year   = {2026},
  url    = {https://github.com/1804071544/HyperWeak-Urban-Dataset}
}
```

## License

Please add the dataset license before public release.

## Contact

- **Name:** Enzhao Zhu
- **Email:** enzhao.zhu01@universitadipavia.it
