
# DeepFusion — Music Genre Classification (Multi-modal)

DeepFusion is a PyTorch-based project for music genre classification using multi-modal audio features (mel-spectrograms, MFCC, chroma, and tempogram).

Key features
- Multi-modal model combining CNN branches for mel-spectrograms and MLPs for auxiliary features.
- Command-line interface (CLI) for training configuration (feature file, epochs, batch size, learning rate, etc.).
- Saves the best model checkpoint and logs training metrics to CSV and PNG files.

Repository structure
- `Propes_model` — Lightweight backwards-compatible runner (calls the package entrypoint).
- `train.py` — Top-level runner that sets up imports and executes the package CLI.
- `src/propes_model/` — Main package modules:
	- `data.py` — feature loading, preprocessing, Dataset and DataLoader creation.
	- `models.py` — model definitions (`MelCNN`, `AuxiliaryFeatureMLP`, `MultiModalNet`).
	- `train.py` — CLI entrypoint and training/evaluation loops.
	- `utils.py` — helper functions (CSV logging, directory utilities).

Quick start

1. Create a virtual environment and install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare or point to your pre-extracted features pickle file. The project expects a list of items where each item is a dict containing keys such as `features` (with `mel_spec_orig`, `mel_spec_harm`, `mel_spec_perc`, `mfcc`, `chroma`, `tempogram`) and `label`.

3. Run training (recommended):

```bash
python train.py --features /path/to/features.pkl --epochs 100 --batch-size 64 --lr 1e-4 --model-dir ./models
```

Compatibility wrapper (optional):

```bash
python Propes_model --features /path/to/features.pkl --epochs 100
```

Output
- The best model checkpoint is written to the directory specified by `--model-dir` as `best_multi_modal_gtzan_model.pth`.
- Training metrics are appended to `training_results_single_split.csv` and a plot is saved as `training_history.png` (or `training_history_single_split.png` depending on runner).

Notes and tips
- `src/propes_model/models.py` uses `torchvision.models.resnet50(pretrained=True)` by default. If you are offline or prefer not to download pretrained weights, change `pretrained=True` to `pretrained=False`.
- The DataLoader `num_workers` is set to `0` by default for portability. Increase it (e.g., `num_workers=4`) for faster data loading on multi-core machines.
- Large model weight files are ignored via `.gitignore`. For versioning large files consider using Git LFS.

Contributing
- Contributions are welcome. Please open issues or submit pull requests. If you add new dependencies, update `requirements.txt` accordingly.

License
- Add a `LICENSE` file in the repository root to specify the project license.

Contact
- For questions or collaboration, create an issue or contact the repository owner.

Hızlı Başlangıç

1. Ortamı hazırlayın (önerilen):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Özellik dosyanızı hazırlayın veya yolunu belirtin. Varsayılan yol: `/kaggle/input/morefeatures2improveddataset/gtzan_multi_modal_features_hps2.pkl`.

3. Eğitimi başlatın:

```bash
# Tercih edilen: paketlenmiş runner
python train.py --features /path/to/features.pkl --epochs 100 --batch-size 64 --lr 1e-4 --model-dir ./models

# Alternatif (eski wrapper uyumluluğu):
python Propes_model --features /path/to/features.pkl --epochs 100
```


Output
- The best model checkpoint is written to the directory specified by `--model-dir` as `best_multi_modal_gtzan_model.pth`.
- Training metrics are appended to `training_results_single_split.csv` and a plot is saved as `training_history.png` (or `training_history_single_split.png` depending on runner).


Notes and tips
- `src/propes_model/models.py` uses `torchvision.models.resnet50(pretrained=True)` by default. If you are offline or prefer not to download pretrained weights, change `pretrained=True` to `pretrained=False`.
- The DataLoader `num_workers` is set to `0` by default for portability. Increase it (e.g., `num_workers=4`) for faster data loading on multi-core machines.
- Large model weight files are ignored via `.gitignore`. For versioning large files consider using Git LFS.


Contributing
- Contributions are welcome. Please open issues or submit pull requests. If you add new dependencies, update `requirements.txt` accordingly.


License
- Add a `LICENSE` file in the repository root to specify the project license.
