## How to run the code:

1. Install dependencies (check `requirements.txt`). We use Python 3.7.
2. Install BackPACK: <https://f-dangel.github.io/backpack/>.
3. In `util/dataloaders.py`, change the path `path = '/mnt/Data/Datasets'` to your liking.
4. **Optional.** Run `python train.py` and `python train_binary.py`. See codes for arguments.
5. Run `python {dataset_name}.py --compute_hessian`.
6. Check out also the `notebooks` directory for the toy experiments.
