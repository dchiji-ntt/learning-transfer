
## Requirements
- torch==1.12.1
- torchvision==0.13.1
- torchaudio==0.12.1
- pyyaml
- pandas
- matplotlib
- scipy

## Usage
```
python exec_parallel.py <command> <exp_name>
```

- `<command>` should be `train` or `transfer`. `train` can be used to train a neural network and `transfer` can be used to reproduce our experiments for transferring learning trajectories, in the specified way by `<exp_name>`.
- `<exp_name>` is one of the keys defined in the `config.yaml`.
- The command sequentially runs each experiment over the grid specified by the `hparams_grid` option in `config.yaml`. It can sweep the grid efficiently if we execute the command multiple times in parallel.


