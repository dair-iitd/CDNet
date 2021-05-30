# CDNet
CDNet: Constraint based Knowledge Base Distillation in End-to-End TaskOriented Dialogs

## Datasets
The datasets are already present in the repository within respective folder names

## Run Environment
We include a `requirements.txt` which has all the libraries installed for the correct run of the CDNet code.
For best practices create a virtual / conda environment and install all dependencies via:
```console
❱❱❱ pip install -r requirements.txt
```

## Training
The model is run using the script `train.py` with flags according to respective dataset
```console
❱❱❱ python3 train.py -m 2 -ds 2 -dp ../CamRest/ -v vocab.json -hp 1 -d 0.00 -n model_name 
❱❱❱ python3 train.py -m 2 -ds 3 -dp ../WOZ/ -v vocab.json -hp 3 -d 0.00 -n model_name 
❱❱❱ python3 train.py -m 2 -ds 1 -dp ../Incar_sketch_standard/ -v vocab.json -hp 3 -d 0.05 -n model_name 
```

The list of parameters to run the script is:
- `--batch` batch size
- `--num_epochs` max number of epochs to train the model
- `--dropout` dropout ratio
- `--lr` learning rate
- `--load` Checkpoint path to load from
- `--ckpt_path` Checkpoint path to save checkpoints into
- `--name` Name of the checkpoints
- `--dataset` Dataset to run: 1 is Incar, 2 is Camrest, 3 is MultiWoz
- `--data` Path of the dataset
- `--logs` Print logs or not while training
- `--test` After specifying the load flag, if test flag is False, model starts training from that checkpoint otherwise just prints the test scores
- `--hops` Number of memory hops
- `--seed` Seed value to seed the run
- `--vocab` Each model run initiates its vocabulary from the vocab.json file specified here. If this file is deleted, saved checkpoints won't reproduce results as the mapping changes 

Look at `utils/config.py` for detailed information on the runtime options

### Training from Saved Model
There is support to start training from a previously saved checkpoint with the *--load* flag. If *--test* flag is set False along with setting *--load*, the model starts training from the checkpoint specified in the *--load* flag

## Testing
To obtain metric scores on the best model run `train.py` with *--test=True* and *--load=<checkpoint_path>*. Make sure all the parameter options match those of the trained model.
```console
❱❱❱ python3 train.py -m 2 -ds 2 -dp ../CamRest/ -v vocab.json -hp 1 -d 0.00 -n model_name -test True -ld ./checkpoints/camrest/model.pt
❱❱❱ python3 train.py -m 2 -ds 3 -dp ../WOZ/ -v vocab.json -hp 3 -d 0.00 -n model_name -test True -ld ./checkpoints/WoZ/model.pt
❱❱❱ python3 train.py -m 2 -ds 1 -dp ../Incar_sketch_standard/ -v vocab.json -hp 3 -d 0.05 -n model_name -test True -ld ./checkpoints/SMD/model.pt
```

## Hyperparameters and Results

| Dataset | Hops | DLD | LR | Val F1 | Test F1 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| CamRest | 1 | 0.0 | 0.0005 | 68.6 | 68.6 |
| SMD | 3 | 0.05 | 0.00025 | 60.3 | 62.9 |
| WoZ | 3 | 0.0 | 0.00025 | 35.5 | 38.7 |
