# Data Augmentation for Entity Resolution: A comparative evaluation

## Description
This repository provides the code for the paper ```Data Augmentation for Entity Resolution: A comparative evaluation``` from Tobias Rettenmeier and Alexander Jesser.


## Installation
- Python 3.7
- Torch 1.10
- Transformers 4.9
- NLTK 3.5
- Luigi 3.3
- MLFlow 1.30

```pip install -e .``` from the project root directory.

## Configuration
A working directory needs to be specified in the ```config.yml``` file.

The data is expected to be within the working directory in ```data/raw/```.


## Usage
To run a specific experiment, execute ```run_experiment.py``` with the name of an experiment found in the ```experiments``` module.

To run all experiments from that module, execute ```run_all_experiments.py```. Execution is managed by ```Luigi```, so make sure you have the scheduler ```luigid``` running.



## Packages used
- Ditto (https://github.com/megagonlabs/ditto)
- Easy Data Augmentation (https://github.com/jasonwei20/eda_nlp)
- NLPAug (https://github.com/makcedward/nlpaug)
- InvDA (https://github.com/megagonlabs/rotom)

