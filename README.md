# Fedn example
This is an example of how to script the Fedn training process.

## Prerequisites
* Linux (WSL might work)
* Docker
* Python (3.10 preferred, might work with other versions)

## Setup
To setup the environment, run the `setup.sh` script. This will clone the Fedn git repository, initialize a virtual environment, create the client config templates, and build the required docker images.

## Running
To run the federated learning, simply run the `run_federated.sh` script. This takes two parameters: the number of clients to use, and the number of rounds to train for. This will run the models and save the metrics in the `metrics` folder.

The code for the models is located in `client/entrypoint`.