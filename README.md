# grpc-tabular-model

## About this Repository

### gRPC and Modzy Container Specification

This is a gRPC + HTTP/2 implementation of the [Scikit Learn Classification tutorial](https://towardsdatascience.com/lime-how-to-interpret-machine-learning-models-with-python-94b0e7e4432e) and is derived from Modzy's [gRPC Python Model Template](https://github.com/modzy/grpc-model-template).

## Installation

Clone the repository:

```git clone https://github.com/modzy/grpc-tabular-model.git```

## Usage

All different methods of testing the gRPC template specification can be found in the [Usage section](https://github.com/modzy/grpc-model-template#Usage) of the gRPC Python Model Template.  

The following usage instructions demonstrate how to build the container image, run the container, open a shell inside the container, and run a test using the `grpc_model.src.model_client` module.

#### Build and run the container

From the parent directory of this repository, build the container image.

```docker build -t grpc-tabular-model .```

Run the container interactively.

```docker run -p 45000:45000 -it grpc-tabular-model:latest```

#### Run a test inside the container

Open a different terminal, create a Python virtual environment, and activate the environment

```
python -m venv ./tabular-model
source tabular-model/bin/activate
```

Submit a test using the `grpc_model.src.model_client` module from within the virtual environment

```python -m grpc_model.src.model_client``` 
