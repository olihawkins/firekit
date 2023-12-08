# firekit

Firekit is a library of classes and functions for training and evaluating PyTorch models. The main focus of the library is a `Trainer` class that performs the standard training and evaluation loops, reports the training and evaluation loss and the evaluation performance on user-defined metrics, saves the model state when performance improves, and reloads the best model at the end of training. 

This project exists to support my work. It is in active development and the API is not stable.

## Installation

Install with `pip` or `pipenv` in the normal way.

```zsh
pip install firekit
```

Use the `---index-url` argument to install an older version of PyTorch for CUDA as a dependency. For example, use the following to get PyTorch with CUDA 11.8.

```zsh
pip install firekit --index-url https://download.pytorch.org/whl/cu118
```

