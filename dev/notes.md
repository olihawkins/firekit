# Development notes

## Environment

```zsh
pipenv --three
pipenv install numpy pandas scikit-learn torch torchvision
pipenv install --dev ipython build twine
```

## Build

```zsh
python -m build
```

## Publish

Remove old builds from the `dist` folder before uploading.

```zsh
python -m twine upload dist/*
```

## Cuda

Use --extra-index-url CUDA_URL to install the package with CUDA version of PyTorch.