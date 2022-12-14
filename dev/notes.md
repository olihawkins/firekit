# Development notes

## Environment

```zsh
pipenv install
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

See [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for more details.

## Cuda

Use `--extra-index-url CUDA_URL` to install the package with CUDA version of PyTorch.

## Tests

Run unit tests.

```zsh
python -m unittest -v
```