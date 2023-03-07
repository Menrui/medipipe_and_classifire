### conda installation

#### for Unix-like platform

download installer and run the script.

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

For more information, please refer to [miniforge official](https://github.com/conda-forge/miniforge) or [miniconda official](https://docs.conda.io/en/latest/miniconda.html)

### create python environment

```bash
conda env create -f=environment.yml
```
