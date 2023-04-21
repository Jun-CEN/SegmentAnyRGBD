## Installation

### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Usage

Install required packages. 

```bash
conda create --name ovseg python=3.8
conda activate ovseg
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

You need to download `detectron2==0.6` following [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```


FurtherMore, install the modified clip package.

```bash
cd third_party/CLIP
python -m pip install -Ue .
```