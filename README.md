# PSA-MER

PSA-MER is a Position- and Syntax-Aware Transformer framework for printed Mathematical Expression Recognition (MER). It converts formula images into LaTeX sequences while improving both spatial structure understanding and syntactic correctness.

The model is built on a Convolutional Vision Transformer backbone with two main components:

1. **Location-Sensing Transformer**  
   The LST module is inserted into the visual encoder to enhance position-aware representation learning. It helps the model capture fine-grained two-dimensional structures such as fractions, radicals, superscripts, and subscripts.

2. **Grammar-Aware Beam Re-ranking**  
   During inference, a lightweight grammar checker evaluates the compatibility between the current decoding prefix and each candidate token. The grammar score is fused with the decoder likelihood during beam search, encouraging structurally valid LaTeX output.

The overall framework aims to improve mathematical expression recognition by combining position-aware visual encoding with syntax-aware decoding.

# Running Example

## 1. Install dependencies
CUDA 12.8

Python 3.12
```bash
pip install -r requirements.txt
```
## 2. Training and Testing
   Edit the parameters in config.yaml
```bash
python train.py --task train --config config.yaml
python train.py --task test --config config.yaml --resume-from yourmodel.pt
```
