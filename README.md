## Vae-Pytorch

This repository has some of my works on VAEs in Pytorch. At the moment I am doing experiments on usual non-hierarchical VAEs. Their architecture are based on https://github.com/3ammor/Variational-Autoencoder-pytorch.

### Currently implemented VAEs:

1. Standard Gaussian based VAE
2. Gamma reparameterized rejection sampling by Naesseth et al. (https://arxiv.org/abs/1610.05683). This implementation is based on the work of Mr. Altosaar (https://github.com/altosaar/gamma-variational-autoencoder).

## How to run
Example 1:
```bash
$ python3 main.py
```

Example 2:
```bash
$ python3 main --model normal --epoches 5000
```

Example 3:
```bash
$ python3 main --model gamma
```

### Usage

```
usage: main.py [-h] [--model MODEL] [--epoches N]

optional arguments:
  -h, --help        show this help message and exit
  --model           vae model to use: gamma | normal, default is normal
  --epoches N        number of total epochs to run, default is 10000
```

## Acknowledgments

* To Mr. Altosaar for helping me on some questions I had in his implementation and several other questions in the subject of VAEs.