# Merger_Ringdown_Test
This repository contains the codes uded for the paper [arXiv number to be added].

The main codes are contained in the [Modules](cpacilio/Merger_Ringdown_Test/Modules/) folder:
- [Ringdown2.py](cpacilio/Merger_Ringdown_Test/Modules/Ringdown2.py): contains the routines to generate ringdown waveforms and compute their signal-to-noise ratio w.r.t. a noise curve;
- [qnm_cvae.py](cpacilio/Merger_Ringdown_Test/Modules/qnm_cvae.py): implements the conditional variational autoencoder class, as well as the routine producing the PP-plot to evaluat the final performances of the autoencoder; the autoencoder is coded in `PyTorch`;
- [train_script.py](cpacilio/Merger_Ringdown_Test/Modules/train_script.py): a `Python` script to train the autoencoder.

The notebook [cvae_notebook_git.ipynb](cpacilio/Merger_Ringdown_Test/cvae_notebook_git.ipynb) is organized in sections, each section performing a different task: generate datasets, produce plots, test the network, etc. The notebook was originally run on the `Google Colab` platform. 
