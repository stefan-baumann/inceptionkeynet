# InceptionKeyNet - Key Estimation in Music Recordings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5624477.svg)](https://doi.org/10.5281/zenodo.5624477)

This repository contains the reference implementation of the paper [Deeper Convolutional Neural Networks and Broad Augmentation Policies Improve Performance in Musical Key Estimation](https://archives.ismir.net/ismir2021/paper/000004.pdf), published at [ISMIR 2021](https://ismir2021.ismir.net/) as a conference paper by Stefan Andreas Baumann.

The paper contains both a new state-of-the-art CNN model for musical key estimation in recordings and an augmentation policy, which can be used during training of this model to increase performance.

## Usage
**Disclaimer**: The code in this repository is an adapted subset of a significantly larger project. As such, it sometimes is _significantly_ more extensive than it would need to be if you just want to quickly train/run a model. Due to copyright limitations, it is unfortunately neither possible to provide the dataset metadata with this repository (although instructions as to how to get them are provided), nor possible to include audio data or ways of gathering it apart from the GiantSteps datasets. 

**If you just want to transfer the InceptionKeyNet model to your own project**, simply copy the model definition from `inceptionkeynet.machine_learning.__models`, and train it like you would any other model. The exact training & preprocessing parameters are specified in `inceptionkeynet.tasks.train`, and the code for executing the preprocessing & augmentation steps can be copied from `inceptionkeynet.processing.transformers`. Please open an issue on GitHub if you run into any problems. If there is enough demand and I find the time, I might provide a separate minimal sample for this usecase.

**If you want to use the project as-is**, you'll want to use the scripts provided in `inceptionkeynet.tasks`. The recommended execution order is `inceptionkeynet.tasks.generate_datasets` to generate the dataset files from their respective sources (see `data/datasets/source_files/README.md` for how to obtain those source files); then `inceptionkeynet.tasks.run_audio_mining` to obtain the audio files for the respective datasets (you'll have to implement an `inceptionkeynet.data_mining.AudioMiner` to get the corresponding audio files from your own database/source). Then, you can run `inceptionkeynet.tasks.run_preprocessing` to perform the necessary ahead-of-time preprocessing for training (be aware - this will create _a lot_ of data), and finally run the training with `inceptionkeynet.tasks.train`.

## Cite
[InceptionKeyNet paper](https://archives.ismir.net/ismir2021/paper/000004.pdf)
```
@inproceedings{stefan_a_baumann_2021_5624477,
  author       = {Stefan A Baumann},
  title        = {{Deeper Convolutional Neural Networks and Broad 
                   Augmentation Policies Improve Performance in
                   Musical Key Estimation}},
  booktitle    = {{Proceedings of the 22nd International Society for 
                   Music Information Retrieval Conference}},
  year         = 2021,
  pages        = {42-49},
  publisher    = {ISMIR},
  address      = {Online},
  month        = nov,
  venue        = {Online},
  doi          = {10.5281/zenodo.5624477},
  url          = {https://doi.org/10.5281/zenodo.5624477}
}
```

## Contact
[stefan-baumann@outlook.com](mailto:stefan-baumann@outlook.com)
