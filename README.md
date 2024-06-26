# Self-training
This repository contains code for the PhD thesis: ["A Study of Self-training Variants for
Semi-supervised Image Classification"] (https://researchcommons.waikato.ac.nz/handle/10289/14678) and publications. This is PyTorch variant of my original repo from
[https://github.com/attaullah/self-training](https://github.com/attaullah/self-training)
1. Sahito A., Frank E., Pfahringer B. (2019) Semi-supervised Learning Using Siamese Networks. In: Liu J., Bailey J. 
(eds) AI 2019: Advances in Artificial Intelligence. AI 2019. Lecture Notes in Computer Science, vol 11919. Springer, 
Cham. [DOI:978-3-030-35288-2_47](https://link.springer.com/chapter/10.1007/978-3-030-35288-2_47) 
2. Sahito A., Frank E., Pfahringer B. (2020) Transfer of Pretrained Model Weights Substantially Improves Semi-supervised
Image Classification. In: Gallagher M., Moustafa N., Lakshika E. (eds) AI 2020: Advances in Artificial Intelligence.
AI 2020. Lecture Notes in Computer Science, vol 12576. Springer, Cham. 
[DOI:978-3-030-64984-5_34](https://doi.org/10.1007/978-3-030-64984-5_34)
3. Sahito, A., Frank, E., Pfahringer, B. (2022). Better Self-training for Image Classification Through Self-supervision. In: Long, G., Yu, X., Wang, S. (eds) AI 2021: Advances in Artificial Intelligence. AI 2022. Lecture Notes in Computer Science(), vol 13151. Springer, Cham 
[DOI:978-3-030-97546-3_52]([https://arxiv.org/abs/2109.00778](https://doi.org/10.1007/978-3-030-97546-3_52))

## Getting started
Start with cloning the repo:
```bash
git clone https://github.com/attaullah/py_self-supervised.git
cd py_self-supervised/
```
### Environment setup
For creating a conda environment,  the yml  file `environment.yml` is provided for replicating the setup.

```bash
conda env create -f environment.yml
conda activate environment
```

### Data preparation
MNIST, Fashion-MNIST, SVHN, and CIFAR-10 datasets are loaded using  torchvision. 

For the PlantVillage dataset [1] please follow instructions at
 [plant-disease-dataset](https://github.com/attaullah/downsampled-plant-disease-dataset). The downloaded files are  saved in the `data/` directory. 


### Repository Structure
Here is a brief overview of the repository.

-`data_utils/`: provides helper functions for loading datasets, details of  datasets like the number of initially labelled
examples: `n_label`, selection percentage of pseudo-labels for each iteration of self-training: `selection_percentile`,
parameter `sigma` for LLGC and data loaders for training the network model.

-`losses/`: implementation of ArcFace, Contrastive, and Triplet loss.

-`models/`: provides the implementation of custom `SIMPLE` convolutional network model used for MNIST, Fashion-MNIST, and 
SVHN, `SSDL` another custom convolutional network model used for CIFAR-10 and PlantVillages.

-`utils/`: contains the implementation of LLGC and other utility functions.


## Example usage
Training can be started using the `train.py` script. Details of self-explanatory command-line 
arguments can be seen by passing `--helpfull` to it.


```bash

       USAGE: train.py [flags]
flags:

flags:
  --batch_size: size of mini-batch
    (default: '128')
    (an integer)
  --confidence_measure: <1-nn|llgc>: confidence measure for pseudo-label selection.
    (default: '1-nn')
  --dataset: <cifar10|svhn|plant32|plant64|plant96>: dataset name
    (default: 'cifar10')
  --epochs: initial training epochs
    (default: '200')
    (an integer)
  --epochs_per_m_iteration: number of epochs per meta-iterations
    (default: '200')
    (an integer)
  --gpu: gpu id
    (default: '0')
  --lbl: <knn|lda|rf|lr>: shallow classifiers for labelling for metric learning losses
    (default: 'knn')
  --lr: learning_rate
    (default: '0.003')
    (a number)
  --lr_sched: <cosine|>: lr_sched: None, cosine,  .
  --lt: <cross-entropy|triplet|arcface|contrastive>: loss_type: cross-entropy, triplet,  arcface or
    contrastive.
    (default: 'cross-entropy')
  --margin: margin for triplet loss calculation
    (default: '1.0')
    (a number)
  --meta_iterations: number of meta_iterations
    (default: '25')
    (an integer)
  --network: <wrn-28-2|resnet18|vgg16|resnet34|resnet20|resnet50|ssdl|vit>: network architecture.
    (default: 'wrn-28-2')
  --opt: <adam|sgd|sgdw|rmsprop>: optimizer.
    (default: 'adam')
  --pre: prefix for log directory
    (default: '')
  --[no]self_training: apply self-training
    (default: 'false')
  --[no]semi: True: N-labelled training, False: All-labelled training
    (default: 'true')
  --verbose: verbose
    (default: '1')
    (an integer)
  --[no]weights: random or ImageNet pretrained weights
    (default: 'false')
 ```


## Citation Information 
If you use the provided code, kindly cite our paper.
```
@phdthesis{attaullah2021self,
    title    = {A study of self-training variants for semi-supervised image classification},
    school   = {The University of Waikato},
    author   = {Attaullah Sahito},
    year     = {2021}, 
    url      = {https://hdl.handle.net/10289/14678}
}

@inproceedings{sahito2022better,
  title={Better self-training for image classification through self-supervision},
  author={Sahito, Attaullah and Frank, Eibe and Pfahringer, Bernhard},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={645--657},
  year={2022},
  organization={Springer}
}
```
## References
1. J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep 
Convolutional Neural Network”, Mendeley Data, V1, doi: 10.17632/tywbtsjrjv.1

## License
[MIT](https://choosealicense.com/licenses/mit/)
