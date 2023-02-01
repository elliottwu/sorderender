# De-rendering the World's Revolutionary Artefacts
#### [Project Page](https://sorderender.github.io/) | [Video](https://youtu.be/pxkYyyw02H0) | [Paper](https://arxiv.org/abs/2104.03954)

In CVPR 2021

[Shangzhe Wu](https://elliottwu.com/)<sup>1,4</sup>, [Ameesh Makadia](http://ameeshmakadia.com/index.html)<sup>4</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>2</sup>, [Noah Snavely](https://www.cs.cornell.edu/~snavely/)<sup>4</sup>, [Richard Tucker](https://research.google/people/RichardTucker/)<sup>4</sup>, [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)<sup>3,4</sup>

<sup>1</sup> University of Oxford, <sup>2</sup> Stanford University, <sup>3</sup> University of California, Berkeley, <sup>4</sup> Google Research


https://user-images.githubusercontent.com/15921740/130113055-30266eee-73a0-4735-9cbe-18e6bf055058.mp4


We propose a model that de-renders a single image of a vase into shape, material and environment illumination, trained using only a single image collection, without explicit 3D, multi-view or multi-light supervision.


## Setup (with conda)

### 1. Install dependencies:
```
conda env create -f environment.yml
```
OR manually:
```
conda install -c conda-forge matplotlib opencv scikit-image pyyaml tensorboard
```


### 2. Install [PyTorch](https://pytorch.org/):
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```
*Note*: The code is tested with PyTorch 1.4.0 and CUDA 10.1. A GPU version is required, as the [neural_renderer](https://github.com/daniilidis-group/neural_renderer) package only has a GPU implementation.


### 3. Install [neural_renderer](https://github.com/daniilidis-group/neural_renderer):
This package is required for training and testing, and optional for the demo. It requires a GPU device and GPU-enabled PyTorch.
```
pip install neural_renderer_pytorch==1.1.3
```

*Note*: If this fails or runtime error occurs, try compiling it from source. If you don't have a gcc>=5, you could one available on conda: `conda install gxx_linux-64=7.3`.
```
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
python setup.py install
```


## Datasets
### 1. Metropolitan Museum Vases
This vase dataset is collected from [Metropolitan Museum of Art Collection](https://www.metmuseum.org/art/collection) through their [open-access API](https://metmuseum.github.io/) under the [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/). It contains 1888 training images and 526 testing images of museum vases with segmentation masks obtained using [PointRend](https://arxiv.org/abs/1912.08193) and [GrabCut](https://dl.acm.org/doi/10.1145/1015706.1015720).

Download the preprocessed dataset using the provided script:
```
cd data && sh download_met_vases.sh
```

### 2. Synthetic Vases
This synthetic vase dataset is generated with random vase-like shapes, poses (elevation), lighting (using spherical Gaussian) and shininess materials. The diffuse texture is generated using the texture maps provided in [CC0 Textures](https://cc0textures.com/) (now called [ambientCG](https://ambientcg.com/)) under the [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/).

Download the dataset using the provided script:
```
cd data && sh download_syn_vases.sh
```

We also provide the scripts for downloading [CC0 Textures](https://cc0textures.com/) and generating this dataset in `data/syn_vases/scripts/`. Note the script uses API V1 of CC0 Textures to download the texture maps, which appears outdated already. Many assets have now been removed. [API V2](https://help.ambientcg.com/04-API/API_v2.html) has been released. Please check and adapt the code to the new API.

## Pretrained Models
Download the pretrained models using the scripts provided in `pretrained/`, eg:
```
cd pretrained && sh download_pretrained_met_vase.sh
```


## Training and Testing
Check the configuration files in `configs/` and run experiments, eg:
```
python run.py --config configs/train_met_vase.yml --gpu 0 --num_workers 4
```


## Evaluation on Synthetic Vases
After generating the results on the test set (see `configs/test_syn_vase.yml`), check and run:
```
python eval/eval_syn_vase.py
```


## Render Animations
To render animations of rotating vases and rotating light, check and run this script:
```
python render_animation.py
```


## Citation
```
@InProceedings{wu2021derender,
  author={Shangzhe Wu and Ameesh Makadia and Jiajun Wu and Noah Snavely and Richard Tucker and Angjoo Kanazawa},
  title={De-rendering the World's Revolutionary Artefacts},
  booktitle = {CVPR},
  year = {2021}
}
```
