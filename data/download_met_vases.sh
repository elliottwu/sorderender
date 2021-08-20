# This vase dataset is collected from [Metropolitan Museum of Art Collection](https://www.metmuseum.org/art/collection) through their [open-access API](https://metmuseum.github.io/) under the [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/). It contains 1888 training images and 526 testing images of museum vases with segmentation masks obtained using [PointRend](https://arxiv.org/abs/1912.08193) and [GrabCut](https://dl.acm.org/doi/10.1145/1015706.1015720).

echo "----------------------- downloading met vase dataset -----------------------"
curl -o met_vases.zip "https://www.robots.ox.ac.uk/~vgg/research/sorderender/data/met_vases.zip" && unzip met_vases.zip
