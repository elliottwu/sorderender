# This synthetic vase dataset is generated with random vase-like shapes, poses (elevation), lighting (using spherical Gaussian) and shininess materials. The diffuse texture is generated using the texture maps provided in [CC0 Textures](https://cc0textures.com/) under the [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/).

echo "----------------------- downloading synthetic vase dataset -----------------------"
curl -o syn_vases.zip "https://www.robots.ox.ac.uk/~vgg/research/sorderender/data/syn_vases.zip" && unzip syn_vases.zip
