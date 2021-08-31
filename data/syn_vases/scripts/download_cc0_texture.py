import os
import random
import torch
import requests
import json
import zipfile


## Download json file
cc0_tex_dir = '../cc0_textures'
json_fpath = os.path.join(cc0_tex_dir, 'cc0_textures_full_json.json')
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

if not os.path.isfile(json_fpath):
    print('Downloading json file...')
    os.makedirs(cc0_tex_dir, exist_ok=True)
    json_url = 'https://cc0textures.com/api/v1/full_json'
    r = requests.get(json_url, headers=headers)
    with open(json_fpath, 'wb') as outfile:
        outfile.write(r.content)
all_json = json.load(open(json_fpath))


## Download assets
all_assets = all_json['Assets']
asset_type = 'PhotoTexturePBR'
out_dir = os.path.join(cc0_tex_dir, asset_type)
download_type = '2K-JPG'
sfxs = {'zip': '.zip'}

print('Downloading assets...')
total_num = len(all_assets)
for i, (name, instance) in enumerate(all_assets.items()):
    if i % 10 == 0:
        print('%d / %d' %(i, total_num))
    if instance['AssetDataTypeID'] == asset_type:
        try:
            download_link = instance['Downloads'][download_type]['PrettyDownloadLink']
            ftype = instance['Downloads'][download_type]['Filetype']
            r = requests.get(download_link, allow_redirects=True, headers=headers)

            out_fold = os.path.join(out_dir, name)
            zip_fpath = os.path.join(out_fold, name+sfxs[ftype])
            os.makedirs(out_fold)
            open(zip_fpath, 'wb').write(r.content)

            zipfile.ZipFile(zip_fpath, 'r').extractall(out_fold)

        except:
            print('Failed! Name:', name)


## Train test split
print('Spliting them into train/test/val...')
asset_list = sorted(os.listdir(out_dir))
random.shuffle(asset_list)
total_num = len(asset_list)
test_num = int(0.1*total_num)
for i, mat_fold in enumerate(asset_list):
    if i < test_num:
        sp = 'test'
    else:
        sp = 'train'
    sp_dir = os.path.join(out_dir, sp)
    os.makedirs(sp_dir, exist_ok=True)
    os.rename(os.path.join(out_dir, mat_fold), os.path.join(sp_dir, mat_fold))
