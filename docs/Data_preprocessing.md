---
sidebar_position: 2
---

# Raw data pre-processing

## Cashew crop labels

 The reference dataset consists of polygons where cashew crops have been estimated to be present in 2016 for the countries of Tanzania and Ivory Coast. Moreover, these polygons account for areas of 6,260 and 73,510 hectares respectively. Additionally, the majority of the labeled cashew crop polygons that will be used in each domain are present in the regions of Mtwara, Tanzania (5,035 hectares) and Vallée du Bandama, Ivory Coast (25,912 hectares), both of which are the areas where more cashew is produced in each country.

 ![labels](/img/label_density.png)
 
The data used for reference labels of Cashew crops in Ivory Coast and Tanzania can be downloaded from [here](https://www.dropbox.com/scl/fo/y8rp4ne0q8vfo5wbr1v4x/h?rlkey=vtyhx69b3jcckpxcnuxt2grh7&dl=0).

## Input images

The datasets used as input data for the training of the deep learning algorithms will mainly consist of of images from Planet Norway’s International Climate and Forest Initiative ([NICFI](https://www.planet.com/nicfi/?gad_source=1&gclid=CjwKCAiAk9itBhASEiwA1my_67JOFQ8L4DPicJ47w-b_bGBjLBM1SymMjL91UsJVmB5jSRwKsoedZxoCb2sQAvD_BwE)). These images were downloaded using google earth engine (See [here](https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_africa) the image collection).

For a detailed way to download patches from google earth engine go to [DataGathering](./Dataset/DataGathering).
## Get ready 

### Download the pre-processed datasets

If you want to access the raw data and pre-process it yourself click [here](./Dataset/DataGathering).

#### Directly through Dropbox

Click this [link](https://www.dropbox.com/scl/fo/ozyb8j2qx21secw2scocq/h?rlkey=1gyptmas8bnxdhmc0j7toi17r&dl=0) to access the data and download it directly to your local machine.

#### Using bash

See [DS_Download.sh](https://github.com/mdominguezd/Thesis_model_training/blob/main/DS_Download.sh)

```bash
#!/bin/bash

# Download three splits for Tanzania
wget -O Tanzania1.zip -q --show-progress https://www.dropbox.com/scl/fi/ueafajskwc5qckynwl7af/PlanetTanzania_18_patch256_split1.zip?rlkey=mf68gq19rs7qyj03rj6tx5250&dl=0 
wget -O Tanzania2.zip -q --show-progress https://www.dropbox.com/scl/fi/pj7su9q8f422k06zy6oyf/PlanetTanzania_18_patch256_split2.zip?rlkey=8xw9al4h4i46h9c9b3azqa8e4&dl=0 
wget -O Tanzania3.zip -q --show-progress https://www.dropbox.com/scl/fi/mhipwkf42vl8dluo4d6uu/PlanetTanzania_18_patch256_split3.zip?rlkey=2jtyw1n4u9wjibn3jnvduo717&dl=0 

# Download three splits for Ivory Coast
wget -O IvoryCoast1.zip -q --show-progress https://www.dropbox.com/s/mn53y84ahj4y0vj/PlanetIvoryCoast_18_patch256_split1.zip?dl=0
wget -O IvoryCoast2.zip -q --show-progress https://www.dropbox.com/scl/fi/6pxnc69et16gncsdd11ys/PlanetIvoryCoast_18_patch256_split2.zip?rlkey=xs0ae6o6nnpelkdi5gthq0x8v&dl=0
wget -O IvoryCoast3.zip -q --show-progress https://www.dropbox.com/scl/fi/37ool0ctxlc43quovnvs7/PlanetIvoryCoast_18_patch256_split3.zip?rlkey=i8fs18b99vh0wmdvtj25gyhg5&dl=0
```


