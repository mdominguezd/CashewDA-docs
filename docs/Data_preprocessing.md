---
sidebar_position: 2
---

# Raw data pre-processing

## Cashew crop labels

 The reference dataset consists of polygons where cashew crops have been estimated to be present in 2016 for the countries of Tanzania and Ivory Coast. Moreover, these polygons account for areas of 6,260 and 73,510 hectares respectively. Additionally, the majority of the labeled cashew crop polygons that will be used in each domain are present in the regions of Mtwara, Tanzania (5,035 hectares) and Vallée du Bandama, Ivory Coast (25,912 hectares), both of which are the areas where more cashew is produced in each country.

 ![labels](/img/label_density.png)
 
The data used for reference lables of Cashew crops in Ivory Coast and Tanzania can be downloaded from [here](https://www.dropbox.com/scl/fo/y8rp4ne0q8vfo5wbr1v4x/h?rlkey=vtyhx69b3jcckpxcnuxt2grh7&dl=0).

## Input images

The datasets used as input data for the training of the deep learning algorithms will mainly consist of of images from Planet Norway’s International Climate and Forest Initiative ([NICFI](https://www.planet.com/nicfi/?gad_source=1&gclid=CjwKCAiAk9itBhASEiwA1my_67JOFQ8L4DPicJ47w-b_bGBjLBM1SymMjL91UsJVmB5jSRwKsoedZxoCb2sQAvD_BwE)). These images were downloaded using google earth engine (See [here](https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_africa) the image collection).

For a detailed way to download patches from google earth engine go to [DataGathering](./Dataset/DataGathering).
