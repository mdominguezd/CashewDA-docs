---
sidebar_position: 1
---

# Introduction to the problem

Here you can find all of the documentation for the code used for the development of my MSc. thesis ***Domain Adaptation for mapping cashew crops in Africa***. The repository with the code can be found [**here**](https://github.com/mdominguezd/Thesis_model_training). 

The following subsections describe the problem that I tried to solve during my thesis, hope you find it useful and I am all ears if you have any additional suggestions to improve this project.

## Food insecurity

Food insecurity is one of the main causes of death in developing countries  (Zaini et al., 2019). In 2022, it is estimated that between 690 and 783 million people suffered hunger in the world (FAO, 2023). These estimations help infer that the world is far from achieving the UN's sustainable development goal of no hunger by 2030. Moreover, food security today is not a question of food scarcity, but rather a by-product of poverty and socio-ecnomic inequality in developing nations (Boliko, 2019). A way to tackle poverty in these countries is by boosting the growth of commodity (cash) crops.

## Deforestation

Cashew crops in Africa have become one of the main cash crops of the region. The continent produces more than half of the raw cashew nut that is produced globally and accounts for 90% of the raw cashew nut exports in the world (United Nations Conference on Trade and Development, 2021). These figures show the important influence that cashew crops have in African countries’ economies. Especially, due to the fact that cashew is mainly grown by smallholders. Nevertheless, the recent growth of cashew crops in these regions has, in some cases, been at the expense of the environment. In fact, commodity crops expansion represents one of the primary drivers of deforestation in tropical countries (DeFries et al., 2010). For the case of cashew plantations, in Benin it has been estimated that between 2015 and 2019 the growth of cashew plantations in protected areas has increased by 55% (Yin et al., 2023). These estimations have prompted major commodity crops importers to implement new rules to avoid the inclusion of products that have come from deforestation to their markets.

## Crop mapping

Multiple countries in the global north have started implementing new policies for commodity crop imports. For example, the European Union and the United States have recently passed a new deal and a new executive order respectively in which only deforestation-free products from some commodity crops are allowed in their markets (European comission, 2022; US Department of State, 2023). These new policies aim to lower the emissions of greenhouse gases associated with agriculture, forestry and other land uses (AFOLU), which represent between 20 and 24 % of the global emissions (OECD, 2021). Due to the global nature of this issue, automated methods for effective monitoring of deforestation, follow-up land use (FLU) and crop extension are paramount.

Automated methods to monitor deforestation and crop extension include a wide variety of techniques. Recently, Deep learning techniques have shown promising results performing these tasks regionally (Masolele et al., 2022; Yin et al., 2023). However, global monitoring is still a difficult task. Novel global approaches to monitor FLU using deep learning models have been conducted, however the results still show that regional models exhibit better results (Masolele et al., 2021). This is a result of the difficulty to generalize of most deep learning models.

## Domain adaptation

Domain Adaptation (DA) emerges as a transfer learning alternative to address both the generalization problem that deep learning models encounter, and the data scarcity characteristic of some domains (Ben-David et al., 2010). In a nutshell, DA makes it possible to apply one model that has been trained in one source domain (dataset) to a target domain (dataset) that has a different data distribution than the source domain and that may have less or may not have any labels for training (HassanPour Zonoozi & Seydi, 2023).

Multiple DA methods have been developed over the past few years to increase the generalization capabilities of deep learning models and decrease the large amount of labeled data dependency during training (HassanPour Zonoozi & Seydi, 2023). These methods have been categorized in multiple different ways. According to HassanPour Zonoozi & Seydi (2023), DA methods can be divided into three categories: Supervised, semi-supervised and unsupervised methods. This study will focus on the latter. Unsupervised Domain Adaptation (UDA) methods have been categorized as Generative, Adversarial or Hybrid (Xu et al., 2022).

In this thesis, domain adaptation techniques were implemented to semantic segmentation models for cashew crop mapping in Africa to assess their potential for increasing the generalization capabilities of the models.

## Research questions

* How much does the domain shift between source and target affect the accuracy of a cashew crop mapping model trained only with source data and applied in the target domain?
* To what extent do adversarial domain adaptation methods impact the accuracy and the generalization capability of semantic segmentation models used for cashew crop mapping?

## Get ready 

### Download the pre-processed datasets

If you want to access the raw data and pre-process it yourself click [here](./Dataset/DataGathering).

#### Directly through Dropbox

Click this [link](https://www.dropbox.com/scl/fo/ozyb8j2qx21secw2scocq/h?rlkey=1gyptmas8bnxdhmc0j7toi17r&dl=0) to access the data and download it directly to your local machine.

#### Using bash

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

