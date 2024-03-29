---
sidebar_position: 1
---

# Introduction to the problem

Here you can find all of the documentation for the code used for the development of my MSc. thesis ***Bridging domains: Assessment of Domain Adaptation methods for mapping Cashew crops in Africa***. The repository with the code can be found [**here**](https://github.com/mdominguezd/Thesis_model_training). 

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

* What are the underlying factors contributing to the domain shift between source and target and how much does it affect the accuracy of a cashew crop mapping model trained only with source data and applied in the target domain?
* To what extent do adversarial domain adaptation methods impact the accuracy and the generalization capability of semantic segmentation models used for cashew crop mapping?
* How can a web application be designed to evaluate cashew crop mapping models in Africa and encourage user participation in generating annotated data?

