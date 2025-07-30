# Neural Dynamics of Language Production

This repository is under active development!

This project started during my internship at [Bases, Corpus, Langage (BCL) lab](https://bcl.cnrs.fr/?lang=en) under supervision of [Dr. Raphael Fargier](https://sites.google.com/site/raphaelfargierpage/home?authuser=0).

## Goals

The goal of this project is to study the spatiotemporal pattern of the language production processing (e.g. lexical selection) in the brain by integrating computational modeling, Neuroimaging, and Behavioral analysis.

## Task and Recordings:

The primary data that we are using is EEG data (128 channels) recorded during an auditory naming from definition task across lifespan (with participants age 11 to 88 years old). In this task the participant hears definition of a word (e.g., '*an animal that produce honey*') and has 2 seconds to overtly produce the correct word ('*Bee*'). Data acquisition has been done years ago in French from French speaking participants; for more information regarding the task read (['Fargier et al., 2023'](https://doi.org/10.3389/fpsyg.2023.1237523)). There are couple of more available datasets (e.g., picture naming) that might be included in the further analysis.

## Methods:

We believe that significant degree of the discrepancies in the field, in particular, regarding the temporal activation of the Neuro-cognitive processes underlying language (e.g., [Serial Vs Parallel theories](https://doi.org/10.1080/02643294.2023.2283239);), roots from traditional averaging approach using ERPs. Hence, in this project we seek to uncover the hidden neurocognitive mechanism with trial-level resolution. So far, the following probabilistic generative models has been considered: 1) Hidden Multivariate Pattern (HMP) which has been introduced by (["Weindel et al., 2024"](https://doi.org/10.1162/imag_a_00400)) 2) Hidden markov Models (HMM) which in the past years had an increase in adoption for neurocognitive modeling ; see for example [(Yu et al. 2023)](https://doi.org/10.1080/10400419.2023.2172871) where gICA has been used for feature extraction and HMM for temporal modeling. Depending on time and resources, We might also consider development of new HMM base computational models that can account for different psycholinguistic models ( serial, cascade, parallel).

## Results:

The preliminary results are illustrated for:\
- fitting HMP model and hyperparameter tunning in [`HMP_fitting_youngGroup.ipynb`](https://github.com/MohammadMMK/Neural_dynamics_of_language_production/blob/main/Hidden%20Multivariate%20Pattern(HMP)/HMP_fitting_youngGroup.ipynb)- HMP event's duration and topographies in [`HMP_visualization_young.ipynb`](https://github.com/MohammadMMK/Neural_dynamics_of_language_production/blob/main/Hidden%20Multivariate%20Pattern(HMP)/HMP_visualization_young.ipynb)- gICA topomaps in [`gICA_youngGroup.ipynb`](https://github.com/MohammadMMK/Neural_dynamics_of_language_production/blob/main/gICA_HMM/gICA_youngGroup.ipynb)

**More result coming soon ...**

## Acknowledgments:

I would like to express my deepest gratitude to Dr. Raphael Fargier for his invaluable supervision, continuous support, and insightful guidance throughout the course of this project. His expertise and unwavering encouragement have been instrumental to my development as a researcher.

I am also sincerely thankful to Dr. patricia reynaud-bouret and Dr. samuel deslauriers-gauthier for their advises and constructive feedback, which greatly contributed to shaping the direction and clarity of this work.

This project was initiated during my internship at the Brain, Cognition and Language Lab (BCL), and I am grateful for the collaborative environment and provided resources that made this work possible.