---
title: 'arfpy: A Python package for adversarial random forests'
tags:
- Python
- machine learning
- generative modeling
- random forest
- data sythesis
date: 22 September 2023
output: pdf_document
authors:
- name: Kristin Blesch
  orcid: 0000-0001-6241-3079
  corresponding: yes
  affiliation: 1, 2
  
- name: Marvin N. Wright
  orcid: 0000-0002-8542-6291
  affiliation: 1, 2, 3

bibliography: references.bib
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.

affiliations:
- name: Leibniz Institute for Prevention Research and Epidemiology -- BIPS, Bremen, Germany
  index: 1
- name: Faculty of Mathematics and Computer Science, University of Bremen, Bremen, Germany
  index: 2
- name:  Department of Public Health, University of Copenhagen, Copenhagen, Denmark
  index: 3


---

# Summary

Generative modeling is a challenging task in machine learning that aims to synthesize new data which is similar to a set of given data. State of the art are computationally intense and tuning-heavy algorithms such as generative adversarial networks [@goodfellow2014;@xu2019], variational autoencoders [@kingma2014], normalizing flows [@rezende2015], diffiusion models [@ho2020] or transformers [@vaswani2017]. A much more lightweight procedure is to use an Adversarial Random Forest (ARF) [@watson2023].  ARFs achieve competitive performance in generative modeling in much faster runtime [@watson2023] and are especially useful for data that comes in a table format, i.e., tabular data. That is because ARFs are based on random forests [@breiman2001] that leverage the advantages that tree-based methods have over neural networks on tabular data (see @grinsztajn2022) for generative modeling. Further, as part of the procedure, ARFs give access to the estimated joint density, which is useful for several other fields of research, e.g., unsupervised machine learning. For the task of density estimation, ARFs have been demonstrated to yield remarkable results as well [@watson2023].  Hence, ARFs are a promising methodological contribution to the field of generative modeling and density estimation. To reach scholars in these fields that are predominantely based in python, and a broad audience more generally, a fast and userfriendly implementation of ARFs in python is highly desirable, which is provided by the software package `arfpy`. 

# Statement of need

The package `arfpy` implements density estimation and generative modeling with ARFs in python. ARFs have been introduced with a solid theoretical background, yet do not have to compromise on a complex algorithmic structure and instead are a low-key algorithm that does not require extensive hyperparameter tuning [@watson2023]. This makes the methodology attractive for both scholars conducting rather theoretical research in statistics, e.g., density estimation, as well as practitioners from other fields that need to generate new data samples. Typical use cases of such synthesized data samples are, for example, the imputation of missing values, data augmentation or the  conduct of analyses that respect data protection rules. With the speciality of ARFs being particulary suitable for tabular data, including a natural incorporation of both continuous and categorical features, the straightforward python implementation of ARFs provides a convenient algorithm to a broad audience from different fields. 

ARFs have already gained some attention in the scientific community [@nock2023], however, the paper by @watson2023 provides the audience with only an \texttt{R} software package. The machine learning and generative modeling community however is mostly using python as a programming language. We aim to fill this gap with the presentend python implementation of ARFs. `arfpy` is inspired by the \texttt{R} implementation `arf` [@wright2023], but transfers the algorithmic structure to match the class-based structure of python code and takes advantage of computationally efficient python functions. This is more robust and convenient to users than calling fragile wrappers like `rpy2` @rpy2 that attempt to make \texttt{R} code running in python. The benefits of a direct python implementation of ARFs for the generative modeling community have already been recognized by now. For example, `arfpy` is integrated in the data synthesizing framework `synthcity` (@synthcity). 


For interested readers, we briefly describe the ARF algorithm below, but refer to [@watson2023] for details. First, naive synthetic data is generated (initial generation step) by sampling from the marginal distributions of the features.  Then, an unsupervised random forest [@shi2006] is fit to distinguish this synthetic from real data (initial discrimination step). By doing so, the unsupervised random forest learns the dependency structure in the data. Using this forest, we can sample observations from the leaves to generate updated synthetic data (generation step). Subsequently, a new unsupervised random forest is fit to differentiate between synthetic and real data (discrimination step). Drawing on the adversarial idea of GANs, this iterative procedure of data generation and discrimination will be repeated until the discriminator cannot distinguish between generated and real data anymore. At this stage, the accuracy of the forest will be $\leq 0.5$ and the forest is assumed to have converged, which implies mutually independent features in the terminal nodes. This facilitates the endeavor of density estimation and generative modeling drastically, as it allows us to formulate the univariate density for each feature separately and then combine them to the joint density, instead of having to model multivariate densities. For data generation, we can use this trait to sample a new observation by drawing a leaf from the forest of the last iteration step and use the data distributions with parameters estimated from that leaf to sample each feature separately. 

Summarizing our contribution, `arfpy` introduces density estimation and generative modeling with ARFs to python. This enables practitioners from a wide variety of fields to generate fast and reliable synthetic data and density estimates using python as a programming language. 

# Acknowledgements
This work was supported by the German Research Foundation (DFG), Emmy Noether Grant 437611051. We thank David S. Watson and Jan Kapar for their contributions to establishing the theoretical groundwork of adversarial random forests. 

# References


