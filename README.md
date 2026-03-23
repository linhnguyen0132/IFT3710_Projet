# Species Classification (GeoLifeCLEF) - IFT3710

This project was developped for the class **IFT3710 - Advanced projects in machine learning**. The objectif is to predict the species in France with data available in Kaggle competitions GeoLifeCLEF 2021 and 2022 and eventually classify the animal as endangered or not.

## Implemented Models

We explored and compared 2 main models with different approches in this project. For each case, we tested the model on data from 2021 and 2022 to compare performances.

### XGBoost

Due to limitations related to RAM and allocated time on Calcul Québec clusters and the model's requirements, only a random sample of 300 or 500 with more than 10 observations was used during training and validation.

#### XGBoost plain
**Approach**     : Base model to set a baseline and K-Means to group data points
**Result**       : Reduced biais due to *latitude* and *longitude*

#### XGBoost with Polynomial Features
**Approach**     : Created interactions between top 5 most important features.
**Result**       : 

#### XGBoost with PCA
**Approach**     : Reduced 19 feactures to 10 components
**Pre-treatment**: Used 'StandardScaler' to normalize data to avoid problems due to different units

*French version*
# Prédiction de la Distribution des espèces (GeoLifeCLEF 2021 -2022) - IFT3710

Ce projet a été réalisé dans le cadre du cours **IFT3710 - Projets avancés en apprentissage automatique**. L'objectif est de prédire l'espèce présente en France en utilisant les données disponibles des compétitions GeoLifeCLEF de 2021 et 2022 sur Kaggle et éventuellement de voir si l'animal est en danger.

## Modèles implémentés

Nous avons exploré et comparé 2 modèles principaux avec plusieurs approches dans ce projet et testé le même modèle dans chaque cas sur les données 2021 et 2022 afin de comparer les performances.

### XGBoost
À cause des limitations au niveau de la mémoire RAM et de temps sur le cluster de Calcul Québec et les exigences du modèle, nous avons pris des **échantillons aléatoires de 300 et 500 espèces** avec plus de 10 observations.

#### XGBoost brut
**Approche**      : Modèle de base pour avoir un baseline et utilisation de l'algo K-Means pour regrouper les latitude et longitude des observations. 
**Résultat**      : Moins de biais sur les caractéristiques *latitude* et *longitude*

#### XGboost avec Polynomial Features
**Approche**      : Création d'interactions entre les top 5 variables bioclimatiques importantes.
**Résultat**      :

#### XGBoost avec PCA
**Approche**      : Réduction des 19 variables à 10 composantes principales.
**Prétraitement** : Utilisation de 'StandardScaler' pour normaliser les données afin d'éviter des problèmes d'unités.