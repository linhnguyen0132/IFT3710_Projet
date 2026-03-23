# Species Classification (GeoLifeCLEF) - IFT3710

## Description

This project was developped for the class IFT3710 - Advanced projects in machine learning. The objectif is to predict the species in France with data available in Kaggle competitions GeoLifeCLEF 2021 and 2022 and eventually classify the animal as endangered or not.

## Getting Started

### Dependencies
* **Operating System** : Linux

* **Python version**   : Python 3.11 or more recent

* **Librairies**       :
    * `pandas`, `numpy`: Data manipulation
    
    * `scikit-learn`   : StandardScaler, PCA, K-Means

    * `xgboost`        : Model

    * `matplotlib`     : Graphics

### Installation

* Clone the repository on your machine / cluster 

```bash
git clone https://github.com/linhnguyen0132/IFT3710_Projet.git
```

### Executing program

* Ensure that dataset paths are updated in `.py`

* Execute using `.sh` files, make sure to modify variables as needed.

```bash
sbatch job.sh
```
----------------------------------------------------------------------------
*FRENCH VERSION*
# Prédiction de la Distribution des espèces (GeoLofeCLEF 2021-2022) - IFT3710

## Description

Ce projet a été réalisé dans le cadre du cours IFT3710 - Projets avancés en apprentissage automatique. L'objectif est de prédire l'espèce présente en France en utilisant les données disponibles des compétitions GeoLifeCLEF de 2021 et 2022 sur Kaggle et éventuellement de voir si l'animal est en danger.

## Initalisation

### Dépendences

* **Système d'exploitation** : Linux

* **Version Python**  : Python 3.11 ou 

* **Librairies**      :

    * `panda`, `numpy`: Manipulation et analyse des données.

    * `scikit-learn`  : StandardScaler, PCA et K-Means.

    * `xgboost`       : modèle

    * `matplotlib`    : Courbes et graphiques

### Installation
* Cloner le dépôt sur votre machine / cluster

```bash
git clone https://github.com/linhnguyen0132/IFT3710_Projet.git
```

### Exécution 

* Mettre à jour les chemins pour les données dans les fichiers `.py`

* Utiliser les fichiers .sh en modifiant les variables selon les besoins (mémoire, temps, etc.)

```bash
sbatch job.sh
```
