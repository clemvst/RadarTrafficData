
# Rapport Final - Machine Learning

<img src="image/timelapse_austin.png" width="634px" height="403px">

Iris Dumeur & Clémence Vast

14/12/2020 - Ecole des Mines de Nancy

## Rappel des consignes
Data: Kaggle "Radar Traffic Data. Traffic data collected from radar sensors deployed by the City of Austin."
- Build a deep learning model that predicts the traffic volume.
- Do not use any other data source ! (in particular, no openstreetmap data...)
- If dataset too big for your laptop, reduce dataset size
- You may do it alone, or within a group of 2 (preferred)
- But you'll get an individual note !
- Score:
    - GIT (4): distribution of the workplan, individual contribution
    - REPORT (10): model(s), experimental process, citations
    - CODE (6): correctness, readability
- Use deep learning models
- Do not spend too much time on data pre-processing
- Most important: modeling, training, evaluating

## Introduction
Rapport de Machine Learning sur le projet Kaggle Radar Traffic Data.

## Analyse des données
Chaque radar fournit des données toutes les 15 minutes. Chaque nouvel apport de données par radar correspond à une ligne sur le CSV, incluant : 
 - le nom du radar
 - la position géographique du radar
 - la date et l’heure de soumission de données (sous différents formats)
 - la direction de circulation des voitures détectées
 - le nombre de voitures détectées
 
Au début de notre projet, nous décidons de nous concentrer sur les données provenant d'un seul radar, afin d'avoir moins
de données à traiter et un modèle plus simple à entrainier.
To describe the data we first focus on explore the data for one radar : ' CAPITAL OF TEXAS HWY / LAKEWOOD DR'


Décrire le nombre d'entités dans les colonnes

Décrire pourquoi on en supprime certaines


| Nom de la colonne | Description | Nombre d'entités | Remarques particulieres |
| ----------------- | ----------- |----------------- | ----------------------- |
| Location     | string, Nom du radar, un nom correspond à une localisation précise | 23  |    |
| location_latitude | latitude de la position du radar |    |    |
| location_logitude | logitude de la position du radar |    |    |
| Year | Année d'acquisition |    |    |
| Month | Mois d'acquisition |    |    |
| Day | Jours d'acquisition |    |    |
| Day of Week | Jours de la semaine, entier allant de de 0 à 6 |    |    |
| Direction | None, NB ou *** , indique la direction du passage des voitures compté par le radar |    |    |
| Volume| Nombre refletant le passage des voitures au niveau du radar entre deux instants|    |    |

On étudie ensuite le comportement des données pour un seul radar : radar1

' CAPITAL OF TEXAS HWY / LAKEWOOD DR'. Deux directions sont disponibles 
On étudie le comportenement selon le jour de la semaine : . On remarque qu'il existe plusieurs données pour un même jour 
d'une même année à la même heure d'acquisition (**insérer screen ??**). On agrège les données ayant la même exacte date
d'acquisition.
**insérer graph de image repertoire**
**Analyser graph de image repertoire**

##Préparer les données
Le traitement des données a été fait de manière à pouvoir choisir la taille des données en entrée du modèle ainsi que 
les labels. Nous souhaitons expérimenter sur différentes échelles, par exemple nous mennerons
comme expérience : 

| Taille total du dataset   | Taille input x  | Taille label y  | 
| ----------------- |  ----------------- | ----------- |
| 1 mois  | 6 jours de données | 1 jours de données| 
| 1 an  | 6 jours de données | 1 jours de données| 
| 1 an | 1 semaine de données| 1 mois de données | 

Afin de rendre ces expériences rapidement implémentables, les fonctions du fichier open_data sont
fortement paramétrables. Par ailleurs sachant que les acquisitions sont faites toutes les 15 minutes, nous
souhaitons également choisir en fonction du dataset que l'on souhaite construire l'écart temporel
entre chaque données du dataset.
De plus, il est également possible que des données ne soient pas disponibles pour certains jours. Nous ne construisons 
alors pas les batch dépendant de ces données.



## Démarches mises en oeuvre
Mettre ici les différentes démarches testées, leurs objectifs, avec un petit résumé de pourquoi nous avons décidé ou non de poursuivre?

Dans notre démarche nous souhaitons partir du modèle simple possible, et le complexifier en fonction des résultats obtenus
sur le premier modèle simple. 

Pour complexifier le modèle regarder : https://towardsdatascience.com/encoder-decoder-model-for-multistep-time-series-forecasting-using-pytorch-5d54c6af6e60
 Prend en compte des features supplémentaire pour la prédiction. Ici pourrait être envisagé par exemple pour ajouter le jour de la semaine
 

## Résultats et analyse

## Conclusion
