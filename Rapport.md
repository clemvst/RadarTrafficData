
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
 
To describe the data we first focus on explore the data for one radar : ' CAPITAL OF TEXAS HWY / LAKEWOOD DR'


Décrire le nombre d'entités dans les colonnes

Décrire pourquoi on en supprime certaines


| Nom de la colonne | Description | Nombre d'entités | Remarques particulieres |
| ----------------- | ----------- |----------------- | ----------------------- |
| Location     | Nom du radar, un nom correspond à une localisation précise | 23  |    |
|     |  |    |    |

On étudie ensuite le comportement des données pour un seul radar : radar1

' CAPITAL OF TEXAS HWY / LAKEWOOD DR'. Deux directions sont disponibles : 
On étudie le comportenement selon le jour de la semaine : . On remarque qu'il existe plusieurs données pour un même jour 
d'une même année à la même heure d'acquisition (**insérer screen ??**). On agrège les données ayant la même exacte date
d'acquisition.
**insérer graph de image repertoire**
**Analyser graph de image repertoire**

## Démarches mises en oeuvre
Mettre ici les différentes démarches testées, leurs objectifs, avec un petit résumé de pourquoi nous avons décidé ou non de poursuivre?


## Résultats et analyse

## Conclusion
