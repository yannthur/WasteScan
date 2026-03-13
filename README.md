# WasteScan — Classificateur de Déchets par Intelligence Artificielle

Interface de classification de déchets basée sur la vision par ordinateur, conçue pour identifier automatiquement la catégorie d'un déchet à partir d'une photographie et fournir des instructions de tri adaptées.

---

## Le problème

Le tri des déchets reste un défi quotidien pour une large partie de la population. Face à la diversité des matériaux — plastiques, métaux, textiles, déchets organiques, appareils électroniques — les consignes de tri varient selon les municipalités et ne sont pas toujours intuitives. Cette confusion entraîne des erreurs de tri fréquentes : des matériaux recyclables finissent en poubelle ordinaire, tandis que des déchets non recyclables contaminent les filières de collecte sélective.

Les campagnes de sensibilisation ne suffisent pas à répondre à ce problème en temps réel. Ce qu'il faut, c'est un outil capable d'identifier instantanément la nature d'un déchet et d'indiquer la bonne conduite à tenir — sans que l'utilisateur ait besoin de chercher l'information lui-même.

---

## Solution

WasteScan est une application web qui permet à n'importe quel utilisateur de photographier ou de déposer l'image d'un déchet pour obtenir immédiatement :

- la catégorie du déchet identifiée par un modèle de deep learning ;
- le niveau de confiance de la prédiction accompagné d'une jauge visuelle ;
- les instructions de tri correspondantes (type de bac, point de collecte, conseils pratiques) ;
- la distribution complète des probabilités sur l'ensemble des 10 catégories reconnues.

L'application s'appuie sur un réseau de neurones convolutif — EfficientNet-B0 par défaut — entraîné sur 10 classes de déchets. Le modèle est chargé localement via PyTorch et l'interface est construite avec Streamlit, ce qui permet un déploiement simple sans infrastructure complexe.

---

## Catégories reconnues

| Categorie    | Destination                        |
|--------------|------------------------------------|
| Battery      | Point de collecte special          |
| Biological   | Bac marron / compost               |
| Cardboard    | Bac bleu                           |
| Clothes      | Bornes textiles                    |
| Glass        | Bac vert                           |
| Metal        | Bac jaune                          |
| Paper        | Bac bleu                           |
| Plastic      | Bac jaune                          |
| Shoes        | Bornes de collecte specialisees    |
| Trash        | Bac noir / gris                    |

---

## Architecture technique

L'application supporte plusieurs architectures de reseaux de neurones, configurables selon le modele entraine disponible :

- `efficientnet_b0` (par defaut)
- `resnet18`
- `mobilenet_v2`
- `densenet121`
- `shufflenet_v2`

Chaque architecture utilise une tete de classification personnalisee composee de couches lineaires, de batch normalization et de dropout, entrainee pour les 10 classes de dechets.

Les images sont redimensionnees a 224x224 pixels et normalisees selon les statistiques ImageNet avant d'etre passees au modele.

---

## Prerequis

- Python 3.9 ou superieur
- pip

---

## Installation

Cloner le depot :

```bash
git clone https://github.com/votre-utilisateur/wastescan.git
cd wastescan
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Contenu minimal du fichier `requirements.txt` :

```
streamlit
torch
torchvision
Pillow
numpy
plotly
```

---

## Utilisation

### 1. Placer le fichier de modele

Le fichier de poids entraine doit etre present a la racine du projet sous le nom :

```
best_model_optimized.pth
```

Ce fichier correspond au modele EfficientNet-B0 entraine sur les 10 classes de dechets. Si vous utilisez une architecture differente, modifiez les parametres de configuration dans `app.py` :

```python
model_path   = "best_model_optimized.pth"
model_name   = "efficientnet_b0"   # resnet18 | mobilenet_v2 | densenet121 | shufflenet_v2
hidden_units = 192
dropout      = 0.456
```

### 2. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur a l'adresse `http://localhost:8501`.

### 3. Classifier un dechet

1. Deposer une image au format JPG, JPEG ou PNG dans la zone d'upload.
2. Cliquer sur le bouton **Analyser l'image**.
3. Lire les resultats : categorie detectee, niveau de confiance, instructions de tri, et distribution des probabilites sur l'ensemble des classes.

---

## Structure du projet

```
wastescan/
    app.py                      # Application principale Streamlit
    best_model_optimized.pth    # Poids du modele entraine (a fournir)
    requirements.txt            # Dependances Python
    README.md
```

---

## Metriques affichees

Apres chaque inference, l'application affiche trois indicateurs :

- **Confiance** : probabilite maximale assignee a la classe predite, exprimee en pourcentage. Une valeur superieure a 75 % indique une prediction fiable.
- **Inference** : temps de traitement en millisecondes, mesure localement.
- **Entropie** : mesure de l'incertitude de la distribution de probabilites. Une entropie elevee signale que le modele hesite entre plusieurs categories.

---

## Personnalisation

Il est possible d'adapter les consignes de tri affichees en modifiant le dictionnaire `RECYCLING_INFO` dans `app.py`. Chaque entree accepte les champs suivants : `recyclable` (booleen), `bin` (destination), et `tips` (conseil pratique).

---

## Licence

Ce projet est distribue sous licence MIT.
