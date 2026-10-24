# TP 3 : Détection et Évaluation Quantitative de Contours
**Auteur :** Youssef BENABBOU  
**Cadre :** Traitement et Analyse d'Images — Aix-Marseille Université  
**Technologies :** C++20 / OpenCV 4.10 / Visual Studio

---

Projet de détection et d'évaluation quantitative de contours sous OpenCV C++ comparant quatre méthodes :
* **Sobel** (norme L1)
* **Laplace** (seuillage d'amplitude)
* **Laplace Zero-Crossing** (passage par zéro morphologique)
* **Canny** (détecteur optimal avec hystérésis)

Le banc d'essai est réalisé sur la base de données RUG (40 images et vérités terrain manuelles).

---

Video demonstrative et Explication du code : https://drive.google.com/file/d/1qYao8HlkdzS2Dtp2yqdE8HSLRIunZOzw/view?usp=sharing

## Utilisation

Le projet s'ouvre avec Visual Studio via `TP_contours.sln`.

L'exécutable compilé est disponible dans `x64/Release/TP_contours.exe`.

### 1. Mode interactif (interface graphique)
```powershell
.\x64\Release\TP_contours.exe
```

L'application affiche 4 cadrans synchronisés : image source, contours détectés, vérité terrain de référence et carte d'erreurs en couleurs (Vert = correct, Rouge = faux positif, Bleu = contours manqués).

**Contrôles :**
* `0` à `3` : changer de méthode (Sobel, Laplace Seuil, Laplace ZC, Canny)
* `N` / `P` : image suivante / précédente
* `V` : basculer entre évaluation stricte (1x1) et tolérance de voisinage (3x3)
* `B` : lancer le traitement par lot sur les 40 images
* `Q` ou `ESC` : quitter
* Curseurs pour ajuster les seuils, la taille de noyau et le flou gaussien

### 2. Mode batch (ligne de commande)
```powershell
.\x64\Release\TP_contours.exe --batch
```
Exécute automatiquement le benchmark complet sur les 40 images et sauvegarde les mesures dans `resultats_lot_cpp.csv`.

---

## Structure du dossier

* `TP_contours/` : code source C++ (`main.cpp`, `laplacianZC.h`) et projet Visual Studio
* `resultats_lot_cpp.csv` : mesures brutes issues du traitement par lot
* `rapport/` : rapport complet (`rapport.pdf`), code source LaTeX (`rapport.tex`) et figures (`IMAGE_LATEX/`)
* `video/` : script de la vidéo de présentation (`script_video_presentation.md`)
