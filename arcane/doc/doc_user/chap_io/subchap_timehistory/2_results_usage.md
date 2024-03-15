# Exploitation des résultats {#arcanedoc_io_timehistory_results_usage}

[TOC]

Voici un exemple des fichiers générés par l'exemple de la partie précédente :

```log
.
└── output
    └── courbes
        ├── curves.acv
        ├── gnuplot
        │   ├── avg_pressure
        │   ├── CpuTime
        │   ├── ElapsedTime
        │   ├── GlobalCpuTime
        │   ├── GlobalElapsedTime
        │   ├── GlobalTime
        │   ├── Mesh0_avg_pressure
        │   ├── Mesh0_SD0_avg_pressure
        │   ├── Mesh0_SD1_avg_pressure
        │   ├── Mesh0_SD2_avg_pressure
        │   ├── Mesh0_SD3_avg_pressure
        │   ├── Mesh1_avg_pressure
        │   ├── Mesh1_SD0_avg_pressure
        │   ├── Mesh1_SD1_avg_pressure
        │   ├── Mesh1_SD2_avg_pressure
        │   ├── Mesh1_SD3_avg_pressure
        │   ├── SD0_avg_pressure
        │   ├── SD1_avg_pressure
        │   ├── SD2_avg_pressure
        │   ├── SD3_avg_pressure
        │   └── TotalMemory
        ├── time_history.json
        └── time_history.xml
```

Nous avons deux types de fichiers ici : le "sommaire" et les fichiers
courbes générés par des écrivains.

Le fichier `time_history.json` (et `time_history.xml`, conservé pour des raisons de compatibilité)
est un "sommaire" au format JSON.
Il contient toutes les informations sur les historiques de valeurs comme le nom de l'historique,
le support, le sous-domaine source et un nom unique.
Il est généré quelque soit les écrivains activés.

Ensuite, deux écrivains sont actifs par défaut (ils implémentent l'interface Arcane::ITimeHistoryCurveWriter2) :

- ArcaneCurveWriter
- GnuplotTimeHistoryCurveWriter2

Leurs documentations sont disponibles dans la documentation développeur %Arcane.

## GnuplotTimeHistoryCurveWriter2

Cet écrivain génère tous les fichiers contenus dans le dossier `gnuplot`.
Chaque fichier est de la forme :

```log
1.7385346701095061E-01 6.3790000000000001E-03
3.4770693402190123E-01 6.5709999999999996E-03
5.2156040103285184E-01 6.1009999999999997E-03
```

Chaque ligne désigne une itération.
Deux colonnes : la première colonne contient le temps de la simulation,
la seconde colonne désigne la valeur de l'historique.

Pour la lecture de ces fichiers, l'utilisation du "sommaire" peut être utile puisque ces fichiers
ne contiennent aucune information supplémentaire.

## ArcaneCurveWriter

Cet écrivain génère le fichier `curves.acv`.
Ce fichier est un format %Arcane et contient toutes les informations de toutes les courbes.
Le fichier "sommaire" n'est donc pas forcément utile pour ce format.

Pour exploiter ce fichier, nous pouvons utiliser le programme `arcane_curves`, trouvable dans
`install_dir/bin/arcane_curves`.

\todo A suivre

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_io_timehistory_howto
</span>
<span class="next_section_button">
\ref arcanedoc_io_timehistory_acv
</span>
</div>
