# Comment ça fonctionne {#arcanedoc_io_timehistory_howto}

[TOC]

L'utilisation du TimeHistory est extrêmement simple.

À chaque itération, une valeur sera enregistrée pour chaque historique de valeurs.
S'il n'y a pas d'enregistrement de valeur explicite lors d'une itération, un 0 sera enregistré.
S'il y a deux valeurs enregistrées pendant une même itération pour un historique de valeurs,
c'est la dernière valeur qui sera réellement enregistrée.

Historiquement, les historiques de valeurs étaient gérés uniquement par le processus 0.
Aujourd'hui, ce n'est plus le cas. Chaque processus peut avoir son historique, avec une même clef.
De plus, on peut lier un historique à un maillage. Ainsi, on peut avoir un historique par maillage, toujours
avec la même clef.

\warning Pour activer l'enregistrement en multi-processus, il est nécessaire de définir la variable
d'environnement `ARCANE_ENABLE_NON_IO_MASTER_CURVES=1`.

## GlobalTimeHistoryAdder

La première structure permettant de gérer des historiques de valeurs est le `GlobalTimeHistoryAdder`.
Il permet d'ajouter des valeurs à un historique.
Le "Global" signifie que les variables internes utilisées pour gérer cet historique sont globales, liées
aux sous-domaines.

Admettons que l'on ai un maillage partagé dans quatre sous-domaines (`SD0`, `SD1`, `SD2`, `SD3`).
Chaque maille possède une pression. On veut avoir, pour chaque itération, la pression moyenne de chaque
sous-domaine. Et en plus, on veut avoir la pression moyenne de tout le domaines.

Utilisons comme clef : `avg_pressure` :
\image html avg_pressure.svg

Chaque sous-domaine possède une pression moyenne et il y a une pression moyenne globale. L'image
présente une seule itération : l'itération 0.

Pour obtenir un historique comme celui-ci, on peut faire comme ceci :
\snippet{c++} TimeHistoryAdderTestModule.cc snippet_timehistory_example1

\remark En interne, `GlobalTimeHistoryAdder` utilise la partie interne du `ITimeHistoryMng` passé en paramètre.
L'objet `GlobalTimeHistoryAdder` peut donc être détruit sans problème.

\note Pour utiliser `GlobalTimeHistoryAdder`, ne pas oublier d'importer les headers nécessaires :
```cpp
#include <arcane/core/ITimeHistoryMng.h>
#include <arcane/core/GlobalTimeHistoryAdder.h>
```

Ce bout de code, s'il est appelé à toutes les itérations, permet d'obtenir les
moyennes à chaque itération.

## MeshTimeHistoryAdder

La seconde structure permettant de gérer des historiques de valeurs est le `MeshTimeHistoryAdder`.
Comme la première structure, il permet d'ajouter des valeurs à un historique.
Le "Mesh" signifie que les variables internes utilisées pour gérer cet historique sont liées au
maillage souhaité. Donc, chaque maillage peut avoir une variable différente de même nom. 

Reprenons l'exemple au-dessus mais avec deux maillages.
Ces deux maillages sont réparties sur quatre sous-domaines. On veut avoir, pour chaque
sous-domaine, la moyenne des pressions des mailles de chaque maillage.
Mais on veut toujours, comme au-dessus, la pression moyenne de chaque
sous-domaine.

Utilisons la même clef : `avg_pressure` :
\image html avg_pressure2.svg

On peut voir qu'en plus des `avg_pressure` de chaque sous-domaine et du globale, il y a des `avg_pressure`
pour les deux maillages.

Voici un exemple de code pour réaliser ce calcul :
\snippet{c++} TimeHistoryAdderTestModule.cc snippet_timehistory_example2

La différence ici, c'est qu'on itère sur les maillages. Pour créer le `MeshTimeHistoryAdder`,
on doit donner, en plus d'un `ITimeHistoryMng*`, un handle de maillage. Ça permet de lier
l'historique au maillage.

\remark En interne, `MeshTimeHistoryAdder` utilise la partie interne du `ITimeHistoryMng` passé en paramètre.
L'objet `MeshTimeHistoryAdder` peut donc être détruit sans problème.

\note Pour utiliser `MeshTimeHistoryAdder`, ne pas oublier d'importer les headers nécessaires :
```cpp
#include <arcane/core/ITimeHistoryMng.h>
#include <arcane/core/MeshTimeHistoryAdder.h>
```

\note Le TimeHistoryMng gère les checkpoints, l'utilisateur n'a donc pas à s'en occuper.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_io_timehistory
</span>
<span class="next_section_button">
\ref arcanedoc_io_timehistory_results_usage
</span>
</div>
