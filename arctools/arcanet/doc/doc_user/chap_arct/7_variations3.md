# Exemple : variations limitées (3) {#arcanetdoc_arct_variations3}

[TOC]

Fichier `example.arct` :

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  },

  "general": {
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc"
    }
  },

  "cases": {
    "algo1": {
      "_": {
        "name": "Algorithme inter-boundingbox",
        "depend_b": ["nb_thread!1"]
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "false"
        }
      }
    },
    "algo2": {
      "_": {
        "name": "Algorithme intra-boundingbox",
        "depend_b": ["enable_post_processing", "nb_thread+1"]
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "true"
        }
      }
    }
  },

  "commons": {
    "enable_post_processing": {
      "_": {
        "name": "Activation de la sortie VtkHdf"
      },
      "arcane": {
        "options": {
          "//nonreg/post-processor-write": "true",
          "//nonreg/mesh-contact-param[]/div-nb-cell": "1"
        }
      }
    }
  },

  "variations": {
    "nb_thread": {
      "1": {
        "_": {
          "name": "1 thread"
        }
      },
      "2": {
        "_": {
          "name": "2 threads"
        },
        "arcane": {
          "options": {
            "S": "2"
          }
        }
      },
      "4": {
        "_": {
          "name": "4 threads"
        },
        "arcane": {
          "options": {
            "S": "4"
          }
        }
      }
    }
  }
}
```

Il est possible qu'un des variants d'une variation ne soit pas compatible avec un bloc.

On peut alors limiter les variants disponibles.

Nous avons ajouté deux limiteurs dans l'exemple : `!` dans l'algo 1 et `+` dans l'algo 2.

Le limiteur `!` permet d'exclure un ou plusieurs variants. Dans l'algo 1, on a `nb_thread!1`, il sera donc impossible de
choisir le variant `nb_thread=1` dans le nom du cas.

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1:nb_thread=1
```
```txt
Erreur
```

Pour un programme multi-cas, il obtiendrait deux listes de blocs :

```shell
./PasArcane --arct=example.arct --case=algo1
```
```txt
{ [general] } -> { nb_thread=2 -> [algo1] }
{ [general] } -> { nb_thread=4 -> [algo1] }
```

Il est possible de mettre plusieurs inclusions ou exclusions à la suite, par exemple : `nb_thread!1!2`. En revanche, il
est impossible de mélanger les deux limiteurs pour une même variation (ça n'aurait pas de sens), par exemple :
`nb_thread+1+2!2`.

## Résolution de variations

Le seul moyen de résoudre une variation est d'utiliser le symbole de résolution `=`. C'est une limitation volontaire qui
force la simplification des fichiers ARCT et des noms de cas d'utilisation.

Si l'on reprend l'exemple au-dessus, si l'on souhaite utiliser uniquement `nb_thread=1`, sans cette règle, on pourrait
résoudre la variation de trois manières :

1. `nb_thread=1`
2. `nb_thread+1`
3. `nb_thread!2!4`

Le *3.* est moins facilement lisible que le *1.*. Le `+` du *2.* est censé être utilisé avec plusieurs inclusions.

\note `nb_thread=1` et `nb_thread+1` sont, sur le papier, identique. Mais la philosophie derrière est différente.
`nb_thread=1` ne donne pas le choix à l'utilisateur, c'est considéré conne une dépendance  `commons` classique.
`nb_thread+1` doit donner le choix à l'utilisateur (même si c'est un seul choix).

### Résolution de variations pour un programme mono-cas

Un programme mono-cas nécessite que toutes les variations soit résolues. Alors les symboles `!` et `+` ne peuvent pas
être utilisés dans un nom de cas d'utilisation :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1:nb_thread+2
```
```txt
Erreur
```

Pour l'algo 2, même si l'on a `nb_thread+1`, il est nécessaire de résoudre la variation dans la ligne de commande :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2:nb_thread=1
```
```txt
{ [general] } -> { enable_post_processing -> nb_thread=1 -> [algo2] }
```

Sinon :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2
```
```txt
Erreur
```

Si l'on ne veut pas que l'utilisateur ait à résoudre la variation, on doit mettre `nb_thread=1` dans le ARCT au lieu de
`nb_thread+1`.

\note Mettre `nb_thread+1` dans le fichier ARCT est autorisé et peut même être utile si l'on souhaite forcer
l'utilisateur à faire une résolution dans la ligne de commande. Le cas d'usage est présenté dans la partie cmake
`arct -> ctest`. Dans cette partie, le nom d'un test est le nom du cas d'utilisation exécuté par CTest. Si l'on a un cas
uniquement séquentiel, il peut être utile d'afficher `test1:nb_proc=1` au lieu de juste afficher `test1`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations2
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations4
</span>
</div>
