# Exemple : variations non résolues (2) {#arcanetdoc_arct_variations2}

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
        "depend_b": ["nb_thread"]
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
        "depend_b": ["enable_post_processing", "nb_thread=1"]
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

Cet exemple permet de montrer tout l'intérêt des variations.

La seule modification réalisée ici par rapport à l'exemple précedent est la suppression du `=4` au `depend_b` de l'algo
1 :
`"depend_b": ["nb_thread"]`.

Nous avons donc une variation non résolue : on ne sait pas quel variant choisir en lisant uniquement le fichier ARCT.

Si l'on prend un programme supportant les multi-cas, on aurait trois listes de blocs pour l'algo 1 :

```shell
./PasArcane --arct=example.arct --case=algo1


{ [general] } -> { nb_thread=1 -> [algo1] }
{ [general] } -> { nb_thread=2 -> [algo1] }
{ [general] } -> { nb_thread=4 -> [algo1] }
```

Pour les programmes ne supportant que le mode mono-cas (comme Arcane et ArcCher), il est nécessaire de résoudre les
variations non résolues avec le nom du cas d'utilisation.

Si l'on souhaite exécuter l'algo 1 avec 4 threads, on ajoute `nb_thread=4` au nom du cas :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1:nb_thread=4


{ [general] } -> { nb_thread=4 -> [algo1] }
```

Le symbole `=` est le seul permettant de résoudre une variation.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations1
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations3
</span>
</div>
