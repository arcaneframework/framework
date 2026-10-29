# Exemple : variations simples (1) {#arcanetdoc_arct_variations1}

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
        "depend_b": ["nb_thread=4"]
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

Pour introduire les variations, on va commencer par une variation simple à appliquer à Arcane.

On a une variation s'appelant `nb_thread` qui permet de changer le nombre de threads (sous-domaine en mémoire partagée).

Trois variants sont disponibles : `1`, `2` et `4`. On dira plutôt : `nb_thread=1`, `nb_thread=2` et `nb_thread=4`.

Le variant `nb_thread=1` ne fait rien. Les deux autres modifient le paramètre `-A,S` d'Arcane.

Dans les `cases`, on peut voir des **variations résolues** : `nb_thread=4` pour `algo1` et `nb_thread=1` pour `algo2`.
Une variation résolue peut être vu comme un `commons` : rien qu'en lisant le fichier ARCT, on sait quel bloc choisir.
Donc le lancement est aussi simple qu'avant :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2


{ [general] } -> { enable_post_processing -> nb_thread=1 -> [algo2] }
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_commons
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations2
</span>
</div>
