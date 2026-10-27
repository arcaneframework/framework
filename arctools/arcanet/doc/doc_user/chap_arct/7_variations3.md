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

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations2
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations4
</span>
</div>
