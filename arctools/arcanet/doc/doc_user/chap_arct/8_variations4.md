# Exemple : variations de variations (4) {#arcanetdoc_arct_variations4}

[TOC]

Fichier `example.arct` :

```json
{
  "versions": {
    "_": 1,
    "arcane": 1,
    "arccher": 1
  },

  "general": {
    "_": {
      "depend_b": ["enable_post_processing"]
    },
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc"
    }
  },

  "cases": {
    "algo1": {
      "_": {
        "name": "Algorithme inter-boundingbox",
        "depend_b": ["test"]
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
        "depend_b": ["test=sequential"]
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
    },
    "nb_proc": {
      "1": {
        "_": {
          "name": "1 proc"
        }
      },
      "2": {
        "_": {
          "name": "2 procs"
        },
        "arccher": {
          "mpi": 2
        }
      },
      "4": {
        "_": {
          "name": "4 procs"
        },
        "arccher": {
          "mpi": 4
        }
      }
    },
    "nb_hybrid": {
      "4": {
        "_": {
          "name": "2 procs + 2 threads",
          "depend_b": ["nb_proc=2", "nb_thread=2"]
        }
      },
      "16": {
        "_": {
          "name": "4 procs + 4 threads",
          "depend_b": ["nb_proc=4", "nb_thread=4"]
        }
      }
    },
    "test": {
      "sequential": {
        "_": {
          "name": "Séquentiel"
        }
      },
      "mpi": {
        "_": {
          "name": "MPI",
          "depend_b": ["nb_proc!1"]
        }
      },
      "thread": {
        "_": {
          "name": "MT",
          "depend_b": ["nb_thread!1"]
        }
      },
      "hybrid": {
        "_": {
          "name": "Hybride",
          "depend_b": ["nb_hybrid"]
        }
      }
    }
  }
}
```

Compliquons les choses.

On a ajouté plusieurs variations afin de tester plusieurs types de parallélisme. On a aussi mis la dépendance
`enable_post_processing` dans le bloc `general` pour simplifier un peu la lecture.

\remark Tous les blocs de configurations peuvent avoir des dépendances. Tous les blocs de configurations peuvent être
des dépendances, **sauf les blocs `cases` et le bloc `general`**.

D'abord, voyons les listes de blocs pour l'algo 1 :

```shell
./PasArcane --arct=example.arct --case=algo1



{ enable_post_processing -> [general] } -> { test=sequential -> [algo1] }

{ enable_post_processing -> [general] } -> { { nb_proc=2 -> [test=mpi] } -> [algo1] }
{ enable_post_processing -> [general] } -> { { nb_proc=4 -> [test=mpi] } -> [algo1] }

{ enable_post_processing -> [general] } -> { { nb_thread=2 -> [test=thread] } -> [algo1] }
{ enable_post_processing -> [general] } -> { { nb_thread=4 -> [test=thread] } -> [algo1] }

{ enable_post_processing -> [general] } -> { { { nb_proc=2 -> nb_thread=2 -> [nb_hybrid=4] } -> [test=hybrid] } -> [algo1] }
{ enable_post_processing -> [general] } -> { { { nb_proc=4 -> nb_thread=4 -> [nb_hybrid=16] } -> [test=hybrid] } -> [algo1] }
```

Ainsi que les noms complets correspondants (dans le même ordre) :

```txt
algo1:test=sequential

algo1:test=mpi:nb_proc=2
algo1:test=mpi:nb_proc=4

algo1:test=thread:nb_thread=2
algo1:test=thread:nb_thread=4

algo1:test=hybrid:nb_hybrid=4
algo1:test=hybrid:nb_hybrid=16
```

La manière la plus aisée de lire ce fichier ARCT est de partir d'un des `cases` (ou du `general`) et d'aller voir les
dépendances une par une (en DFS).

\remark Pour un potentiel format **ARCT v2**, la possibilité de pouvoir ajouter des variants dans les blocs de
configurations est étudiée. Le problème est l'ajout d'une couche de complexité supplémentaire :
```jsonc
"algo1": {
  "_": {
    "name": "Algorithme inter-boundingbox",
    "depend_b": ["enable_post_processing"],
    "variant_b": {
      "sequential": [],
      "mpi": ["nb_proc!1"],
      "thread": ["nb_thread!1"],
      "hybrid": ["nb_hybrid"]
    }
  }
}
```
Et l'un des noms complets : `algo1=mpi:nb_proc=2`.
<br>
Ajouter des variants dans des blocs de configurations de variants de variations va vraiment compliquer les choses, cette
partie a donc été retirée de **ARCT v1**.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations3
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake
</span>
</div>
