# Example: variations of variations (4) {#arcanetdoc_arct_variations4}

[TOC]

File `example.arct`:

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
        "name": "Inter-boundingbox algorithm",
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
        "name": "Intra-boundingbox algorithm",
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
        "name": "Enabling the VtkHdf output"
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
          "name": "Sequential"
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
          "name": "Hybrid",
          "depend_b": ["nb_hybrid"]
        }
      }
    }
  }
}
```

Let's complicate things.

We added several variations in order to test several types of parallelism. We also put the
`enable_post_processing` dependency in the `general` block to simplify the reading a bit.

\remark All the configuration blocks can have dependencies. All the configuration blocks can be
dependencies, **except the `cases` blocks and the `general` block**.

First, let's look at the block lists for algo 1:

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

As well as the corresponding complete names (in the same order):

```txt
algo1:test=sequential

algo1:test=mpi:nb_proc=2
algo1:test=mpi:nb_proc=4

algo1:test=thread:nb_thread=2
algo1:test=thread:nb_thread=4

algo1:test=hybrid:nb_hybrid=4
algo1:test=hybrid:nb_hybrid=16
```

The easiest way to read this ARCT file is to start from one of the `cases` (or from the `general`) and to
look at the dependencies one by one (DFS).

\remark For a potential **ARCT v2** format, the possibility of adding variants in configuration blocks is
being studied. The problem is the addition of an extra layer of complexity:
```jsonc
"algo1": {
  "_": {
    "name": "Inter-boundingbox algorithm",
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
And one of the complete names: `algo1=mpi:nb_proc=2`.
<br>
Adding variants in configuration blocks of variants of variations will really complicate things, so this
part was removed from **ARCT v1**.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations3
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake
</span>
</div>
