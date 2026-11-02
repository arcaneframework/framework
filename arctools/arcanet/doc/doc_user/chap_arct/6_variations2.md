# Example: unresolved variations (2) {#arcanetdoc_arct_variations2}

[TOC]

File `example.arct`:

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
        "name": "Inter-boundingbox algorithm",
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
        "name": "Intra-boundingbox algorithm",
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
    }
  }
}
```

This example shows the whole point of variations.

The only change made here compared to the previous example is the removal of the `=4` in the `depend_b` of
algo 1: `"depend_b": ["nb_thread"]`.

We thus have an unresolved variation: we do not know which variant to choose by reading only the ARCT
file.

If we take a program supporting multi-case, we will have three lists of blocks for algo 1:

```shell
./PasArcane --arct=example.arct --case=algo1


{ [general] } -> { nb_thread=1 -> [algo1] }
{ [general] } -> { nb_thread=2 -> [algo1] }
{ [general] } -> { nb_thread=4 -> [algo1] }
```

For programs supporting only the single-case mode (like Arcane and ArcCher), it is necessary to resolve
the unresolved variations with the use case name.

If we want to run algo 1 with 4 threads, we add `nb_thread=4` to the case name:

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1:nb_thread=4


{ [general] } -> { nb_thread=4 -> [algo1] }
```

The `=` symbol is the only one making it possible to resolve a variation.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations1
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations3
</span>
</div>
