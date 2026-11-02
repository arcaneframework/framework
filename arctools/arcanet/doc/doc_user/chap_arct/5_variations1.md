# Example: simple variations (1) {#arcanetdoc_arct_variations1}

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

To introduce variations, we will start with a simple variation to apply to Arcane.

We have a variation named `nb_thread` which makes it possible to change the number of threads
(shared-memory subdomains).

Three variants are available: `1`, `2` and `4`. We rather say: `nb_thread=1`, `nb_thread=2` and
`nb_thread=4`.

The `nb_thread=1` variant does nothing. The other two modify the Arcane `-A,S` parameter.

In the `cases`, we can see **resolved variations**: `nb_thread=4` for `algo1` and `nb_thread=1` for
`algo2`. A resolved variation can be seen as a `commons`: just by reading the ARCT file, we know which
block to choose. So the launch is as simple as before:

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
