# Example: base cases {#arcanetdoc_arct_cases}

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
        "name": "Inter-boundingbox algorithm"
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "false"
        }
      }
    },
    "algo2": {
      "_": {
        "name": "Intra-boundingbox algorithm"
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "true"
        }
      }
    }
  }
}
```

We stay on a simple example, but adding two base cases: `algo1` and `algo2`.

We leave the `dataset` in `general`: this ARCT is dedicated to `config_2D_diagonal_circle`. We can simply
choose between two different algorithms.

We could very well repeat the `dataset` in the two `cases` blocks, it comes down to the same here. All the
configuration blocks can contain the same elements.

To run algo 1, we can do as follows:

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1


{ [general] } -> { [algo1] }
```

If we do not add `-A,Case=algo1`, only the `general` block will be taken into account.

Here are the blocks received by Arcane:

```shell
./PotatoesSample1 -A,ParamFile=example.arct


{ [general] }
```

gives:

```txt
// Block 1
"arcane": {
  "dataset": "config_2D_diagonal_circle.arc"
}
```

----

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1


{ [general] } -> { [algo1] }
```

gives:

```txt
// Block 1
  "arcane": {
    "dataset": "config_2D_diagonal_circle.arc"
  }




// Block 2
  "_": {
    "name": "Inter-boundingbox algorithm"
  },
  "arcane": {
    "options": {
      "//nonreg/contact-pairs[]/use-contact-intra": "false"
    }
  }
```

----

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2


{ [general] } -> { [algo2] }
```

gives:

```txt
// Block 1
  "arcane": {
    "dataset": "config_2D_diagonal_circle.arc"
  }


// Block 2
  "_": {
    "name": "Intra-boundingbox algorithm"
  },
  "arcane": {
    "options": {
      "//nonreg/contact-pairs[]/use-contact-intra": "true"
    }
  }
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_general
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_commons
</span>
</div>
