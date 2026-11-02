# Example: commons blocks {#arcanetdoc_arct_commons}

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
        "name": "Intra-boundingbox algorithm",
        "depend_b": ["enable_post_processing"]
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
  }
}
```

Here we add a `commons` block. This block, in this example, makes it possible to enable the outputs of our
program.

We have two types of dependencies: `depend_b` (**before**) and `depend_a` (**after**). The only difference
is that the blocks of the **before** dependencies are added before our block, while the blocks of the
**after** dependencies are added after.

\note The dependency blocks are added in the same order as the one given in the `depend_b`/`depend_a`
arrays of the ARCT file.
<br>
If:
```jsonc
...
"algo2": {
...
"depend_b": ["aaa", "bbb"],
"depend_a": ["ccc", "ddd"],
...
```
then we will have:
```txt
... -> { aaa -> bbb -> [algo2] -> ccc -> ddd }
```

If we run `algo1`, the result will be the same as in the previous example. However, for algo 2:

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2


{ [general] } -> { enable_post_processing -> [algo2] }
```

There is a `depend_b` in `algo2` so the `enable_post_processing` block will be read before `algo2`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cases
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations1
</span>
</div>
