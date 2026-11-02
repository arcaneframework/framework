# Simple example {#arcanetdoc_arct_general}

[TOC]

File `example.arct`:

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  },
  
  "general": {
    "_": {
      "name": "Basic run"
    },
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc",
      "options": {
        "MaxIteration": "10"
      }
    }
  }
}
```

This example is very simple. It gives two pieces of information to Arcane: the dataset to use and the
number of iterations to carry out.

Since Arcane supports this type of file, it is enough to pass it as an argument for it to be read:

```shell
./PotatoesSample1 -A,ParamFile=example.arct


{ [general] }
```

There is no base case in the file, so no need to specify one.

Note that a file like this one also works and is perfectly valid (but does strictly nothing):

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  }
}
```

\note The `versions` block is added automatically.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_basis
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cases
</span>
</div>
