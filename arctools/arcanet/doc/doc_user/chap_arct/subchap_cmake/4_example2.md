# Example: base cases {#arcanetdoc_arct_cmake_example2}

[TOC]

File `testlist.arct`:

```json
{
  "versions": {
    "_": 1,
    "arcane": 1,
    "arccher": 1
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

File `tests.cmake`:

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_arcane_dataset(PATH "config_2D_diagonal_circle.arc")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo1")
arct_config_define_name(NAME "Inter-boundingbox algorithm")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "false")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo2")
arct_config_define_name(NAME "Intra-boundingbox algorithm")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "true")
arct_config_create_or_edit_end()

arct_to_ctest()
```

Generated tests:

```shell
ctest -N

algo1
algo2
```

Let's revisit the ARCT example \ref arcanetdoc_arct_cases.

Two cases without variations so two generated tests.

We can notice that only the first `arct_config_create_or_edit_XXX()` has the `OVERWRITE_ARCT` option.
Only the first call must overwrite the existing file, if it exists. Then, we complete this new file.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example1
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example3
</span>
</div>
