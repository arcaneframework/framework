# Example: restricted variations (3) {#arcanetdoc_arct_cmake_example6}

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
        "name": "Inter-boundingbox algorithm",
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
        "name": "Intra-boundingbox algorithm",
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

File `tests.cmake`:

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_arcane_dataset(PATH "config_2D_diagonal_circle.arc")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo1")
arct_config_define_name(NAME "Inter-boundingbox algorithm")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "false")
arct_config_variation(ADD "nb_thread" EXCLUDE "1")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo2")
arct_config_define_name(NAME "Intra-boundingbox algorithm")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "true")
arct_config_dependency(ADD "enable_post_processing")
arct_config_variation(ADD "nb_thread" INCLUDE "1")
arct_config_create_or_edit_end()

arct_config_create_or_edit_common(NAME_COMMON "enable_post_processing")
arct_config_define_name(NAME "Enabling the VtkHdf output")
arct_arcane_option(ADD "//nonreg/post-processor-write" VALUE "true")
arct_arcane_option(ADD "//nonreg/mesh-contact-param[]/div-nb-cell" VALUE "1")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_thread" NAME_VARIANT "1")
arct_config_define_name(NAME "1 thread")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_thread" NAME_VARIANT "2")
arct_config_define_name(NAME "2 threads")
arct_arcane_option(ADD "S" VALUE "2")
arct_config_create_or_edit_end()

arct_config_create_or_edit_variation(NAME_VARIATION "nb_thread" NAME_VARIANT "4")
arct_config_define_name(NAME "4 threads")
arct_arcane_option(ADD "S" VALUE "4")
arct_config_create_or_edit_end()

arct_to_ctest()
```

Generated tests:

```shell
ctest -N

algo1:nb_thread=2
algo1:nb_thread=4
algo2:nb_thread=1
```

Let's revisit the ARCT example \ref arcanetdoc_arct_variations3.

We have two limiters: `!` and `+`.

`!` corresponds to the `EXCLUDE` keyword and `+` corresponds to the `INCLUDE` keyword.

We can also see the difference between `nb_thread=1` and `nb_thread+1`: for `nb_thread+1`, it is necessary
to resolve the variation, even if there is only one included variant (so put `nb_thread=1` in the case
name).

Here, the reason is simply to be able to see `nb_thread=1` explicitly in the test name.
Another advantage is to be able to easily add inclusions by adding a line:
`arct_config_variation(ADD "nb_thread" INCLUDE "2")` for example.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example5
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example7
</span>
</div>
