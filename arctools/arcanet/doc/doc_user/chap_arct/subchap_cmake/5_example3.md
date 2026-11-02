# Example: commons blocks {#arcanetdoc_arct_cmake_example3}

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
arct_config_dependency(ADD "enable_post_processing")
arct_config_create_or_edit_end()

arct_config_create_or_edit_common(NAME_COMMON "enable_post_processing")
arct_config_define_name(NAME "Enabling the VtkHdf output")
arct_arcane_option(ADD "//nonreg/post-processor-write" VALUE "true")
arct_arcane_option(ADD "//nonreg/mesh-contact-param[]/div-nb-cell" VALUE "1")
arct_config_create_or_edit_end()

arct_to_ctest()
```

Generated tests:

```shell
ctest -N

algo1
algo2
```

Let's revisit the ARCT example \ref arcanetdoc_arct_commons.

Still no variations so two tests.

CMake takes care of the dependency array, we just have to ask CMake to add an element in it (or to create
it if it does not exist).

We can also see that a dependency can be added without it being created before. Indeed, the validity of
the information is not checked in this part. The check will be done by `arct_to_ctest()`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example2
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example4
</span>
</div>
