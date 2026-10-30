# Exemple : variations simples (1) {#arcanetdoc_arct_cmake_example4}

[TOC]

Fichier `testlist.arct` :

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
        "name": "Algorithme inter-boundingbox",
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
        "name": "Algorithme intra-boundingbox",
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
    }
  }
}
```

Fichier `tests.cmake` :

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_arcane_dataset(PATH "config_2D_diagonal_circle.arc")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo1")
arct_config_define_name(NAME "Algorithme inter-boundingbox")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "false")
arct_config_variation(ADD "nb_thread" EQUAL "4")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo2")
arct_config_define_name(NAME "Algorithme intra-boundingbox")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "true")
arct_config_dependency(ADD "enable_post_processing")
arct_config_variation(ADD "nb_thread" EQUAL "1")
arct_config_create_or_edit_end()

arct_config_create_or_edit_common(NAME_COMMON "enable_post_processing")
arct_config_define_name(NAME "Activation de la sortie VtkHdf")
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

Tests générées :

```shell
ctest -N

algo1
algo2
```

Reprenons l'exemple ARCT \ref arcanetdoc_arct_variations1.

On a une variation avec trois variants.

Dans les deux cas, on résout cette variation donc on a toujours deux tests.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example3
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example5
</span>
</div>
