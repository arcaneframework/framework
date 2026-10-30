# Exemple : blocs commons {#arcanetdoc_arct_cmake_example3}

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
        "name": "Algorithme inter-boundingbox"
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
        "name": "Activation de la sortie VtkHdf"
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

Fichier `tests.cmake` :

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_arcane_dataset(PATH "config_2D_diagonal_circle.arc")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo1")
arct_config_define_name(NAME "Algorithme inter-boundingbox")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "false")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo2")
arct_config_define_name(NAME "Algorithme intra-boundingbox")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "true")
arct_config_dependency(ADD "enable_post_processing")
arct_config_create_or_edit_end()

arct_config_create_or_edit_common(NAME_COMMON "enable_post_processing")
arct_config_define_name(NAME "Activation de la sortie VtkHdf")
arct_arcane_option(ADD "//nonreg/post-processor-write" VALUE "true")
arct_arcane_option(ADD "//nonreg/mesh-contact-param[]/div-nb-cell" VALUE "1")
arct_config_create_or_edit_end()

arct_to_ctest()
```

Tests générées :

```shell
ctest -N

algo1
algo2
```

Reprenons l'exemple ARCT \ref arcanetdoc_arct_commons.

Toujours pas de variations donc deux tests.

CMake s'occupe du tableau de dépendance, on a juste à demander à CMake d'ajouter un élément dedans (ou de le créer s'il
n'existe pas).

On peut aussi voir que l'on peut ajouter une dépendance sans qu'elle soit créée avant. En effet, on ne vérifie pas la
validité des informations dans cette partie. La vérification sera effectuée par `arct_to_ctest()`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example2
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example4
</span>
</div>
