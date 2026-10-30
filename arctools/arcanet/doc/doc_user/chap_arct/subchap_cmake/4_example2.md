# Exemple : cas de base {#arcanetdoc_arct_cmake_example2}

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
        "name": "Algorithme intra-boundingbox"
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
arct_config_create_or_edit_end()

arct_to_ctest()
```

Tests générées :

```shell
ctest -N

algo1
algo2
```

Reprenons l'exemple ARCT \ref arcanetdoc_arct_cases.

Deux cas sans variations donc deux tests générés.

On peut remarquer que seul le premier `arct_config_create_or_edit_XXX()` possède l'option `OVERWRITE_ARCT`. Seul le
premier appel doit écraser le fichier existant, s'il existe. Ensuite, on complète ce nouveau fichier.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example1
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example3
</span>
</div>
