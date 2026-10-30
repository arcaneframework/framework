# Simple exemple {#arcanetdoc_arct_cmake_example1}

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
    "_": {
      "name": "Lancement basique"
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

Fichier `tests.cmake` :

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_config_define_name(NAME "Lancement basique")
arct_arcane_dataset(PATH "config_2D_diagonal_circle.arc")
arct_arcane_option(ADD "MaxIteration" VALUE "10")
arct_config_create_or_edit_end()

arct_to_ctest()
```

Tests générées :

```shell
ctest -N

#Aucun
```

Reprenons l'exemple ARCT \ref arcanetdoc_arct_general.

Nous avons quatre informations éditables dans ce fichier ARCT : une information de type `config` et deux informations de
type `arcane`.

Les fonctions débutantes par `arct_config_...` concernent le bloc réservé ARCT (`_`).

Les fonctions débutantes par `arct_arcane_...` concernent le bloc réservé Arcane (`arcane`).

Les fonctions débutantes par `arct_arccher_...` concernent le bloc réservé ArcCher (`arccher`).

Comme il n'y a pas de cas de base, aucun test n'est généré par la fonction `arct_to_ctest()`. Ce genre de fichier peut
servir pour être lu par Arcane directement.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_arct_to_ctest
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example2
</span>
</div>
