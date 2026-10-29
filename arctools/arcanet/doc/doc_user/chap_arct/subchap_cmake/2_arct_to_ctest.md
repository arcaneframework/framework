# Convertisseur ARCT -> CTest {#arcanetdoc_arct_cmake_arct_to_ctest}

[TOC]

Pour pouvoir lancer les différents cas d'un fichier ARCT, une fonction CMake a été ajoutée pour lire un fichier ARCT et
ajouter un test par cas d'utilisation.


--------------------------

### arct_to_ctest

```cmake
arct_to_ctest(ARCT_PATH <arct_path> ARCT_CONFIG_PATH <arct_config_path> CTEST_NAME_PREFIX <prefix> ARCCHER_PATH <arccher_path>)
```

Cette fonction permet de lire un fichier ARCT et de créer un test par cas d'utilisation.

Les noms des tests correspondent aux noms complets de chaque cas d'utilisation.

Quatre arguments `oneValueArgs` sont disponibles :

- `ARCT_PATH <arct_path>` (facultatif) : permet de donner l'emplacement du fichier ARCT à lire. La valeur par défaut est
  `${CMAKE_BINARY_DIR}/testlist.arct`.
- `ARCT_CONFIG_PATH <arct_config_path>` (facultatif) : permet de donner l'emplacement d'un second fichier ARCT à lire.
  Seul le bloc `general` sera lu. Il sera lu en premier.
- `CTEST_NAME_PREFIX <prefix>` (facultatif) : permet de donner un préfixe à ajouter aux noms des tests CTest.
- `ARCCHER_PATH <arccher_path>` (facultatif) : permet de donner le chemin de l'exécutable ArcCher. Il sera défini par
  défaut par Arcane, si utilisation de ArcaNet de l'installation d'Arcane.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_arct_writer
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_arct_writer_example1
</span>
</div>
