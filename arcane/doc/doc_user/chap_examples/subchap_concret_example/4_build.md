# Compilation et lancement {#arcanedoc_examples_concret_example_build}

[TOC]

Cette dernière partie sera assez courte étant donné qu'il n'y a pas
de grandes différences avec le CMakeLists.txt du HelloWorld du chapitre
précedent (\ref arcanedoc_examples_simple_example_cmake).
De plus, les commandes de compilations ne changent pas grandement non
plus (\ref arcanedoc_examples_simple_example_build).

## CMakeLists.txt {#arcanedoc_examples_concret_example_build_cmakeliststxt}

Voici le CMakeLists.txt de Quicksilver :
```cmake
cmake_minimum_required(VERSION 3.16)
project(Quicksilver LANGUAGES CXX)

set(BUILD_SHARED_LIBS TRUE)

find_package(Arcane REQUIRED)

add_executable(Quicksilver
  main.cc
  qs/QSModule.cc qs/QS_axl.h
  sampling_mc/SamplingMCModule.cc sampling_mc/SamplingMC_axl.h
  tracking_mc/TrackingMCModule.cc tracking_mc/TrackingMC_axl.h
  rng/RNGService.cc rng/RNG_axl.h
  tracking_mc/NuclearData.cc)

arcane_generate_axl(qs/QS)
arcane_generate_axl(sampling_mc/SamplingMC)
arcane_generate_axl(tracking_mc/TrackingMC)
arcane_generate_axl(rng/RNG)


arcane_add_arcane_libraries_to_target(Quicksilver)
target_compile_options(Quicksilver PUBLIC -Wpedantic)
target_include_directories(Quicksilver PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(Quicksilver.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
```

Concentrons-nous sur les nouvelles lignes.

```cmake
set(BUILD_SHARED_LIBS TRUE)
```
Cette ligne demande à cmake de générer des librairies dynamiques
au lieu de librairies statiques.

____

```cmake
target_compile_options(Quicksilver PUBLIC -Wpedantic)
```
Cette ligne permet d'ajouter une option de compilation. L'option
ajoutée ici demande l'ajout de plus de warnings lors de la compilation. 


## Compilation {#arcanedoc_examples_concret_example_build_commands}

Pour la compilation, on peut réutiliser les commandes présentées dans ce
chapitre : \ref arcanedoc_examples_simple_example_build, en n'oubliant
pas de modifier `HelloWorld` en `quicksilver`.

La liste des exemples `.arc` est disponible dans le `readme.md` de
`Quicksilver`.

____

Si un problème est présent dans ce sous-chapitre, vous pouvez ouvrir une
issue dans le GitHub d'%Arcane.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example_rng
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>