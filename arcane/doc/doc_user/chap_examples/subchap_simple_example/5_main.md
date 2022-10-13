# Fichier main {#arcanedoc_examples_simple_example_main}

[TOC]

À présent, voyons le fichier `main.cc`. Ce fichier contient la fonction `main()` 
qui sera lancée à l'ouverture de HelloWorld. Voici à quoi elle ressemble :

## main.cc {#arcanedoc_examples_simple_example_main_maincc}
```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  auto& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("HelloWorld");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  return ArcaneLauncher::run();
}
```
La fonction `main()` sert à lancer %Arcane et notre application.
Cette fonction ne sera donc pratiquement jamais modifié (sauf pour faire évoluer
la version du code). Dans des cas plus avancés, on doit modifier `main()` pour, 
par exemple, changer l'allocateur mémoire qui sera utilisé dans %Arcane pour
notre application.
On peut aussi utiliser cette fonction pour lancer des calculs sans modules,
en utilisant les classes %Arcane.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_config
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_cmake
</span>
</div>