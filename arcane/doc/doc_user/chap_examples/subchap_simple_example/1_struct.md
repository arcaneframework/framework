# Structure générale {#arcanedoc_examples_simple_example_struct}

[TOC]

## HelloWorld {#arcanedoc_examples_simple_example_struct_helloworld}

Voici un schéma représentant la structure de notre Hello World :

\image html HW_schema.svg

Dans cette application, nous avons un module appelé `SayHello`
contenant trois fichiers :
- un header (.h),
- un fichier source (.cc),
- un fichier contenant les options du jeu de données (.axl).

Et hors du module, nous avons quatre fichiers :
- un fichier `main.cc` permettant de lancer notre application,
- un fichier `CMakeLists.txt` permettant de compiler notre application,
- un fichier `.config` permettant de configurer notre application,
- un fichier `.arc` contenant un jeu de données pour notre application.

Tous ces élements constituent notre application `HelloWorld`.

\note
Il est possible de générer un template d'application avec le programme `arcane-template`.
Pour générer un template de notre `HelloWorld` avec `arcane-template`, voici la commande :
```sh
./arcane_templates generate-application -code-name HelloWorld --module-name SayHello --output-directory ~/HelloWorld
```
Ce programme est trouvable dans le dossier `bin` du répertoire d'installation de %Arcane : `arcane_install/bin/`.


Dans la section suivante, nous allons voir le module `SayHello`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_module
</span>
</div>