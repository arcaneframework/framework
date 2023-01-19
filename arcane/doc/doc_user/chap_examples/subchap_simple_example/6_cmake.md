# Fichier CMakeLists.txt {#arcanedoc_examples_simple_example_cmake}

[TOC]

Le `CMakeLists.txt` est le dernier fichier que l'on va étudier.
Pour expliquer toutes les possibilités offertes par CMake,
il faudrait un tutoriel dédié, donc ici, on va juste faire un résumé
pour bien commencer.

CMake permet (entre autres choses) de générer un makefile utilisable par `Make`. 
`Make` est un outil permettant d'automatiser la compilation de projet en C/C++.

Voici un exemple de Makefile écrit à la main :
```makefile
# https://github.com/AlexlHer/CMolecule
MAKEDIR = build

LEX_INPUT = analyse.l
YACC_INPUT = analyse.y
OTHER_INPUT = Systeme.cpp Systeme.hpp\
	Gestion.hpp\
	main.cpp

LEX_OUTPUT = $(MAKEDIR)/lex.yy.cpp
YACC_OUTPUT = $(MAKEDIR)/y.tab.cpp
OTHER_OUTPUT = $(MAKEDIR)/Systeme.cpp $(MAKEDIR)/Systeme.hpp\
	$(MAKEDIR)/Gestion.hpp\
	$(MAKEDIR)/main.cpp

GCC_OUTPUT = projet_ter

############################

$(GCC_OUTPUT) : $(LEX_OUTPUT) $(YACC_OUTPUT) $(OTHER_OUTPUT)
	g++ $(MAKEDIR)/* -ll -ly -fopenmp -O2 -o $(GCC_OUTPUT)

$(LEX_OUTPUT) : $(LEX_INPUT)
	mkdir -p $(MAKEDIR)
	lex -o $(LEX_OUTPUT) $(LEX_INPUT)

$(YACC_OUTPUT) : $(YACC_INPUT)
	mkdir -p $(MAKEDIR)
	bison -d $(YACC_INPUT) -o $(YACC_OUTPUT)

$(OTHER_OUTPUT) : $(OTHER_INPUT)
	mkdir -p $(MAKEDIR)
	cp $(OTHER_INPUT) $(MAKEDIR)
```
On peut définir des variables (par exemple `LEX_INPUT = analyse.l`)
puis définir des travaux à faire.

Par exemple :
```makefile
$(YACC_OUTPUT) : $(YACC_INPUT)
	mkdir -p $(MAKEDIR)
	bison -d $(YACC_INPUT) -o $(YACC_OUTPUT)
```
On a un travail qui va générer un fichier appelé `$(YACC_OUTPUT)`.
Ce travail dépend du fichier `$(YACC_INPUT)`. Si ce fichier
a été modifié en deux lancement de make, alors le travail
sera lancé.
Le travail à effectuer est constitué de deux lignes de commandes
(juste en dessous : mkdir et bison).

Tout ceci représente un graphe de dépendance avec comme racine
le premier travail :
```makefile
$(GCC_OUTPUT) : $(LEX_OUTPUT) $(YACC_OUTPUT) $(OTHER_OUTPUT)
	g++ $(MAKEDIR)/* -ll -ly -fopenmp -O2 -o $(GCC_OUTPUT)
```
Ce travail génère le fichier `$(GCC_OUTPUT)`.
Ce travail dépend des fichiers `$(LEX_OUTPUT)`, `$(YACC_OUTPUT)` et `$(OTHER_OUTPUT)`.

Pour résumer, on peut représenter ce makefile comme ceci :

\image html MF_schema.svg

Sur un projet de quelques fichiers, c'est faisable d'écrire le `makefile`
à la main mais pour un projet comme %Arcane, il est nécessaire
d'utiliser un outil tier comme CMake.

CMake va, lui, utiliser des `CMakeLists.txt` pour générer
des `makefiles`. `CMakeLists.txt` contient les informations
nécessaires pour construire ce `makefile`.

## CMakeLists.txt {#arcanedoc_examples_simple_example_cmake_cmakeliststxt}

Voici le CMakeLists.txt fournie par `arcane_template` :
```cmake
cmake_minimum_required(VERSION 3.16)
project(HelloWorld LANGUAGES CXX)

find_package(Arcane REQUIRED)

add_executable(HelloWorld SayHelloModule.cc main.cc SayHello_axl.h)

arcane_generate_axl(SayHello)
arcane_add_arcane_libraries_to_target(HelloWorld)
target_include_directories(HelloWorld PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
configure_file(HelloWorld.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
```
Commençons :
```cmake
cmake_minimum_required(VERSION 3.16)
```
Cette première ligne demande la présence de la version 3.16 ou plus de CMake.
Ça permet de s'assurer que CMake pourra reconnaitre toutes les commandes
qu'on lui donne.

____

```cmake
project(HelloWorld LANGUAGES CXX)
```
On donne le nom du projet et le langage dans lequel
il est écrit (`CXX` = `C++`).

____

```cmake
find_package(Arcane REQUIRED)
```
Notre projet a besoin de %Arcane d'installé (voir la prochaine section
pour dire à CMake où est installé %Arcane).

____

```cmake
add_executable(HelloWorld SayHelloModule.cc main.cc SayHello_axl.h)
```
On donne aussi les différents fichiers qui composeront notre exécutable.
\note
Pas besoin de mettre `SayHelloModule.hh` vu qu'il est importé par `SayHelloModule.cc`.

____

```cmake
arcane_generate_axl(SayHello)
```
On demande à CMake de générer le fichier `SayHello_axl.h`.
On donne la position du fichier `SayHello.axl` en argument (sans l'extension `.axl`).
Ici, `SayHello.axl` est à la racine de notre projet donc on doit juste mettre `SayHello`.

____

```cmake
arcane_add_arcane_libraries_to_target(HelloWorld)
```
On ajoute les librairies %Arcane pour notre projet.

____

```cmake
target_include_directories(HelloWorld PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
```
On inclut tous les fichiers sources.

____

```cmake
configure_file(HelloWorld.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
```
On copie le `.config` dans le dossier de build.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_main
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_build
</span>
</div>
