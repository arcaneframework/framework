# Fichier CMakeLists.txt {#arcanedoc_examples_simple_example_cmake}

[TOC]

The `CMakeLists.txt` is the last file we will study.
To explain all the possibilities offered by CMake, a dedicated tutorial would be
needed, so here, we will just provide a summary to get started.

CMake allows (among other things) generating a Makefile usable by `Make`.
`Make` is a tool that allows automating the compilation of C/C++ projects.

Here is an example of a Makefile written by hand:
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
We can define variables (for example `LEX_INPUT = analyse.l`) and then define
tasks to be performed.

For example:
```makefile
$(YACC_OUTPUT) : $(YACC_INPUT)
	mkdir -p $(MAKEDIR)
	bison -d $(YACC_INPUT) -o $(YACC_OUTPUT)
```
We have a task that will generate a file called `$(YACC_OUTPUT)`.
This task depends on the file `$(YACC_INPUT)`. If this file has been modified in
two runs of make, then the task will be launched.
The task consists of two lines of commands (just below: mkdir and bison).

All of this represents a dependency graph with the first task as the root:
```makefile
$(GCC_OUTPUT) : $(LEX_OUTPUT) $(YACC_OUTPUT) $(OTHER_OUTPUT)
	g++ $(MAKEDIR)/* -ll -ly -fopenmp -O2 -o $(GCC_OUTPUT)
```
This task generates the file `$(GCC_OUTPUT)`.
This task depends on the files `$(LEX_OUTPUT)`, `$(YACC_OUTPUT)`, and
`$(OTHER_OUTPUT)`.

To summarize, we can represent this makefile as follows:

\image html MF_schema.svg

For a project with a few files, it is possible to write the `makefile` by hand,
but for a project like %Arcane, it is necessary to use a third-party tool like
CMake.

CMake, in turn, will use `CMakeLists.txt` files to generate `makefiles`.
`CMakeLists.txt` contains the necessary information to build this `makefile`.

## CMakeLists.txt {#arcanedoc_examples_simple_example_cmake_cmakeliststxt}

Here is the CMakeLists.txt provided by `arcane_template`:
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
Let's start:
```cmake
cmake_minimum_required(VERSION 3.16)
```
This first line requests the presence of CMake version 3.16 or higher.
This ensures that CMake can recognize all the commands we give it.

____

```cmake
project(HelloWorld LANGUAGES CXX)
```
We give the name of the project and the language it is written in (`CXX` =
`C++`).

____

```cmake
find_package(Arcane REQUIRED)
```
Our project needs %Arcane installed (see the next section to tell CMake where
%Arcane is installed).

____

```cmake
add_executable(HelloWorld SayHelloModule.cc main.cc SayHello_axl.h)
```
We also give the different files that will compose our executable.
\note
No need to include `SayHelloModule.hh` since it is imported by
`SayHelloModule.cc`.

____

```cmake
arcane_generate_axl(SayHello)
```
We ask CMake to generate the file `SayHello_axl.h`.
We provide the position of the file `SayHello.axl` as an argument (without the
`.axl` extension).
Here, `SayHello.axl` is in the root of our project, so we just need to put
`SayHello`.

____

```cmake
arcane_add_arcane_libraries_to_target(HelloWorld)
```
We add the %Arcane libraries for our project.

____

```cmake
target_include_directories(HelloWorld PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
```
We include all the source files.

____

```cmake
configure_file(HelloWorld.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
```
We copy the `.config` into the build directory.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_main
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_build
</span>
</div>
