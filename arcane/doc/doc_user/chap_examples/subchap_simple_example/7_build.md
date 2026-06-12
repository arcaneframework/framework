# Compilation and Launch {#arcanedoc_examples_simple_example_build}

[TOC]

To compile our HelloWorld application, we will use CMake.

Here is the list of commands to complete to compile and launch our application:

```sh
ARCANE_INSTALL_PATH=#A compléter

HW_SOURCE_DIR=#A compléter
HW_BUILD_DIR=#A compléter
HW_BUILD_TYPE=Release
HW_EXE=${HW_BUILD_DIR}/HelloWorld
HW_ARC=${HW_SOURCE_DIR}/HelloWorld.arc

cd ${HW_BUILD_DIR}

cmake \
  -S ${HW_SOURCE_DIR} \
  -B ${HW_BUILD_DIR} \
  -DCMAKE_PREFIX_PATH=${ARCANE_INSTALL_PATH} \
  -DCMAKE_BUILD_TYPE=${HW_BUILD_TYPE}

cmake --build ${HW_BUILD_DIR}

${HW_EXE} ${HW_ARC}
```
At the beginning of this list, we define variables to make everything more
readable and easily modifiable.

```sh
ARCANE_INSTALL_PATH=# A compléter
```
This line allows us to define the installation directory of %Arcane.
\note
Example line:
```sh
ARCANE_INSTALL_PATH=~/install_arcane
```

____

```sh
HW_SOURCE_DIR=# A compléter
```
This line allows us to define the directory containing the sources for our
HelloWorld.
\note
Example line:
```sh
HW_SOURCE_DIR=~/src_helloworld
```

____

```sh
HW_BUILD_DIR=# A compléter
```
This line allows us to define the directory where our application will be
compiled.
\note
Example line:
```sh
HW_BUILD_DIR=~/build_hw
```

____

```sh
HW_BUILD_TYPE=Release
```
This line allows us to define the desired build type.
We have the choice between `Debug`, `Check`, and `Release`.
`Debug` allows us to have additional information to debug with a debugger.
`Check` allows us to add extra checks to prevent problems like array overflow.

\note `Debug` automatically includes `Check`.

____

```sh
HW_EXE=${HW_BUILD_DIR}/HelloWorld
HW_ARC=${HW_SOURCE_DIR}/HelloWorld.arc

cd ${HW_BUILD_DIR}
```
Using the information we provided previously, we can deduce the location of the
executable (`HW_EXE`) and the location of our dataset (`HW_ARC`).

Personally, I prefer to be in the build folder for the following commands, so I
execute a `cd ${HW_BUILD_DIR}`, but this remains optional.

____

```sh
cmake \
  -S ${HW_SOURCE_DIR} \
  -B ${HW_BUILD_DIR} \
  -DCMAKE_PREFIX_PATH=${ARCANE_INSTALL_PATH} \
  -DCMAKE_BUILD_TYPE=${HW_BUILD_TYPE}
```
We ask CMake to configure the build directory using the CMakeLists.txt files so
that Make can compile our project. This command will not modify the directory
containing our project sources (this is true for all the commands presented
here, by the way).

____

```sh
cmake --build ${HW_BUILD_DIR}
```
We ask CMake to call Make to compile our project.
We could use the `make` command directly, but if one day we want to use a
program other than Make (like Ninja), we will have to change this command.
Currently, CMake handles it according to the configuration made previously.

____

```sh
${HW_EXE} ${HW_ARC}
```
Finally, we can launch our HelloWorld!
We must also specify the location of our dataset.
If we have multiple datasets, we just need to change the `${HW_ARC}` variable
and relaunch HelloWorld without needing to recompile.

____

That concludes this subsection dedicated to building a hello world with %Arcane.
The basics are presented here. Nevertheless, to go further, it is recommended to
read all chapters of this documentation.

If there is a problem in this subsection, you can open an issue on the %Arcane
GitHub.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_cmake
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example
</span>
</div>
