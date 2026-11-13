# ARCT file writer in CMake {#arcanetdoc_arct_cmake_arct_writer}

[TOC]

To write non-regression tests, we have created several CMake functions in order to build an ARCT file
step by step. These functions can replace the `add_test()` but the way they work is totally different.

`add_test()` adds a test by specifying its name and the command to execute to run the test. A test passes
if the command returns 0 (by default).

The contribution of the ArcaNet project adds an extra step: the ARCT file.

Instead of making `add_test()`, the user can use the provided CMake functions to describe their test
cases, add variations, etc. With ArcCher, it is also possible to run the tests with MPI (and maybe even
on a computer later).

Once the ARCT file is built, it is currently possible to convert it automatically into CTest via another
CMake function. Later, it will be possible to use this file in another test case launcher: Schedane or
maybe PCVS.

## ARCT file initialization

Two modes of use of these functions are available; one making it possible to write the ARCT file step by
step into a CMake variable and then, at the end, to write its content into a file; the other opens, reads,
writes and closes an ARCT file at each step.

The first mode makes it possible to optimize the performances a bit but requires having access to the
CMake variable in read and write mode at each call to the ARCT CMake functions (so no user functions,
only macros).

The second is simpler but can become slow for big ARCT files. Moreover, it requires to give the location
of the ARCT file each time (if the default one is not used).

In all the examples of this documentation, we will use the second mode. So we will present the first mode
now.

This mode requires the use of two functions. Between these two functions, the ARCT file can be described
with the functions presented later.

--------------------------

### arct_begin

```cmake
arct_begin(OVERWRITE_ARCT ARCT_PATH <path>)
```

This first function opens the ARCT file and copies it into a CMake variable if it exists. If this
file does not exist, the variable will simply contain an empty JSON object.

One option is available:

- `OVERWRITE_ARCT`: overwrites the existing ARCT file if it exists.

One optional `oneValueArgs` argument is available:

- `ARCT_PATH <path>`: gives the location of the ARCT file to write. The default value is
  `${CMAKE_BINARY_DIR}/testlist.arct`.

--------------------------

### arct_end

```cmake
arct_end()
```

This function simply writes the ARCT file from the CMake variable.

--------------------------

In the second mode, the file is opened/read by the `arct_config_create_or_edit_XXX()` functions and is
written by the `arct_config_create_or_edit_end()` function.

No need to specify one mode or the other: the use of `arct_begin()` is enough to determine the use of the
first mode.

\warning It is not yet possible to create custom reserved blocks. The only ones available are those of
Arcane and ArcCher.

## Opening and closing configuration blocks

In an ARCT file, there are four parts containing configuration blocks:

- `general`,
- `cases`,
- `commons`,
- `variations`.

For each of these parts, there is a function to create or edit a configuration block:

- `arct_config_create_or_edit_general()`,
- `arct_config_create_or_edit_case()`,
- `arct_config_create_or_edit_common()`,
- `arct_config_create_or_edit_variation()`.

And a function to end the edition (and write the ARCT file in the case of the second mode):

- `arct_config_create_or_edit_end()`.

The ARCT CMake functions between the `arct_config_create_or_edit_XXX()` functions and
`arct_config_create_or_edit_end()` must be able to modify the CMake variables defined by
`arct_config_create_or_edit_XXX()`.

The `arct_config_create_or_edit_XXX()` functions being able to substitute for the `arct_begin()` function,
these functions therefore take back the parameters of `arct_begin()`.

These functions make it possible to create a part of an ARCT file (if it does not already exist) and to
modify it. All the ARCT CMake functions between `arct_config_create_or_edit_XXX()` and
`arct_config_create_or_edit_end()` will concern the `XXX` block.

--------------------------

### arct_config_create_or_edit_general

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT ARCT_PATH <path>)
```

This function makes it possible to create and/or modify the `general` configuration block of the ARCT
file:

```json5
{
  "general": {
    // <config_block>
  }
}
```

One option is available:

- `OVERWRITE_ARCT`: overwrites the existing ARCT file if it exists. No effect if the first mode is used.

One optional `oneValueArgs` argument is available:

- `ARCT_PATH <path>`: gives the location of the ARCT file to write. The default value is
  `${CMAKE_BINARY_DIR}/testlist.arct`. No effect if the first mode is used.

--------------------------

### arct_config_create_or_edit_case

```cmake
arct_config_create_or_edit_case(OVERWRITE_ARCT ARCT_PATH <path> NAME_CASE <name_case>)
```

This function makes it possible to create and/or modify one of the configuration blocks of the `cases`
part of the ARCT file:

```json5
{
  "cases": {
    "<name_case>": {
      // <config_block>
    }
  }
}
```

One option is available:

- `OVERWRITE_ARCT`: overwrites the existing ARCT file if it exists. No effect if the first mode is used.

Two `oneValueArgs` arguments are available:

- `ARCT_PATH <path>` (optional): gives the location of the ARCT file to write. The default value is
  `${CMAKE_BINARY_DIR}/testlist.arct`. No effect if the first mode is used.
- `NAME_CASE <name_case>` (**mandatory**): gives the name of the base case to create/modify afterwards.

--------------------------

### arct_config_create_or_edit_common

```cmake
arct_config_create_or_edit_common(OVERWRITE_ARCT ARCT_PATH <path> NAME_COMMON <name_common>)
```

This function makes it possible to create and/or modify one of the configuration blocks of the `commons`
part of the ARCT file:

```json5
{
  "commons": {
    "<name_common>": {
      // <config_block>
    }
  }
}
```

One option is available:

- `OVERWRITE_ARCT`: overwrites the existing ARCT file if it exists. No effect if the first mode is used.

Two `oneValueArgs` arguments are available:

- `ARCT_PATH <path>` (optional): gives the location of the ARCT file to write. The default value is
  `${CMAKE_BINARY_DIR}/testlist.arct`. No effect if the first mode is used.
- `NAME_COMMON <name_common>` (**mandatory**): gives the name of the common configuration to
  create/modify afterwards.

--------------------------

### arct_config_create_or_edit_variation

```cmake
arct_config_create_or_edit_variation(OVERWRITE_ARCT ARCT_PATH <path> NAME_VARIATION <name_variation> NAME_VARIANT <name_variant>)
```

This function makes it possible to create and/or modify one of the configuration blocks of a variant of a
variation of the `variations` part of the ARCT file:

```json5
{
  "variations": {
    "<name_variation>": {
      "<name_variant>": {
        // <config_block>
      }
    }
  }
}
```

One option is available:

- `OVERWRITE_ARCT`: overwrites the existing ARCT file if it exists. No effect if the first mode is used.

Three `oneValueArgs` arguments are available:

- `ARCT_PATH <path>` (optional): gives the location of the ARCT file to write. The default value is
  `${CMAKE_BINARY_DIR}/testlist.arct`. No effect if the first mode is used.
- `NAME_VARIATION <name_variation>` (**mandatory**): gives the name of the variation to create/modify
  afterwards.
- `NAME_VARIANT <name_variant>` (**mandatory**): gives the name of the variant to create/modify
  afterwards.

--------------------------

### arct_config_create_or_edit_end

```cmake
arct_config_create_or_edit_end()
```

This function updates the variable containing the ARCT JSON. Moreover, in the second mode, it writes the
variable into the ARCT file.

--------------------------

## Modifying the properties of the ARCT reserved block

Several functions are available to modify the properties of the ARCT reserved block:

```json5
{
  "XXX": {
    "_": {
      // <reserved_block>
    }
  }
}
```

*`XXX` corresponds to the type of block chosen when calling `arct_config_create_or_edit_XXX()`.*


--------------------------

### arct_config_define_name

```cmake
arct_config_define_name(NAME <name>)
```

This function adds a name for a configuration block. If a name already exists, it will be overwritten:

```json5
{
  "XXX": {
    "_": {
      "name": "<name>"
    }
  }
}
```

\warning This is a display name: the name of the configuration block does not change.

This name can therefore contain all the symbols you want.

One `oneValueArgs` argument is available:

- `NAME <name>` (**mandatory**): gives a display name to the configuration block.

--------------------------

### arct_config_dependency

```cmake
arct_config_dependency([BEFORE|AFTER] ADD <name_dep>)
```

This function adds a dependency to a configuration block.

By default, the `BEFORE` option is selected.

If the `BEFORE` option is selected:

```json5
{
  "XXX": {
    "_": {
      "depend_b": [..., "<name_dep>"]
    }
  }
}
```

If the `AFTER` option is selected:

```json5
{
  "XXX": {
    "_": {
      "depend_a": [..., "<name_dep>"]
    }
  }
}
```

Two options are available:

- `BEFORE`: adds a *before* dependency. Default option.
- `AFTER`: adds an *after* dependency.

One `oneValueArgs` argument is available:

- `ADD <name_dep>` (**mandatory**): adds a dependency at the end of the dependency array.

--------------------------

### arct_config_variation

```cmake
arct_config_variation([BEFORE|AFTER] ADD <name_variation> [EQUAL <name_variant>|INCLUDE <name_variant>...|EXCLUDE <name_variant>...])
```

This function adds or modifies a variation-type dependency.

By default, the `BEFORE` option is selected.

If the `BEFORE` option is selected:

```json5
{
  "XXX": {
    "_": {
      "depend_b": [..., "<name_variation><equal_include_exclude><name_variant>"]
    }
  }
}
```

If the `AFTER` option is selected:

```json5
{
  "XXX": {
    "_": {
      "depend_a": [..., "<name_variation><equal_include_exclude><name_variant>"]
    }
  }
}
```

The `ADD` argument adds a variation to the dependency list. If the variation is already present in the
list, the function will add a variant limiter to this dependency (if the list contains `nb_proc!1`,
`ADD nb_proc EXCLUDE 4` would give `nb_proc!1!4`).

Two options are available:

- `BEFORE`: adds a *before* dependency. Default option.
- `AFTER`: adds an *after* dependency.

Three `oneValueArgs` arguments are available:

- `ADD <name_variation>` (**mandatory**): adds a variation at the end of the dependency array or adds one
  or several variant inclusions/exclusions to a variation already present in the dependency array.
- `EQUAL <name_variant>` (optional): resolves the variation passed with the `ADD` argument (`=` symbol).

Two `multiValueArgs` arguments are available:

- `INCLUDE <name_variant>...` (optional): restricts the variation passed with the `ADD` argument to
  certain variants (`+` symbol).
- `EXCLUDE <name_variant>...` (optional): excludes variants from the variation passed with the `ADD`
  argument (`!` symbol).

## Modifying the properties of the Arcane reserved block

Several functions are available to modify the properties of the Arcane reserved block:

```json5
{
  "XXX": {
    "arcane": {
      // <reserved_block>
    }
  }
}
```

*`XXX` corresponds to the type of block chosen when calling `arct_config_create_or_edit_XXX()`.*

--------------------------

### arct_arcane_dataset

```cmake
arct_arcane_dataset(PATH <path_dataset>)
```

This function defines a dataset for Arcane. If a path already exists, it will be overwritten:

```json5
{
  "XXX": {
    "arcane": {
      "dataset": "<path_dataset>"
    }
  }
}
```

One `oneValueArgs` argument is available:

- `PATH <path_dataset>` (**mandatory**): gives a path to find the dataset. The path can be relative to
  the working directory.

--------------------------

### arct_arcane_option

```cmake
arct_arcane_option(ADD <name_option> VALUE <value_option>)
```

This function adds an option to the list of options to give to Arcane:

```json5
{
  "XXX": {
    "arcane": {
      "options": {
        ...,
        "<name_option>": "<value_option>"
      }
    }
  }
}
```

A classic Arcane option is passed through the command line. For example: `-A,MaxIteration=10` gives here
`arct_arcane_option(ADD "MaxIteration" VALUE "10")`.

Compatible with all the Arcane options (even the dataset modifications, example:
`arct_arcane_option(ADD "//meshes/mesh[]/generator/nb-part-x" VALUE "4")`).

Two `oneValueArgs` arguments are available:

- `ADD <name_option>` (**mandatory**): gives the name of the option to add.
- `VALUE <value_option>` (**mandatory**): gives the value of the option to add.

## Modifying the properties of the ArcCher reserved block

Several functions are available to modify the properties of the ArcCher reserved block (Arcane launCher):

```json5
{
  "XXX": {
    "arccher": {
      // <reserved_block>
    }
  }
}
```

*`XXX` corresponds to the type of block chosen when calling `arct_config_create_or_edit_XXX()`.*

--------------------------

### arct_arccher_mpi

```cmake
arct_arccher_mpi(PATH <path_mpirun> NB_PROC <nb_proc_mpi>)
```

This function defines the MPI options:

```json5
{
  "XXX": {
    "arccher": {
      "mpi_binary": "<path_mpirun>",
      "nb_proc": <nb_proc_mpi>
    }
  }
}
```

Two actions can be done, either in a single call, or in two:

```cmake
arct_arccher_mpi(PATH /usr/bin/mpirun NB_PROC 4)
```

is equivalent to:

```cmake
arct_arccher_mpi(PATH /usr/bin/mpirun)
arct_arccher_mpi(NB_PROC 4)
```

Two `oneValueArgs` arguments are available:

- `PATH <path_mpirun>` (optional): gives a path for the `mpirun/mpiexec` binary.
- `NB_PROC <nb_proc_mpi>` (optional): gives a number of processes to use.

--------------------------

### arct_arccher_thread

```cmake
arct_arccher_thread(NB_THREAD <nb_thread>)
```

This function defines the number of thread:

```json5
{
  "XXX": {
    "arccher": {
      "nb_thread": <nb_thread>
    }
  }
}
```

One `oneValueArgs` arguments are available:

- `NB_THREAD <nb_thread>` (optional): gives a number of thread to use.

--------------------------

### arct_arccher_envvar

```cmake
arct_arccher_envvar(ADD <name_envvar> VALUE <value_envvar>)
```

This function defines environment variables which will be set at launch:

```json5
{
  "XXX": {
    "arccher": {
      "env_var": {
        "<name_envvar>": "<value_envvar>"
      }
    }
  }
}
```

Two `oneValueArgs` arguments are available:

- `ADD <name_envvar>` (**mandatory**): gives the name of the environment variable to add.
- `VALUE <value_envvar>` (**mandatory**): gives the value of the environment variable to add.

--------------------------

### arct_arccher_executable

```cmake
arct_arccher_executable(PATH <path_executable>)
```

This function defines where the executable of the program to be launched by ArcCher is located:

```json5
{
  "XXX": {
    "arccher": {
      "code_binary": "<path_executable>"
    }
  }
}
```

One `oneValueArgs` argument is available:

- `PATH <path_executable>` (optional): gives a path for the executable to launch.

--------------------------

### arct_arccher_working_dir

```cmake
arct_arccher_working_dir(PATH <path_workingdir>)
```

This function defines where the working directory is. This makes it possible to use relative paths for
the executable, the dataset, the meshes of the dataset, etc.

```json5
{
  "XXX": {
    "arccher": {
      "working_dir": "<path_workingdir>"
    }
  }
}
```

One `oneValueArgs` argument is available:

- `PATH <path_workingdir>` (optional): gives the path of the working directory.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cmake_arct_to_ctest
</span>
</div>
