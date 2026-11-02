# Principles of the ARCT format {#arcanetdoc_arct_basis}

[TOC]

An ARCT file describes one or more use cases, for a given project.

To obtain the desired result, several programs may have to be launched in a chain. This file makes it
possible to store parameters for each program. However, this file does not store the execution order of
these programs.

For Arcane for example (or for programs using Arcane), it is possible to give it as a parameter the path
of an ARCT file and the name of a use case for Arcane to execute it.

Obviously, not all programs will support this file format. For those, the ArcCher launcher will take care
of reading the ARCT file and of passing them the right parameters, so that it is transparent for the
user.

The ARCT file format was initially designed to replace the variation files existing in some codes using
Arcane.

A variation file describes one or more property variations of an Arcane dataset (.arc).

```xml
<?xml version='1.0' ?>
<variations>
  <variation>
    TODO
  </variation>
</variations>
```

With an intermediate program, it is possible to generate one or more dataset files, one per possible
variation.

```txt
[dataset.arc] + [auto_variations.xml]
->
- [variant1/dataset.arc]
- [variant2/dataset.arc]
- ...
```

The approach adopted by the ARCT format is different.

Instead of generating dataset variations, we sought to generate execution variations.

## Basic building blocks

An ARCT file has this structure:

```json5
{
  "versions": {
    "_": 1,
    "arcane": 1,
    "arccher": 1
  },
  
  "general": {
    // <config_block>
  },
  
  "cases": {
    "case_1": {
      // <config_block>
    }
  },
  
  "commons": {
    "common_config_1": {
      // <config_block>
    }
  },
  
  "variations": {
    "variation_1": {
      "variant_1": {
        // <config_block>
      }
    }
  }
}
```

A configuration block (`<config_block>`) is a JSON object bringing a piece of configuration to one or
several programs.

A configuration block is itself made up of blocks reserved for programs (which are also JSON objects).
For example:

```json5
{
  "_": {
    // <reserved_block>
  },
  "arcane": {
    // <reserved_block>
  },
  "arccher": {
    // <reserved_block>
  }
}
```

The `arcane` block is reserved for Arcane. It is up to Arcane to define its content and to version it. It
does not depend on the ARCT format.

A requesting program is a program which will take an ARCT file as an argument and which will use an ARCT
file reader to get the configuration it will have to apply.

The ARCT file reader will provide one or several lists of configuration blocks to the requesting program.

The requesting program will have to tell the reader whether it supports multi-case (launch of several
cases at once) or not. If it does not support it, then only a single list of blocks will be provided by
the reader (otherwise, error).

```txt
[ARCT file reader] + [file.arct]
->
[
  [<config_block_1>, <config_block_2>, ...],
  [<config_block_3>, <config_block_2>, ...],
  ...
]
```

The requesting program will have to apply the configuration contained in each block. It will choose how to
merge the information of these blocks.

Example:
We can have a block giving the number of iterations to do:

```json
{
  "_": {
    "name": "10 iterations"
  },
  "arcane": {
    "options": {
      "MaxIteration": "10"
    }
  }
}
```

Another one which runs Arcane with 4 shared-memory subdomains:

```json
{
  "_": {
    "name": "4 threads"
  },
  "arcane": {
    "options": {
      "S": "4"
    }
  }
}
```

Arcane applies the options one by one and if an option appears in two blocks, the most recent one is kept.

## Block order

In an ARCT file reader, a list of configuration blocks is simply a list of JSON objects sorted according
to the reading order which is necessary to follow.

To write these lists in a humanly readable way, this syntax has been defined:

- `{}` designates a block as well as its dependencies,
- `[]` designates the parent block; thus, all the blocks before are "before" dependencies and the blocks
  after are "after" dependencies,
- `->` is simply a block separator which shows the reading order.

Example:

```txt
{ config_1 -> [general] } -> { { [config_2] -> config_4 } -> [case_1] -> config_3 }
```

We can see that we have two main blocks: `[general]` and `[case_1]`:

- `[general]` has a "before" dependency which is `config_1`.
- `[case_1]` has two dependencies:
  - an "after" dependency: `config_3`
  - a "before" dependency: `[config_2]` which has an "after" dependency:
    - `config_4`

The list of JSON objects, to be read from left to right in order, is simply this one:

```txt
array = [config_1, general, config_2, config_4, case_1, config_3];
```

\note We just replace the `->` by `,` and remove the `[]` and the `{}`...

If we set aside the dependencies, the reading order is systematically this one:

```txt
{ [general] } -> { [case_1] }
```

`general` is always the first block (unless it is not present). Then, the block representing the base case
chosen by the user (here: `case_1`) is added to the list.

## Use case names

There are two types of (use case) names: complete names and incomplete names.

A complete name is a name which, associated with an ARCT file, designates a single use case (a single list
of configuration blocks).

An incomplete name is a name which, associated with an ARCT file, designates several use cases (several
lists of configuration blocks).

These two types of names are related to variations.

A resolved variation is a variation for which we know the variant which is necessary to put in the list of
configuration blocks.

An unresolved variation is a variation for which we do not know which variant to choose from the
information of the ARCT file.

If a block contains an unresolved variation, we will have several possible executions. It is possible to
choose, in the use case name, the variant to use. The variation is thus resolved with a complete name. If
an incomplete name is used, then we will have several lists of configuration blocks.

To sum up, in a use case name, the name of the base case must be found first, then the different variation
resolutions if necessary.

To build these names, we use the following symbols: `:`, `~`, `=`, `!` and `+`.

The `:` symbol separates the different elements of the case name. It must be used only to separate the
different elements making up the case name. It must not be put at the beginning or at the end of the case
name (if there is no base case to run, only variations to resolve, no `:` at the beginning).

The `=` symbol makes it possible to resolve a variation.

The `!` symbol excludes variants from the set of variants available to run.

The `+` symbol includes variants to run, among all the available variants.

The `!` and `+` symbols restrict the names of variants to process. With these symbols, it is impossible to
totally resolve a variation. These symbols can therefore be used only in the case of incomplete names.

Example: `case_1:nb_proc=4:nb_iteration+10+20:eos-model!StiffenedGas`

## The format

In the following pages, several examples will be explained, from the simplest to the most complete. Two
programs will be used: Arcane and ArcCher. The specifications of their reserved blocks will be presented
throughout the examples.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_general
</span>
</div>
