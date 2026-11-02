# ARCT v1 {#arct_specs}

ArcaNet File. Version 1.

## File name extension

ArcaNet files use the `.arct` extension.

## Goals

This file aims at describing how to launch one or more use cases for a given project.

All the programs of the computing chain of this project can take this file as a parameter, as well as the name of
the case to execute, and it is up to the programs to configure themselves according to the given parameters.

The construction of the case name will also be specified in this file.

This file does not describe the launch order of the programs, only the launch/configuration parameters.

In this file, each program can have one or more JSON objects totally reserved for it, at its own charge (up to it
to update them, for example).

In practice, not all the programs of the computing chain will support this type of file. A program launcher,
supporting this type of file, can take care of the configuration of these programs, in order to make this
incompatibility invisible for the end user. This program launcher will not be specified here.

## General operation

A base block of this file schema is a JSON object called **config_block** ("configuration block") containing JSON
objects intended for the requesting programs, called **reserved_block** ("reserved block (for a requesting
program)").

The basic principle of this file is to build one or more lists of **config_block**, lists which will depend on the
case name given by the user.

A requesting program may or may not support the launch of several use cases at once. If the program can only
launch a single case at a time, then it will only get a single list of **config_block**, and the user will have to
make sure that their ARCT file and case name, once processed, designate only a single use case (an error must be
displayed if an incompatibility is detected).

The order of the blocks of these lists depends on the different dependencies between the configuration blocks.

The requesting program, via an ARCT file reader, will get these lists and will have to process them, block by
block. This processing is dependent on the requesting program (example: if an array is present in two of these
blocks, it can merge them or overwrite them or...).

The requesting program will have lists of configuration blocks, with the reserved blocks of all the programs. It
can read the reserved blocks of other programs, even update them if it wishes, but it will have to take into
account the specifications imposed by the program owning the block.

The ARCT file reader must support all the features and the constraints exposed here to display the support of
this file schema, in the version shown on the first line of this file.

## Specifications

- ARCT is a standard JSON file schema,
- Its encoding must be UTF-8 (with or without BOM),
- The property names will follow the `snake_case` convention.

### JSON comments (JSONC)

Comments in JSON (`//` and `/* */`) are an unofficial extension of the JSON format. Thus, not all JSON
readers/writers necessarily support them.

This schema does not require comment support for JSON readers.

Nor for writers. Thus, an update of an ARCT file can cause a loss of comments.

For these reasons, the use of JSONC and comments is discouraged (the examples of this document use comments as a
documentation convention).

To work around this, it is possible to comment a classic ARCT/JSON file with the property names `"//X"`, `X`
designating any number or other (the properties starting with `//` will not be used).

Example:

```json
{
  "//1": "Any comment."
}
```

## General schema

The ARCT schema is made up of five JSON objects at the root:

- `general`: general configuration,
- `cases`: configurations of the use cases,
- `commons`: common configurations,
- `variations`: configuration variations with several variants,
- `versions`: versions of the **reserved_block**.

The minimal ARCT file only requires the `versions` object filled as follows:

```jsonc
{
  "versions": {
    "_": 1
  }
}
```

### General

This object only contains a configuration block:

```jsonc
{
  "general": {
    // <config_block>
  }
}
```

This object is optional.

### Cases

This object must contain all the use cases for the project.

Example of this object:

```jsonc
{
  "cases": {
    "case_1": {
      // <config_block>
    },
    "case_2": {
      // <config_block>
    }
  }
}
```

This object is optional.

The format of the base case names is free (except some characters which are reserved by the schema). By usage, we
use either `camelCase` or `snake_case`, without spaces.

There cannot be two cases with the same name.

Also, the name of the base cases must be different from the name of the common configurations and from the name of
the configuration variations. If this is not the case, and as the JSON format does not forbid it, an error must be
emitted by the ARCT file reader.

#### Reserved characters

The following characters cannot be used in the name of the base cases:

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Commons

This object must contain all the common configurations (configurations which can be requested by other
configurations).

Example of this object:

```jsonc
{
  "commons": {
    "common_config_1": {
      // <config_block>
    },
    "common_config_2": {
      // <config_block>
    }
  }
}
```

This object is optional.

The format of the name of the configurations is free (except some characters which are reserved by the schema).
By usage, we use either `camelCase` or `snake_case`, without spaces.

There cannot be two configurations with the same name.

The name of the configurations must be different from the name of the use cases and from the name of the
configuration variations. If this is not the case, and as the JSON format does not forbid it, an error must be
emitted by the ARCT file reader.

#### Reserved characters

The following characters cannot be used in the name of the configuration:

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Variations/Variants

This object must contain all the configuration variations, themselves containing all the variants of the
configuration.

Example of this object:

```jsonc
{
  "variations": {
    "variation_1": {
      "variant_1_of_variation_1": {
        // <config_block>
      },
      "variant_2": {
        // <config_block>
      }
    },
    "variation_2": {
      "variant_1_of_variation_2": {
        // <config_block>
      },
      "variant_2": {
        // <config_block>
      }
    }
  }
}
```

This object is optional.

The format of the name of the variations and variants is free (except some characters which are reserved by the
schema). By usage, we use either `camelCase` or `snake_case`, without spaces.

There cannot be two variations with the same name. Within a variation, there cannot be two variants with the same
name. However, two variants of two distinct variations can have the same name.

The name of the variations must be different from the name of the use cases and from the name of the common
configurations. If this is not the case, and as the JSON format does not forbid it, an error must be emitted by
the ARCT file reader. The name of the variants can be identical to the names of the variations, to the names of
the use cases and to the names of the common configurations.

#### Reserved characters

The following characters cannot be used in the name of the variations/variants:

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Versions

This object must contain the version of each **reserved_block**.

Example of this object:

```jsonc
{
  "versions": {
    "_": 1,
    "program_1": 1,
    "program_2": [1, 0]
  }
}
```

The `"_"` key is reserved for the version number of the file schema itself, described in this file, as well as
for the version of the ARCT reserved block.

The only element which will not be able to evolve in the ARCT file schema is the position of this version number.
Thus, all the past and future ARCT file readers will be able to determine their ability to read the file only
with this number.

The format of the version value is imposed. It must be either a positive integer, or an array of positive
integers.

If a requesting program owns a classic reserved block and/or an alternative reserved block in a configuration
block, it must version it, so put the version number in the `versions` object. The classic and alternative
reserved blocks share the same version number. The name of the alternative must therefore not appear in the
`versions` object.

The requesting program must define a specification for each version, which it can make public if it wishes other
programs to be able to read its reserved block correctly. The semantics of the version number must also be
specified (to be able to read and to be able to compare the version numbers correctly).

This program cannot mix several versions of these blocks.

## Schema of the configuration block

### General schema

The configuration block is a JSON object containing reserved blocks (themselves JSON objects).

A configuration block can be an empty object.

Example of this object:

```jsonc
{
  "_": {
    // <reserved_block>
  },
  "program_1": {
    // <reserved_block>
  },
  "program_2": {
    // <reserved_block>
  }
}
```

Each object of the configuration block must be optional.

If an object is present, even empty, a version number must be present in the part provided for this purpose.

The format of the program names is free (except some characters which are reserved by the schema). The only
reserved name is `"_"`.

#### Reserved characters

The following characters cannot be used in the name of the reserved blocks:

- `:`,
- `=`,
- `~` (except to define an alternative reserved block),
- `!`,
- `+`.

### Alternative reserved blocks

The JSON file format imposes that the names be unique. Nevertheless, it can be necessary to have several reserved
blocks appear for the same program, depending on the launch mode for example. These are the alternative reserved
blocks (in opposition to the classic reserved blocks).

One of the reserved characters is precisely provided for this case: `~`.

Example:

```jsonc
"case_1": {
  "_": {
    // <reserved_block>
  },
  "program_1~init": {
    // <reserved_block>
  },
  "program_1": {
    // <reserved_block>
  }
}
```

To request the launch of this alternative path, it must be specified in the case name when launching the
program.

A classic launch would be done with the case name `case_1`. To launch an alternative path, the name of the
alternative must be added to the case name, separated by the `~` symbol: `case_1~init`.

We know that the requesting program will receive a list of configuration blocks. So, in theory, it would be up to
it to take care of retrieving the right block according to the chosen alternative. To facilitate this retrieval,
the ARCT file reader must provide a method taking a configuration block as an argument and returning the adequate
reserved block.

For a given configuration block and for a requested alternative:

- if the alternative reserved block with the right name exists, it is used,
- if the alternative reserved block with the right name does not exist, the classic reserved block is used,
- if the classic reserved block does not exist, an empty block is returned (no error, the reserved blocks being
  optional).

It is important to note that the name of the base case is optional (it can be useless if we want to launch only
the `general` block). A case name can therefore start with `~`. However, `~` alone is not allowed.

### ARCT reserved block

This reserved block is described by this specification file and can be used by the requesting programs (it is up
to them to pay attention to the version number of the file).

This block has several reasons to exist:

- to bring details on the use of this configuration block (with a more precise name, for example),
- to bring common parameters for all the requesting programs,
- to easily extend the file schema without too many difficulties,
- to describe the dependencies between the configuration blocks.

Here is an exhaustive example of this block:

```jsonc
{
  "name": "Example of case No. 1",
  "depend_b": ["common_block_1"],
  "depend_a": ["variation_1"]
}
```

The name can give more details on the function of this block.

The `depend_b` property describes the dependencies of the block to read before this block ("b" stands for
"before").

The `depend_a` property describes the dependencies of the block to read after this block ("a" stands for
"after").

The requesting program can read these properties if it wishes. The resolution of the dependencies is a task
reserved for the ARCT file reader. The requesting program gets a list of configuration blocks in the right
order.

## Reading order of the configuration blocks and dependencies

### Syntax of a list of configuration blocks

In an ARCT file reader, a list of configuration blocks is simply a list of JSON objects sorted according to the
reading order which is necessary to follow.

To write these lists in a humanly readable way, this syntax has been defined:

- `{}` designates a block as well as its dependencies,
- `[]` designates the parent block; thus, all the blocks before are "before" dependencies and the blocks after
  are "after" dependencies,
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

The list of JSON objects is simply:

```txt
array = [config_1, general, config_2, config_4, case_1, config_3];
```

### General case (without dependencies)

Let's take this simple example:

```jsonc
// file.arct
{
  "versions": {
    "_": 1,
    "program_1": 1
  },
  "general": {
    "program_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "case_1": {
      "program_1": {
        // <reserved_block>
      }
    }
  }
}
```

And let's suppose that `program_1` takes the ARCT file as an argument via the `-j` option and the case name via
the `-c` option.

#### Case 1

If we specify only the path of the ARCT file, without case name, only the `general` configuration block will be
read:

`./program_1 -j file.arct`

```txt
{ [general] }
```

If the `general` block is not present, then no block will be presented to the requesting program. It is up to the
requesting program to decide whether it must emit an error or not. The file itself will be valid.

#### Case 2

Let's now add the case name `case_1`:

`./program_1 -j file.arct -c case_1`

```txt
{ [general] } -> { [case_1] }
```

Without dependencies, the `general` block is always read first and the `case` block second.

But be careful: all the configuration blocks are optional. If the `general` block does not exist, it will,
obviously, not be read. Only the `case` block will be (first, then).

However, if the case `case_1` is not present in the ARCT file while it is requested by the requesting program,
then the reader must emit an error.

Reminder: it is up to the requesting program to process the list of blocks which is presented to it (up to it to
determine how to process the duplicate properties, for example).

### Case with dependencies

#### Case 1

Let's add `commons` blocks and a dependency in the example:

```jsonc
// file.arct
{
  "versions": {
    "_": 1,
    "program_1": 1
  },
  "general": {
    "program_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "case_1": {
      "_":{
        "depend_a": ["config_1"]
      },
      "program_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "config_1": {
      "program_1": {
        // <reserved_block>
      }
    }
  }
}
```

The number of blocks read will change if we request the case `case_1`. Indeed, in this block, we find the
`depend_a` property which makes it possible to request the reading of blocks after `case_1`:

`./program_1 -j file.arct -c case_1`

```txt
{ [general] } -> { [case_1] -> config_1 }
```

If we replace the `depend_a` property by `depend_b`, we will get a list like this one:

```txt
{ [general] } -> { config_1 -> [case_1] }
```

#### Case 2

Let's complicate the example a bit:

```jsonc
// file.arct
{
  // "versions" and "general" blocks omitted.
  "cases": {
    "case_1": {
      "_":{
        "depend_b": ["config_1"],
        "depend_a": ["config_2", "config_3"]
      },
      "program_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "config_1": {
      "program_1": {
        // <reserved_block>
      }
    },
    "config_2": {
      "program_1": {
        // <reserved_block>
      }
    },
    "config_3": {
      "program_1": {
        // <reserved_block>
      }
    }
  }
}
```

We mix `depend_b` and `depend_a`:

`./program_1 -j file.arct -c case_1`

```txt
{ [general] } -> { config_1 -> [case_1] -> config_2 -> config_3 }
```

#### Case 3

Last example of this type:

```jsonc
// file.arct
{
  // "versions" block omitted.
  "general": {
    "_":{
      "depend_b": ["config_1"]
    },
    "program_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "case_1": {
      "_":{
        "depend_b": ["config_2"],
        "depend_a": ["config_3"]
      },
      "program_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "config_1": {
      "program_1": {
        // <reserved_block>
      }
    },
    "config_2": {
      "_":{
        "depend_a": ["config_4"]
      },
      "program_1": {
        // <reserved_block>
      }
    },
    "config_3": {
      "program_1": {
        // <reserved_block>
      }
    },
    "config_4": {
      "program_1": {
        // <reserved_block>
      }
    }
  }
}
```

All the configuration blocks can have dependencies:

`./program_1 -j file.arct -c case_1`

```txt
{ config_1 -> [general] } -> { { [config_2] -> config_4 } -> [case_1] -> config_3 }
```

#### Case with variations

The order of the blocks in the list with variations does not change compared to the order with `commons`
blocks:

```jsonc
// file.arct
{
  // "versions" block omitted.
  "general": {
    "_":{
      "depend_b": ["config_1"]
    },
    "program_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "case_1": {
      "_":{
        "depend_b": ["variation_1"],
        "depend_a": ["variation_2=variant_1"]
      },
      "program_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "config_1": {
      "program_1": {
        // <reserved_block>
      }
    }
  },
  "variations": {
    "variation_1": {
      "variant_1": {
        "program_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "program_1": {
          // <reserved_block>
        }
      }
    },
    "variation_2": {
      "variant_1": {
        "program_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "program_1": {
          // <reserved_block>
        }
      }
    }
  }
}
```

The syntax to resolve variations is explained in the next part. To make it simple, if we launch this case with
the same command line as before, we get several cases to read:

`./program_1 -j file.arct -c case_1`

```txt
{ config_1 -> [general] } -> { variation_1=variant_1 -> [case_1] -> variation_2=variant_1 }
{ config_1 -> [general] } -> { variation_1=variant_2 -> [case_1] -> variation_2=variant_1 }
```

The reading order however remains the same as with classic dependencies.

### Dependency limitations

Some rules have been established for the dependencies:

- No cycles (`{ config_1 -> config_2 -> config_1 -> ... }`): the cycles must be detected by the ARCT file reader
  and trigger an error.
- A configuration block can only be read once per use case (otherwise, the ARCT file reader emits an error).
- An unresolved variation can only be requested once per launch path (otherwise, the ARCT file reader emits an
  error).
- All the dependencies written in the `depend_b` and `depend_a` properties must exist.
- The `cases` configuration blocks cannot be dependencies, just like the `general` block.

### Shared dependencies (diamonds)

(Sub-section written by AI)

The dependencies of a launch path form a tree: the same configuration block cannot be requested by two blocks of
the same path. This is a direct consequence of the rule "a block can only be read once".

If sharing was allowed, the reader would have to decide at which position to insert the common dependency in the
list of configuration blocks: before the first requester, after the last one, between the two requesters... None
of these positions is more natural than another, and each requesting program would moreover have to decide how
to process the two readings of the same block. This complexity justifies the prohibition.

The workaround is to carry the shared dependency by the common ancestor of the blocks which request it.

Example of an invalid file (`foundation` would be read twice):

```jsonc
// file.arct
{
  "cases": {
    "case_1": {
      "_": {
        "depend_a": ["feature_1", "feature_2"]
      }
    }
  },
  "commons": {
    "foundation": {
    },
    "feature_1": {
      "_": {
        "depend_b": ["foundation"]
      }
    },
    "feature_2": {
      "_": {
        "depend_b": ["foundation"]
      }
    }
  }
}
```

Equivalent valid file:

```jsonc
// file.arct
{
  "cases": {
    "case_1": {
      "_": {
        "depend_b": ["foundation"],
        "depend_a": ["feature_1", "feature_2"]
      }
    }
  },
  "commons": {
    "feature_1": {
    },
    "feature_2": {
    }
  }
}
```

Note: two sibling variants of the same variation can depend on the same block: since only one variant is read per
use case, there is never a multiple reading.

The same rule applies to unresolved variations: a variation requested by two blocks of the same path (for
example by the base case and by a common block which depends on it) triggers an error. Otherwise, the reader
would have to decide which request (and therefore which filter) prevails, then at which position to insert the
block of the chosen variant in the list of configuration blocks. The dependency must, here too, be carried by the
common ancestor of the requesters.

## Variations

Let's take this example to illustrate:

```jsonc
// file.arct
{
  "versions": {
    "_": 1,
    "program_1": 1
  },
  "general": {
    "program_1": {
      // <reserved_block>
    }
  },
  // See below for the `cases` block.
  "commons": {
    "config_1": {
      "program_1": {
        // <reserved_block>
      }
    }
  },
  "variations": {
    "variation_1": {
      "variant_1": {
        "program_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "program_1": {
          // <reserved_block>
        }
      },
      "variant_3": {
        "program_1": {
          // <reserved_block>
        }
      }
    },
    "variation_2": {
      "variant_1": {
        "program_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "program_1": {
          // <reserved_block>
        }
      }
    }
  }
}
```

### Resolved variations

A resolved variation is a variation for which we know the variant which is necessary to put in the list of
configuration blocks.

Let's take the following `cases` block:

```jsonc
"cases": {
  "case_1": {
    "_":{
      "depend_a": ["variation_1=variant_1"]
    },
    "program_1": {
      // <reserved_block>
    }
  }
}
```

If we put a resolved variation in the `depend_a` property (or `depend_b`, it does not matter here), it behaves
like a base `commons` dependency:

`./program_1 -j file.arct -c case_1`

```txt
{ [general] } -> { [case_1] -> variation_1=variant_1 }
```

### Unresolved variations

An unresolved variation is a variation for which we do not know which variant to choose from the information of
the ARCT file.

Let's take a slightly different `cases` block:

```jsonc
"cases": {
  "case_1": {
    "_":{
      "depend_a": ["variation_1", "variation_2"]
    },
    "program_1": {
      // <reserved_block>
    }
  }
}
```

The purpose of an unresolved variation is to easily define several case combinations to, for example, test a
program.

This example actually defines six cases:

- `case_1:variation_1=variant_1:variation_2=variant_1`,
- `case_1:variation_1=variant_2:variation_2=variant_1`,
- `case_1:variation_1=variant_3:variation_2=variant_1`,
- `case_1:variation_1=variant_1:variation_2=variant_2`,
- `case_1:variation_1=variant_2:variation_2=variant_2`,
- `case_1:variation_1=variant_3:variation_2=variant_2`.

The order of the list of lists of configuration blocks resulting from unresolved variations is not defined in
this schema since all the use cases must be independent from each other. It is up to the reader to choose (the
order must however remain the same between two launches, for the same file).

Note: the examples always present unresolved variations in the `cases` blocks. We can put unresolved variations
in any configuration block. Even in unresolved variations.

### Definition of variations

Several symbols are reserved in order to refine the selection of the variations in a configuration block. We
have already seen the `=` symbol which makes it possible to resolve a variation straight away.

Two other symbols are available to restrict the number of variations: `!` and `+`.

These three symbols cannot be mixed for the same variation.

The `!` symbol excludes variants from the set of variants available to run.

The `+` symbol includes variants to run, among all the available variants.

If, for a variation, all its variants are excluded, an error must be emitted by the ARCT file reader.

#### Example

In the previous example, we had six cases.

Let's suppose that we want to exclude `variation_2=variant_2`. The `depend_a` property (or `depend_b`) must
contain:

```jsonc
"depend_a": ["variation_1", "variation_2!variant_2"]
```

In this case, we are left with only three cases:

- `case_1:variation_1=variant_1:variation_2=variant_1`,
- `case_1:variation_1=variant_2:variation_2=variant_1`,
- `case_1:variation_1=variant_3:variation_2=variant_1`.

We can also want to exclude `variation_1=variant_1` and `variation_1=variant_2`:

```jsonc
"depend_a": ["variation_1!variant_1!variant_2", "variation_2!variant_2"]
```

In this case, we are left with a single case:

- `case_1:variation_1=variant_3:variation_2=variant_1`.

We can get this result simply with `=` or `+`:

```jsonc
"depend_a": ["variation_1=variant_1", "variation_2=variant_1"]
```

```jsonc
"depend_a": ["variation_1+variant_1", "variation_2+variant_1"]
```

Remark: a variation is resolved if and only if a single variant is designated by `=`. `!` and `+` only filter
the set of candidate variants: even reduced to a single one, the variation remains unresolved and the name
remains incomplete; for a single-case program, the user must then resolve it with `=` in the case name.

| Symbol | Role       | Semantics                                                                      | Is the variation resolved? | Example (variants: 10, 20, 30) | Obtained case names                               |
|--------|------------|--------------------------------------------------------------------------------|----------------------------|--------------------------------|---------------------------------------------------|
| =      | Resolution | Designates exactly one variant; the variation becomes a classic dependency     | Yes                        | nb_iterations=10               | case_1:nb_iterations=10 (a single case)           |
| !      | Exclusion  | Removes variants from the set of candidates                                    | No                         | nb_iterations!10               | case_1:nb_iterations=20, case_1:nb_iterations=30  |
| +      | Inclusion  | Selects variants among the available ones                                      | No                         | nb_iterations+20+30            | case_1:nb_iterations=20, case_1:nb_iterations=30  |

## Schema of a use case name

The name of a use case can be complete or incomplete.

A complete name makes it possible to designate one and only one list of configuration blocks. This is necessary
for the requesting programs which can only launch a single use case at a time. A complete name implies that all
the variations of the launch path are totally resolved (so a single use case), whether via the case name and/or
directly via the ARCT file.

An incomplete name makes it possible to designate several lists of configuration blocks. This is useful for the
requesting programs which can process several use cases at once. An incomplete name implies that there is at
least one variation of the launch path which is not totally resolved, which therefore implies several use cases
(one per variant).

To build these names, we use the same symbols as in the ARCT file: `:`, `~`, `=`, `!` and `+`.

The `:` symbol separates the different elements of the case name. It must be used only to separate the different
elements making up the case name. It must not be put at the beginning or at the end of the case name (if there
is no base case to run, only variations to resolve, no `:` at the beginning).

The `=` symbol makes it possible to resolve a variation.

The `!` and `+` symbols restrict the names of variants to process. With these symbols, it is impossible to
totally resolve a variation. These symbols can therefore be used only in the case of incomplete names.

The variations resolved in the ARCT file are considered as classic dependencies. So, if a resolved variation
reappears in the use case name, an error will be emitted by the ARCT file reader.

If a variation is resolved, totally or partially, in the use case name while this variation does not appear in
the final block list, an error will be emitted by the ARCT file reader.

There cannot be two resolutions (partial or total) for the same variation in a case name.

### Order of the elements in the name

A case name must always start with the name of the base case (of the `cases` configuration block) unless there
is no base case to run (the `general` block can have variation dependencies).

Then, there can be variation resolutions (complete or partial). The order of these resolutions is not defined.

A resolution must have the form: `<variation><symbol><variant>[<symbol><variant>...]`.

A case name should look like this:
`[[<base_case>][~<alternative_name>][:<variation><symbol><variant>[<symbol><variant> ...]][: ...]]`.

Examples of valid names:

- `case_1:variation_1=variant_1`.
- `case_1:variation_1=variant_1:variation_2+variant_2+variant_3`.
- `case_1~init:variation_1=variant_1:variation_2+variant_2+variant_3`.
- `~init:variation_1=variant_1:variation_2+variant_2+variant_3`.
- `~init`.
- `variation_1=variant_1`.

Examples of invalid names:

- `case_1:variation_1=variant_1:`.
- `:case_1:variation_1=variant_1:`.
- `variation_1=variant_1:case_1`.
- `variation_1=variant_1:case_1~init`.
- `variation_1=variant_1:~init`.

Note: spaces in the name are not supported.

### Example

For the requesting programs requiring a complete name, it is possible to specify in the command line the name of
the resolved case(s) with the `=` symbol (the other symbols not being supported here):

`./program_1 -j file.arct -c case_1:variation_1=variant_1:variation_2=variant_2`

```txt
{ [general] } -> { [case_1] -> variation_1=variant_1 -> variation_2=variant_2 }
```

For the requesting programs able to support an incomplete name, it is possible to use the other symbols in the
command line:

`./program_1 -j file.arct -c case_1:variation_2!variant_2`

```txt
{ [general] } -> { [case_1] -> variation_1=variant_1 -> variation_2=variant_1 }
{ [general] } -> { [case_1] -> variation_1=variant_2 -> variation_2=variant_1 }
{ [general] } -> { [case_1] -> variation_1=variant_3 -> variation_2=variant_1 }
```

Note: it is up to the requesting program to tell the ARCT file reader whether it supports incomplete names or
not. The reader will analyze the file and the case name and will return an error if there is an incompatibility.
