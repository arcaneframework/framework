# Example: restricted variations (3) {#arcanetdoc_arct_variations3}

[TOC]

File `example.arct`:

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  },

  "general": {
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc"
    }
  },

  "cases": {
    "algo1": {
      "_": {
        "name": "Inter-boundingbox algorithm",
        "depend_b": ["nb_thread!1"]
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "false"
        }
      }
    },
    "algo2": {
      "_": {
        "name": "Intra-boundingbox algorithm",
        "depend_b": ["enable_post_processing", "nb_thread+1"]
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "true"
        }
      }
    }
  },

  "commons": {
    "enable_post_processing": {
      "_": {
        "name": "Enabling the VtkHdf output"
      },
      "arcane": {
        "options": {
          "//nonreg/post-processor-write": "true",
          "//nonreg/mesh-contact-param[]/div-nb-cell": "1"
        }
      }
    }
  },

  "variations": {
    "nb_thread": {
      "1": {
        "_": {
          "name": "1 thread"
        }
      },
      "2": {
        "_": {
          "name": "2 threads"
        },
        "arcane": {
          "options": {
            "S": "2"
          }
        }
      },
      "4": {
        "_": {
          "name": "4 threads"
        },
        "arcane": {
          "options": {
            "S": "4"
          }
        }
      }
    }
  }
}
```

It is possible that one or more variants of a variation are not compatible with a block.

We can then restrict the available variants.

We added two limiters in the example: `!` in algo 1 and `+` in algo 2.

The `!` limiter excludes one or more variants. In algo 1, we have `nb_thread!1`, so it will be impossible
to choose the `nb_thread=1` variant in the case name.

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1:nb_thread=1


Error
```

For a multi-case program, it would get two lists of blocks:

```shell
./PasArcane --arct=example.arct --case=algo1


{ [general] } -> { nb_thread=2 -> [algo1] }
{ [general] } -> { nb_thread=4 -> [algo1] }
```

It is possible to put several inclusions or exclusions in a row, for example: `nb_thread!1!2`. However, it
is impossible to mix the two limiters for the same variation (it would not make sense), for example:
`nb_thread+1+2!2`.

## Variation resolution

The only way to resolve a variation is to use the resolution symbol `=`. This is a deliberate limitation
which forces the simplification of ARCT files and use case names.

If we take the example above, if we want to use only `nb_thread=1`, without this rule, we could resolve
the variation in three ways:

1. `nb_thread=1`
2. `nb_thread+1`
3. `nb_thread!2!4`

*3.* is less easily readable than *1.*. The `+` of *2.* is supposed to be used with several inclusions.

\note `nb_thread=1` and `nb_thread+1` are, on paper, identical. But the philosophy behind is different.
`nb_thread=1` does not leave a choice to the user, it is considered as a classic `commons` dependency.
`nb_thread+1` must leave a choice to the user (even if it is a single choice).

### Variation resolution for a single-case program

A single-case program requires all the variations to be resolved. So the `!` and `+` symbols cannot be
used in a use case name:

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1:nb_thread+2


Error
```

For algo 2, even if we have `nb_thread+1`, it is necessary to resolve the variation in the command line:

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2:nb_thread=1


{ [general] } -> { enable_post_processing -> nb_thread=1 -> [algo2] }
```

Otherwise:

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2


Error
```

If we do not want the user to have to resolve the variation, we must put `nb_thread=1` in the ARCT file
instead of `nb_thread+1`.

\note Putting `nb_thread+1` in the ARCT file is allowed and can even be useful if we want to force the
user to make a resolution in the command line. The use case is presented in the CMake part `arct ->
ctest`. In this part, the name of a test is the name of the use case executed by CTest. If we have a
sequential-only case, it can be useful to display `test1:nb_proc=1` instead of simply displaying `test1`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_variations2
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations4
</span>
</div>
