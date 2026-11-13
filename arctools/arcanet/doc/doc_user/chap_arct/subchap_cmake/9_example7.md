# Example: variations of variations (4) {#arcanetdoc_arct_cmake_example7}

[TOC]

File `testlist.arct`:

```json
{
  "versions": {
    "_": 1,
    "arcane": 1,
    "arccher": 1
  },

  "general": {
    "_": {
      "depend_b": ["enable_post_processing"]
    },
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc"
    }
  },

  "cases": {
    "algo1": {
      "_": {
        "name": "Inter-boundingbox algorithm",
        "depend_b": ["test"]
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
        "depend_b": ["test=sequential"]
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
        },
        "arcane": {
          "options": {
            "S": "1"
          }
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
    },
    "nb_proc": {
      "1": {
        "_": {
          "name": "1 proc"
        },
        "arccher": {
          "nb_proc": 1
        }
      },
      "2": {
        "_": {
          "name": "2 procs"
        },
        "arccher": {
          "nb_proc": 2
        }
      },
      "4": {
        "_": {
          "name": "4 procs"
        },
        "arccher": {
          "nb_proc": 4
        }
      }
    },
    "nb_hybrid": {
      "4": {
        "_": {
          "name": "2 procs + 2 threads",
          "depend_b": ["nb_proc=2", "nb_thread=2"]
        }
      },
      "16": {
        "_": {
          "name": "4 procs + 4 threads",
          "depend_b": ["nb_proc=4", "nb_thread=4"]
        }
      }
    },
    "test": {
      "sequential": {
        "_": {
          "name": "Sequential"
        }
      },
      "mpi": {
        "_": {
          "name": "MPI",
          "depend_b": ["nb_proc!1"]
        }
      },
      "thread": {
        "_": {
          "name": "MT",
          "depend_b": ["nb_thread!1"]
        }
      },
      "hybrid": {
        "_": {
          "name": "Hybrid",
          "depend_b": ["nb_hybrid"]
        }
      }
    }
  }
}
```

File `tests.cmake`:

```cmake
arct_config_create_or_edit_general(OVERWRITE_ARCT)
arct_arcane_dataset(PATH "config_2D_diagonal_circle.arc")
arct_config_dependency(ADD "enable_post_processing")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo1")
arct_config_define_name(NAME "Inter-boundingbox algorithm")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "false")
arct_config_variation(ADD "test")
arct_config_create_or_edit_end()

arct_config_create_or_edit_case(NAME_CASE "algo2")
arct_config_define_name(NAME "Intra-boundingbox algorithm")
arct_arcane_option(ADD "//nonreg/contact-pairs[]/use-contact-intra" VALUE "true")
arct_config_variation(ADD "test" EQUAL "sequential")
arct_config_create_or_edit_end()

arct_config_create_or_edit_common(NAME_COMMON "enable_post_processing")
arct_config_define_name(NAME "Enabling the VtkHdf output")
arct_arcane_option(ADD "//nonreg/post-processor-write" VALUE "true")
arct_arcane_option(ADD "//nonreg/mesh-contact-param[]/div-nb-cell" VALUE "1")
arct_config_create_or_edit_end()


macro(generate_para)
  foreach(i 1 2 4)
    arct_config_create_or_edit_variation(NAME_VARIATION "nb_thread" NAME_VARIANT "${i}")
    arct_config_define_name(NAME "${i} threads")
    arct_arcane_option(ADD "S" VALUE "${i}")
    arct_config_create_or_edit_end()
  
    arct_config_create_or_edit_variation(NAME_VARIATION "nb_proc" NAME_VARIANT "${i}")
    arct_config_define_name(NAME "${i} procs")
    arct_arccher_mpi(NB_PROC "${i}")
    arct_config_create_or_edit_end()
  endforeach()
  
  
  
  arct_config_create_or_edit_variation(NAME_VARIATION "nb_hybrid" NAME_VARIANT "4")
  arct_config_define_name(NAME "2 procs + 2 threads")
  arct_config_variation(ADD "nb_proc" EQUAL "2")
  arct_config_variation(ADD "nb_thread" EQUAL "2")
  arct_config_create_or_edit_end()
  
  arct_config_create_or_edit_variation(NAME_VARIATION "nb_hybrid" NAME_VARIANT "16")
  arct_config_define_name(NAME "4 procs + 4 threads")
  arct_config_variation(ADD "nb_proc" EQUAL "4")
  arct_config_variation(ADD "nb_thread" EQUAL "4")
  arct_config_create_or_edit_end()
  
  
  
  arct_config_create_or_edit_variation(NAME_VARIATION "test" NAME_VARIANT "sequential")
  arct_config_define_name(NAME "Sequential")
  arct_config_create_or_edit_end()
  
  arct_config_create_or_edit_variation(NAME_VARIATION "test" NAME_VARIANT "mpi")
  arct_config_define_name(NAME "MPI")
  arct_config_variation(ADD "nb_proc" EXCLUDE "1")
  arct_config_create_or_edit_end()
  
  arct_config_create_or_edit_variation(NAME_VARIATION "test" NAME_VARIANT "thread")
  arct_config_define_name(NAME "MT")
  arct_config_variation(ADD "nb_thread" EXCLUDE "1")
  arct_config_create_or_edit_end()
  
  arct_config_create_or_edit_variation(NAME_VARIATION "test" NAME_VARIANT "hybrid")
  arct_config_define_name(NAME "Hybrid")
  arct_config_variation(ADD "nb_hybrid")
  arct_config_create_or_edit_end()
endmacro()

generate_para()

arct_to_ctest()
```

Generated tests:

```shell
ctest -N

algo1:test=sequential
algo1:test=mpi:nb_proc=2
algo1:test=mpi:nb_proc=4
algo1:test=thread:nb_thread=2
algo1:test=thread:nb_thread=4
algo1:test=hybrid:nb_hybrid=4
algo1:test=hybrid:nb_hybrid=16
algo2:test=sequential
```

Let's revisit the ARCT example \ref arcanetdoc_arct_variations4.

Here, there are quite a few variations. We test pretty much all the parallelism combinations.

We can create a macro if we use several ARCT files, in order to avoid copying the same thing each time.

\remark The ARCT v1 format does not allow importing ARCT files into ARCT files.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cmake_example6
</span>
<!--<span class="next_section_button">
\ref arcanetdoc_arct_cmake_example7
</span>-->
</div>
