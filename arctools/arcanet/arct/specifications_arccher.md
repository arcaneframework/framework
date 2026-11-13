{
  "versions": {
    "_": 0,
    "arccher": 0
  },
  "general": {
    "arccher": {
      "code_binary": "/home/aaa/debug/template/HelloWorld",
      "working_dir": "/home/aaa/debug",
      "executors": {
        "none": {
          "cmdline": "{_CODE}"
        },
        "openmpi": {
          "cmdline": "{executor_binary} {nb_proc} {_CODE}",
          "cmdopt": {
            "executor_binary": {
              "opt": "{}",
              "optional": false,
              "default": "/usr/bin/mpiexec"
            },
            "nb_proc": {
              "opt": "-n {}",
              "optional": false,
              "default": "1"
            }
          }
        }
      },
      "codes": {
        "arcane": {
          "cmdline": "{_EXECUTOR} {code_binary} {_ARCT_PATH} {_CASE_NAME}",
          "cmdopt": {
            "code_binary": {
              "opt": "{}",
              "optional": false
            },
            "_ARCT_PATH": {
              "opt": "-A,ParamFile={}",
              "optional": false
            },
            "_CASE_NAME": {
              "opt": "-A,Case={}",
              "optional": true
            },
            "_EXECUTOR": {
              "optional": true
            }
          }
        }
      }
    }
  },
  "cases": {
    "mon_test_1": {
      "_": {
        "depend_b": [
          "small_test"
        ],
        "name": "Mon Test 1"
      },
      "arcane": {
        "dataset": "truc.arc"
      },
      "arccher": {
        "executor_name": "openmpi",
        "executor_binary": "/bin/mpiexec",
        "run": [
          "arcane",
          "arcane~init"
        ],
        "arcane": {
          "code_binary": "bin_path",
          "env_var": {
            "VARIABLE": "VAL"
          }
        },
        "arcane~init": {
        }
      }
    },
    "mon_test_2": {
      "arcane": {
        "dataset": "truc.arc"
      },
      "arccher": {
        "run": ["arcane"],
        "working_dir": "/home/aaa/debug",
        "executor_name": "openmpi",
        "executor_binary": "/bin/mpiexec",
        "code_binary": "/home/aaa/debug/template/HelloWorld",
        "env_var": {
          "VARIABLE": "VAL"
        }
      }
    }
  }
}
