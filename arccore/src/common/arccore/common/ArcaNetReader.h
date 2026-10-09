// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaNetReader.h                                             (C) 2000-2026 */
/*                                                                           */
/* Reader for ArcaNet files.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARCANETREADER_H
#define ARCCORE_COMMON_ARCANETREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/StringBuilder.h"

#include "arccore/common/Array.h"
#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Example of ArcaNet file (RC2) :
{
  "versions": {
    "_": 0,
    "arcane": 0,
    "arccher": 0
  },

  // Coucou

  "general": {
    "_": {
      "name": "Common"
    },
    "arcane": {
      "dataset": "./dataset.arc",
      "options": {
        "//meshes/mesh/filename": "aaa.msh",
        "T": 4
      }
    },
    "arccher": {
      "mpi": 4
    }
  },

  "commons": {
    "4procs": {
      "_": {
        "name": "4 Procs"
      },
      "arccher": {
        "mpi": 4
      }
    },
    "4threads": {
      "_": {
        "name": "4 Threads"
      },
      "arcane": {
        "options": {
          "S": 4
        }
      }
    },
    "16mpithreads": {
      "_": {
        "name": "16 Hybrid",
        "depend_a": ["4procs", "4threads"]
      }
    }
  },

  "variations": {
    "nb_iterations": {
      "10": {
        "_": {
          "name": "10 itérations"
        },
        "arcane": {
          "options": {
            "MaxIteration": 10
          }
        }
      },
      "20": {
        "_": {
          "name": "20 itérations"
        },
        "arcane": {
          "options": {
            "MaxIteration": 20
          }
        }
      },
      "30": {
        "_": {
          "name": "30 itérations"
        },
        "arcane": {
          "options": {
            "MaxIteration": 30
          }
        }
      }
    }
  },

  "cases": {
    // Reserved symbol for cases name : ":", "=", "~"

    // How to call "case1" :
    // - "case1" -> ok
    "case1": {
      "_": {
        "name": "Cas 1",
        // Reserved symbol for depend_a/depend_b elements : "=", "!", "+"
        "depend_a": ["4procs", "nb_iterations=10"]
      },

      // Reserved symbol for prog name : ":", "=", "~"
      "arcane": {
        "options": {
          "//meshes/mesh/filename": "aaa.msh",
          "T": 4
        }
      },

      "arccher": {
        "mpi": 2
      }
    },

    // How to call "case2" :
    // - "case2" -> error
    // - "case2:nb_iterations=10" -> ok
    // - "case2:nb_iterations=20" -> error
    // - "case2~init:nb_iterations=10" -> ok
    // - "case2~init:nb_iterations=20" -> error
    "case2": {
      "_": {
        "name": "Cas 2",
        "depend_b": ["16mpithreads"],
        "depend_a": ["nb_iterations!20"]
      },
      "arcane": {
        "options": {
          "//meshes/mesh/filename": "bbb.msh",
          "T": 8
        }
      },
      "arcane~init": {
        "options": {
          "//meshes/mesh/filename": "bbb.msh",
          "T": 16
        }
      }
    },
    // How to call "case3" :
    // - "case3" -> error
    // - "case3:nb_iterations=10" -> ok
    // - "case3:nb_iterations=20" -> ok
    // - "case3:nb_iterations=30" -> error
    "case3": {
      "_": {
        "name": "Cas 3",
        "depend_a": ["4procs", "nb_iterations+10+20"]
      },

      "arcane": {
        "options": {
          "//meshes/mesh/filename": "ccc.msh",
          "T": 4
        }
      },

      "arccher": {
        "mpi": 2
      }
    },
  }
}
 */

/*!
 * \brief Class allowing to read ArcaNet files.
 */
class ARCCORE_COMMON_EXPORT ArcaNetReader
{
 public:

  /*!
   * \brief Constructor.
   * \param root root element of ArcaNet file.
   */
  ArcaNetReader(const JSONValue& root);

 public:

  /*!
   * \brief Method allowing to initialise the file reader.
   *
   * \param case_and_variations_values The case name with variations values
   * (example : "case1~init:nb_proc=4").
   * \param prog_name Program name to read in file (example : "arcane').
   */
  void init(const String& case_and_variations_values, const String& prog_name);

  /*!
   * \brief Method allowing to read the file and fill the chain of dependencies.
   * First, you need to call 'init()' method.
   */
  void read();

  /*!
   * \brief Method allowing to get all JSONValue to read, following the chain
   * of dependencies.
   */
  ArrayView<JSONValue> jsonValues() { return m_values; }

  /*!
   * \brief Method allowing to get the file part of a JSONValue (from
   * 'jsonValues()' array).
   *
   * \param config_part An element of 'jsonValues()' array.
   * \return The file part.
   */
  JSONValue extractFilePart(const JSONValue& config_part);

  /*!
   * \brief Method allowing to get the program part of a JSONValue (from
   * 'jsonValues()' array).
   *
   * \param config_part An element of 'jsonValues()' array.
   * \return The program part.
   */
  JSONValue extractProgPart(const JSONValue& config_part);

  /*!
   * \brief Method allowing to get a part of a JSONValue (from
   * 'jsonValues()' array).
   *
   * \param part_name The name of the part to get.
   * \param config_part An element of 'jsonValues()' array.
   * \return The json part.
   */
  JSONValue extractPart(const String& part_name, const JSONValue& config_part);

  /*!
   * \brief Method allowing to determining if the path is recorded.
   * To enable the recording, you can set ARCCORE_ARCANET_PATH environment
   * variable.
   */
  bool computePath() const { return m_build_path; }

  /*!
   * \brief Method allowing to get the dependencies path.
   */
  String path() const { return m_path; }

  /*!
   * \brief Method allowing to get the final name of program part.
   */
  String configPartName() { return m_prog_custom_part_name.empty() ? m_prog_part_name : m_prog_custom_part_name; }

 private:

  void _readDependPart(const JSONValue& depend_part);
  void _readConfigPart(const JSONValue& config_part, const String& node_name);
  void _addElemInPath(const String& elem);
  void _readFileBeforePart(const JSONValue& config_part);
  void _readFileAfterPart(const JSONValue& config_part);
  void _readArcanePart(const JSONValue& config_part);

 private:

  bool m_build_path = false;
  StringBuilder m_path;
  Integer m_level_path = 0;
  const JSONValue& m_root;
  UniqueArray<String> m_explored;
  UniqueArray<String> m_variations_key_resolved;
  UniqueArray<String> m_variations_value_resolved;
  String m_prog_part_name;
  String m_prog_custom_part_name;
  UniqueArray<JSONValue> m_values;
  String m_case_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
