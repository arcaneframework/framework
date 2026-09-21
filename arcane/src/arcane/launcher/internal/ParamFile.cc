// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParamFile.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Reader of parameter files.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/internal/ParamFile.h"

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/JSONReader.h"
#include "arccore/common/List.h"

#include <iostream>

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
        // Reserved symbol for depend_a/depend_b elements : "=", "!"
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
    }
  }
}
 */

class ParamFile::Reader
{
 public:

  Reader(CommandLineArguments& cargs, const JSONValue& root);

 public:

  void read(const String& case_to_read);
  void readDependPart(const JSONValue& depend_part);
  void readCommonPart(const JSONValue& common_part);
  void readFileBeforePart(const JSONValue& config_part);
  void readFileAfterPart(const JSONValue& config_part);
  void readArcanePart(const JSONValue& config_part);

 public:

  StringBuilder m_name;
  bool m_is_name_empty = true;
  CommandLineArguments& m_cargs;
  const JSONValue& m_root;
  UniqueArray<String> m_explored;
  UniqueArray<String> m_variations_key_resolved;
  UniqueArray<String> m_variations_value_resolved;
  String m_arcane_part_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParamFile::Reader::Reader(CommandLineArguments& cargs, const JSONValue& root)
: m_cargs(cargs)
, m_root(root)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
read(const String& case_to_read)
{
  String case_clean;
  if (!case_to_read.empty()) {
    UniqueArray<String> split_case;
    case_to_read.split(split_case, ':');

    for (const String& elem : split_case) {
      if (elem.contains("=")) {
        UniqueArray<String> split_elem;
        elem.split(split_elem, '=');
        if (split_elem.size() != 2) {
          ARCANE_FATAL("Bad element in Case option.");
        }
        m_variations_key_resolved.add(split_elem[0]);
        m_variations_value_resolved.add(split_elem[1]);
        std::cout << "Add key=value : " << split_elem[0] << " = " << split_elem[1] << std::endl;
      }
      else {
        if (!case_clean.empty()) {
          ARCANE_FATAL("You must choose a variation for '{0}'.", elem);
        }
        if (elem.contains("~")) {
          UniqueArray<String> split_elem;
          elem.split(split_elem, '~');
          if (split_elem.size() != 2) {
            ARCANE_FATAL("Bad element in Case option.");
          }
          case_clean = split_elem[0];
          m_arcane_part_name = "arcane~" + split_elem[1];
        }
        else {
          case_clean = elem;
        }
      }
    }
  }

  // "general" part
  {
    const JSONValue cases = m_root.child("general");
    if (!cases.isNull()) {
      std::cout << "General part" << std::endl;

      readFileBeforePart(cases);
      readArcanePart(cases);
      readFileAfterPart(cases);
    }
  }
  if (!case_clean.empty()) {

    const JSONValue cases = m_root.child("cases").child(case_clean);
    if (cases.null()) {
      ARCANE_FATAL("Case '{0}' not found.", case_clean);
    }

    std::cout << "Case part : " << case_clean << std::endl;

    readFileBeforePart(cases);
    readArcanePart(cases);
    readFileAfterPart(cases);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
readDependPart(const JSONValue& depend_part)
{
  if (depend_part.null())
    return;

  const JSONValueList array = depend_part.valueAsArray();
  for (const JSONValue elem_value : array) {
    String elem = elem_value.valueAsStringView();
    if (elem.contains("=")) {
      UniqueArray<String> split_elem;
      elem.split(split_elem, '=');
      if (split_elem.size() != 2) {
        ARCANE_FATAL("Element '{0}' is invalid.", elem);
      }

      const JSONValue variation = m_root.child("variations").child(split_elem[0]).child(split_elem[1]);
      if (variation.null()) {
        ARCANE_FATAL("Element '//variations/{0}/{1}' not found.", split_elem[0], split_elem[1]);
      }
      if (m_explored.contains(elem)) {
        ARCANE_FATAL("Element '//variations/{0}/{1}' already explored.", split_elem[0], split_elem[1]);
      }
      m_explored.add(elem);
      readCommonPart(variation);
    }
    else {
      const JSONValue common = m_root.child("commons").child(elem);
      if (!common.null()) {
        if (m_explored.contains(elem)) {
          ARCANE_FATAL("Element '//commons/{0}' already explored.", elem);
        }
        m_explored.add(elem);
        readCommonPart(common);
      }
      else {
        String name_depend;
        UniqueArray<String> split_depend_name;
        if (elem.contains("!")) {
          elem.split(split_depend_name, '!');
          name_depend = split_depend_name[0];
        }
        else {
          name_depend = elem;
        }
        auto pos_elem = m_variations_key_resolved.span().findFirst(name_depend);

        if (!pos_elem.has_value()) {
          const JSONValue error_ = m_root.child("variations").child(name_depend);
          if (error_.null()) {
            ARCANE_FATAL("Element '//commons/{0}' not found.", name_depend);
          }
          ARCANE_FATAL("Element '{0}' not found in 'Case' cmd line option.", name_depend);
        }

        String value = m_variations_value_resolved[pos_elem.value()];
        if (!split_depend_name.empty())
        {
          ArrayView excluded_values(split_depend_name.subView(1, split_depend_name.size()-1));
          if (excluded_values.contains(value)) {
            ARCANE_FATAL("Element '//variations/{0}/{1}' is excluded.", name_depend, value);
          }
        }
        String key_value = name_depend + "=" + value;

        const JSONValue variation = m_root.child("variations").child(name_depend).child(value);
        if (variation.null()) {
          ARCANE_FATAL("Element '//variations/{0}/{1}' not found.", name_depend, value);
        }
        if (m_explored.contains(key_value)) {
          ARCANE_FATAL("Element '//variations/{0}/{1}' already explored.", name_depend, value);
        }
        m_explored.add(key_value);
        readCommonPart(variation);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
readCommonPart(const JSONValue& common_part)
{
  std::cout << "Common part" << std::endl;

  readFileBeforePart(common_part);
  readArcanePart(common_part);
  readFileAfterPart(common_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
readFileBeforePart(const JSONValue& config_part)
{
  const JSONValue file_part = config_part.child("_");
  if (file_part.null())
    return;

  {
    const JSONValue depend_before_var = file_part.child("depend_b");
    readDependPart(depend_before_var);
  }

  const JSONValue name_var = file_part.child("name");
  if (!name_var.isNull()) {
    if (m_is_name_empty)
      m_is_name_empty = false;
    else
      m_name += ".";

    m_name += name_var.valueAsStringView();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
readFileAfterPart(const JSONValue& config_part)
{
  const JSONValue file_part = config_part.child("_");
  if (file_part.null())
    return;

  {
    const JSONValue depend_after_var = file_part.child("depend_a");
    readDependPart(depend_after_var);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
readArcanePart(const JSONValue& config_part)
{
  JSONValue arcane_part;
  if (!m_arcane_part_name.empty()) {
    arcane_part = config_part.child(m_arcane_part_name);
  }
  if (arcane_part.null()) {
    arcane_part = config_part.child("arcane");
  }

  const JSONValue dataset = arcane_part.child("dataset");
  if (!dataset.isNull()) {
    m_cargs.addParameterLine(String::format("CaseDatasetFileName={0}", dataset.value()));
    //std::cout << "Dataset : " << dataset.value() << std::endl;
  }

  const JSONValue arcane_params = arcane_part.child("options");

  if (!arcane_params.isNull()) {
    const JSONKeyValueList params = arcane_params.keyValueChildren();

    for (auto elem : params) {
      m_cargs.addParameterLine(String::format("{0}={1}", elem.name(), elem.value().valueAsStringView()));

      // std::cout << "Elem : " << elem.name() << " -- val : " << elem.value().valueAsStringView() << std::endl;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::
editParams(const String& param_file_name, const String& variation)
{
  CommandLineArguments cargs = ArcaneLauncher::applicationInfo().commandLineArguments();

  JSONDocument json_doc;
  {
    UniqueArray<Byte> bytes;

    if (platform::readAllFile(param_file_name, false, bytes)) {
      ARCANE_FATAL("Param file not available");
    }

    json_doc.parse(bytes, param_file_name, (JSONDocument::ParseCommentsFlag | JSONDocument::ParseNumbersAsStringsFlag));
  }

  const JSONValue root = json_doc.root();
  Reader reader(cargs, root);

  reader.read(variation);

  std::cout << "Name variation : " << reader.m_name << std::endl;

  StringList names;
  StringList values;
  cargs.fillParameters(names, values);
  for (Integer i = 0, n = names.count(); i < n; ++i) {
    std::cout << "Final Elem : " << names[i] << " -- val : " << values[i] << std::endl;
  }

  ArcaneLauncher::applicationInfo().setCommandLineArguments(cargs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
