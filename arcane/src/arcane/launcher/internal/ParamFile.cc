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

  void read(const String& case_and_variations_values);
  bool displayDetails() const { return m_build_path; }
  String path() const { return m_path; }
  String configPartName() { return m_arcane_custom_part_name.empty() ? m_arcane_part_name : m_arcane_custom_part_name; }

 private:

  void _readDependPart(const JSONValue& depend_part);
  void _readConfigPart(const JSONValue& config_part, const String& node_name);
  void _addElemInPath(const String& elem);
  void _readFileBeforePart(const JSONValue& config_part);
  void _readFileAfterPart(const JSONValue& config_part);
  void _readArcanePart(const JSONValue& config_part);

 private:

  // StringBuilder m_name;
  // bool m_is_name_empty = true;
  bool m_build_path = false;
  StringBuilder m_path;
  Integer m_level_path = 0;
  CommandLineArguments& m_cargs;
  const JSONValue& m_root;
  UniqueArray<String> m_explored;
  UniqueArray<String> m_variations_key_resolved;
  UniqueArray<String> m_variations_value_resolved;
  const String m_arcane_part_name = "arcane";
  String m_arcane_custom_part_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParamFile::Reader::Reader(CommandLineArguments& cargs, const JSONValue& root)
: m_build_path(!platform::getEnvironmentVariable("ARCANE_ARCANET_PATH").null())
, m_cargs(cargs)
, m_root(root)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
read(const String& case_and_variations_values)
{
  String case_name;
  if (!case_and_variations_values.empty()) {
    UniqueArray<String> split_case;
    case_and_variations_values.split(split_case, ':');

    for (const String& elem : split_case) {
      if (elem.contains("=")) {
        UniqueArray<String> split_variation;
        elem.split(split_variation, '=');
        if (split_variation.size() != 2) {
          ARCANE_FATAL("Bad element in Case option.");
        }
        m_variations_key_resolved.add(split_variation[0]);
        m_variations_value_resolved.add(split_variation[1]);
        // std::cout << "Add key=value : " << split_elem[0] << " = " << split_elem[1] << std::endl;
      }
      else {
        if (!case_name.empty()) {
          ARCANE_FATAL("You must choose a variation for '{0}'.", elem);
        }
        if (elem.contains("~")) {
          UniqueArray<String> split_case_name_with_prog_subname;
          elem.split(split_case_name_with_prog_subname, '~');
          if (split_case_name_with_prog_subname.size() != 2) {
            ARCANE_FATAL("Bad element in Case option.");
          }
          case_name = split_case_name_with_prog_subname[0];
          m_arcane_custom_part_name = m_arcane_part_name + "~" + split_case_name_with_prog_subname[1];
        }
        else {
          case_name = elem;
        }
      }
    }
  }

  // "general" part
  {
    const JSONValue cases = m_root.child("general");
    if (!cases.isNull()) {
      // std::cout << "General part" << std::endl;
      m_explored.add("general");
      _readConfigPart(cases, "general");
    }
  }
  if (!case_name.empty()) {
    const JSONValue cases = m_root.child("cases").child(case_name);
    if (cases.null()) {
      ARCANE_FATAL("Case '{0}' not found.", case_name);
    }
    // std::cout << "Case part : " << case_clean << std::endl;
    m_explored.add(case_name);
    _readConfigPart(cases, case_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
_readDependPart(const JSONValue& depend_part)
{
  if (depend_part.null())
    return;

  const JSONValueList array = depend_part.valueAsArray();
  for (const JSONValue elem : array) {
    JSONValue dependence_node;
    String dependence_name = elem.valueAsStringView();
    {
      if (dependence_name.contains("=")) {
        UniqueArray<String> split_variation;
        dependence_name.split(split_variation, '=');
        if (split_variation.size() != 2) {
          ARCANE_FATAL("Element '{0}' is invalid.", dependence_name);
        }

        const JSONValue variation = m_root.child("variations").child(split_variation[0]).child(split_variation[1]);
        if (variation.null()) {
          ARCANE_FATAL("Element '//variations/{0}/{1}' not found.", split_variation[0], split_variation[1]);
        }
        if (m_explored.contains(dependence_name)) {
          ARCANE_FATAL("Element '//variations/{0}/{1}' already explored.", split_variation[0], split_variation[1]);
        }
        m_explored.add(dependence_name);
      }
      else {
        dependence_node = m_root.child("commons").child(dependence_name);
        if (!dependence_node.null()) {
          if (m_explored.contains(dependence_name)) {
            ARCANE_FATAL("Element '//commons/{0}' already explored.", dependence_name);
          }
          m_explored.add(dependence_name);
        }
        else {
          String cleaned_dependence_name;
          UniqueArray<String> split_dependence_name;
          if (dependence_name.contains("!")) {
            dependence_name.split(split_dependence_name, '!');
            cleaned_dependence_name = split_dependence_name[0];
          }
          else {
            cleaned_dependence_name = dependence_name;
          }
          auto pos_elem = m_variations_key_resolved.span().findFirst(cleaned_dependence_name);

          if (!pos_elem.has_value()) {
            const JSONValue error_ = m_root.child("variations").child(cleaned_dependence_name);
            if (error_.null()) {
              ARCANE_FATAL("Element '//commons/{0}' not found.", cleaned_dependence_name);
            }
            ARCANE_FATAL("Element '{0}' not found in 'Case' cmd line option.", cleaned_dependence_name);
          }

          String value = m_variations_value_resolved[pos_elem.value()];
          if (!split_dependence_name.empty()) {
            ArrayView excluded_values(split_dependence_name.subView(1, split_dependence_name.size() - 1));
            if (excluded_values.contains(value)) {
              ARCANE_FATAL("Element '//variations/{0}/{1}' is excluded.", cleaned_dependence_name, value);
            }
          }
          dependence_name = cleaned_dependence_name + "=" + value;

          dependence_node = m_root.child("variations").child(cleaned_dependence_name).child(value);
          if (dependence_node.null()) {
            ARCANE_FATAL("Element '//variations/{0}/{1}' not found.", cleaned_dependence_name, value);
          }
          if (m_explored.contains(dependence_name)) {
            ARCANE_FATAL("Element '//variations/{0}/{1}' already explored.", cleaned_dependence_name, value);
          }
          m_explored.add(dependence_name);
        }
      }
    }
    _readConfigPart(dependence_node, dependence_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
_readConfigPart(const JSONValue& config_part, const String& node_name)
{
  if (m_build_path) {
    _addElemInPath("{");
    m_level_path++;
    _readFileBeforePart(config_part);
    _addElemInPath(node_name);
    _readArcanePart(config_part);
    _readFileAfterPart(config_part);
    m_level_path--;
    _addElemInPath("}");
  }
  else {
    _readFileBeforePart(config_part);
    _readArcanePart(config_part);
    _readFileAfterPart(config_part);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
_addElemInPath(const String& elem)
{
  m_path += "\n";
  for (Integer i = 0; i < m_level_path; ++i)
    m_path += "  ";
  m_path += elem;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
_readFileBeforePart(const JSONValue& config_part)
{
  const JSONValue depend_before_var = config_part.child("_").child("depend_b");
  _readDependPart(depend_before_var);

  // const JSONValue name_var = config_part.child("_").child("name");
  // if (!name_var.isNull()) {
  //   if (m_is_name_empty)
  //     m_is_name_empty = false;
  //   else
  //     m_name += " -> ";
  //
  //   m_name += "\"";
  //   m_name += name_var.valueAsStringView();
  //   m_name += "\"";
  //
  //   std::cout << "Config part: " << name_var.valueAsStringView() << std::endl;
  // }
  // else {
  //   std::cout << "Config part" << std::endl;
  // }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
_readFileAfterPart(const JSONValue& config_part)
{
  const JSONValue depend_after_var = config_part.child("_").child("depend_a");
  _readDependPart(depend_after_var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
_readArcanePart(const JSONValue& config_part)
{
  JSONValue arcane_part;
  if (!m_arcane_custom_part_name.empty()) {
    arcane_part = config_part.child(m_arcane_custom_part_name);
  }
  if (arcane_part.null()) {
    arcane_part = config_part.child(m_arcane_part_name);
  }
  else if (m_build_path) {
    m_path += "(";
    m_path += m_arcane_custom_part_name;
    m_path += ")";
    // m_name += "(";
    // m_name += m_arcane_custom_part_name;
    // m_name += ")";
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

  if (reader.displayDetails()) {
    std::cout << "ArcaNet path: " << reader.path() << std::endl;
    std::cout << "ArcaNet config part readed: " << reader.configPartName() << std::endl;
  }

  // StringList names;
  // StringList values;
  // cargs.fillParameters(names, values);
  // for (Integer i = 0, n = names.count(); i < n; ++i) {
  //   std::cout << "Final Elem : " << names[i] << " -- val : " << values[i] << std::endl;
  // }

  ArcaneLauncher::applicationInfo().setCommandLineArguments(cargs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
