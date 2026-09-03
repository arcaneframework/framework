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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Example of parameter file (WiP) :
{
  "versions": {
    "file": 0,
    "arcane": 0,
    "run_opt": 0
  },

  // Coucou

  "general": {
    "file": {
      "name": "Common"
    },
    "arcane": {
      "dataset": "./dataset.arc",
      "options": {
        "//meshes/mesh/filename": "aaa.msh",
        "T": 4
      }
    },
    "run_opt": {
      "mpi": 4
    }
  },

  // Reserved for a next version
  "commons": {
  },
  "variations": {
  },

  "cases": {
    // Reserved symbol for names, for a next version : ":"
    "case1": {
      "file": {
        "name": "Cas 1"
      },

      "arcane": {
        "options": {
          "//meshes/mesh/filename": "aaa.msh",
          "T": 4
        }
      },

      "run_opt": {
        "mpi": 2
      }
    },
    "case2": {
      "file": {
        "name": "Cas 2"
      },

      "arcane": {
        "options": {
          "//meshes/mesh/filename": "bbb.msh",
          "T": 8
        }
      }
    }
  }
}
 */

class ParamFile::Reader
{
 public:

  void readFilePart(const JSONValue& file_part);
  static void readArcanePart(CommandLineArguments& cargs, const JSONValue& arcane_part);

 public:

  StringBuilder m_name;
  bool m_is_name_empty = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::Reader::
readFilePart(const JSONValue& file_part)
{
  JSONValue name_var = file_part.child("name");
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
readArcanePart(CommandLineArguments& cargs, const JSONValue& arcane_part)
{
  const JSONValue dataset = arcane_part.child("dataset");
  if (!dataset.isNull()) {
    cargs.addParameterLine(String::format("CaseDatasetFileName={0}", dataset.value()));
    //std::cout << "Dataset : " << dataset.value() << std::endl;
  }

  JSONValue arcane_params = arcane_part.child("options");

  if (!arcane_params.isNull()) {
    const JSONKeyValueList params = arcane_params.keyValueChildren();

    for (auto elem : params) {
      cargs.addParameterLine(String::format("{0}={1}", elem.name(), elem.value().valueAsStringView()));

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
  Reader reader;

  // "general" part
  {
    const JSONValue cases = root.child("general");
    if (!cases.isNull()) {
      // std::cout << "General part" << std::endl;

      reader.readFilePart(cases.child("file"));
      Reader::readArcanePart(cargs, cases.child("arcane"));
    }
  }

  if (!variation.empty()) {
    const JSONValue cases = root.child("cases").expectedChild(variation);
    // std::cout << "Variation part : " << variation << std::endl;

    reader.readFilePart(cases.child("file"));
    Reader::readArcanePart(cargs, cases.child("arcane"));
  }

  // std::cout << "Name variation : " << reader.m_name << std::endl;

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
