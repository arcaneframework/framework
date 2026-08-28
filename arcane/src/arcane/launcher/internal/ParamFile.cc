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

  "dataset": "./dataset.arc",
  // Coucou

  "common": {
    "name": "Common",
    "arcane": {
      "options": {
        "//meshes/mesh/filename": "aaa.msh",
        "T": 4
      }
    },
    "run_opt": {
      "mpi": 4
    }
  },

  "variations": {
    "variation1": {
      "name": "Variation1",
      "arcane": {
        "options": {
          "//meshes/mesh/filename": "aaa.msh",
          "T": 4
        }
      },
      "run_opt": {
        "mpi": 4
      }
    },
    "variation2": {
      "name": "Variation2",
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

void ParamFile::
editParams(const String& param_file_name, const String& variation)
{
  CommandLineArguments cargs = ArcaneLauncher::applicationInfo().commandLineArguments();

  UniqueArray<Byte> bytes;

  if (platform::readAllFile(param_file_name, false, bytes)) {
    ARCANE_FATAL("Param file not available");
  }

  JSONDocument json_doc;
  json_doc.parse<JSONDocument::ParseCommentsFlag | JSONDocument::ParseNumbersAsStringsFlag>(bytes, param_file_name);

  const JSONValue root = json_doc.root();

  {
    const JSONValue dataset = root.child("dataset");
    if (!dataset.isNull()) {
      cargs.addParameterLine(String::format("CaseDatasetFileName={0}", dataset.value()));
      // std::cout << "Dataset : " << dataset.value() << std::endl;
    }
  }

  StringBuilder name;

  {
    JSONValue variations = root.child("common");
    if (!variations.isNull()) {
      {
        JSONValue name_var = variations.child("name");
        if (!name_var.isNull()) {
          name += name_var.valueAsStringView();
          name += ".";
        }
      }

      JSONValue arcane_params = variations.child("arcane").child("options");
      if (!arcane_params.isNull()) {
        const JSONKeyValueList params = arcane_params.keyValueChildren();

        for (auto elem : params) {
          cargs.addParameterLine( String::format("{0}={1}", elem.name(), elem.value().valueAsStringView()));

          // std::cout << "Elem : " << elem.name() << " -- val : " << elem.value().valueAsStringView() << std::endl;
        }
      }
    }
  }

  if (!variation.empty()) {
    const JSONValue variations = root.child("variations");
    if (!variations.isNull()) {
      const JSONValue asked_var = variations.expectedChild(variation);

      {
        const JSONValue name_var = asked_var.child("name");
        if (!name_var.isNull()) {
          name += name_var.valueAsStringView();
        }
      }

      const JSONValue arcane_params = asked_var.child("arcane").child("options");
      if (!arcane_params.isNull()) {
        const JSONKeyValueList params = arcane_params.keyValueChildren();

        for (auto elem : params) {
          cargs.addParameterLine( String::format("{0}={1}", elem.name(), elem.value().valueAsStringView()));

          // std::cout << "Elem : " << elem.name() << " -- val : " << elem.value().valueAsStringView() << std::endl;
        }
      }
    }
  }

  // std::cout << "Name variation : " << name << std::endl;

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
