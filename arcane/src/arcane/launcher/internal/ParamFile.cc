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
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/JSONReader.h"
#include "arccore/common/ArcaNetReader.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParamFile::ArgsFill
{
 public:

  ArgsFill(CommandLineArguments& args)
  : m_args(args)
  {}

 public:

  void readFilePart(JSONValue file_part);
  void readArcanePart(JSONValue arcane_part);
  String name() const { return m_name; }

 private:

  CommandLineArguments& m_args;
  StringBuilder m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::ArgsFill::
readFilePart(const JSONValue file_part)
{
  const JSONValue dataset = file_part.child("name");
  if (!dataset.isNull()) {
    m_name += " -> \"";
    m_name += dataset.valueAsStringView();
    m_name += "\"";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParamFile::ArgsFill::
readArcanePart(const JSONValue arcane_part)
{
  const JSONValue dataset = arcane_part.child("dataset");
  if (!dataset.isNull()) {
    m_args.addParameterLine(String::format("CaseDatasetFileName={0}", dataset.value()));
    //std::cout << "Dataset : " << dataset.value() << std::endl;
  }

  const JSONValue arcane_params = arcane_part.child("options");

  if (!arcane_params.isNull()) {
    const JSONKeyValueList params = arcane_params.keyValueChildren();

    for (auto elem : params) {
      m_args.addParameterLine(String::format("{0}={1}", elem.name(), elem.value().valueAsStringView()));
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
  CommandLineArguments args = ArcaneLauncher::applicationInfo().commandLineArguments();

  JSONDocument json_doc;
  {
    UniqueArray<Byte> bytes;

    if (platform::readAllFile(param_file_name, false, bytes)) {
      ARCANE_FATAL("Param file not available");
    }

    json_doc.parse(bytes, param_file_name, (JSONDocument::ParseCommentsFlag | JSONDocument::ParseNumbersAsStringsFlag));
  }

  ArcaNetReader reader(json_doc.root());
  reader.init(variation, "arcane");
  reader.read();

  ArgsFill args_fill(args);

  ArrayView<JSONValue> json_values = reader.jsonValues();
  for (JSONValue elem : json_values) {
    args_fill.readFilePart(reader.extractFilePart(elem));
    args_fill.readArcanePart(reader.extractProgPart(elem));
  }

  std::cout << "ArcaNet path: " << args_fill.name() << std::endl;

  if (reader.computePath()) {
    std::cout << "ArcaNet full path: " << reader.path() << std::endl;
    std::cout << "ArcaNet config part readed: " << reader.configPartName() << std::endl;
  }

  // StringList names;
  // StringList values;
  // args.fillParameters(names, values);
  // for (Integer i = 0, n = names.count(); i < n; ++i) {
  //   std::cout << "Final Elem : " << names[i] << " -- val : " << values[i] << std::endl;
  // }

  ArcaneLauncher::applicationInfo().setCommandLineArguments(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
