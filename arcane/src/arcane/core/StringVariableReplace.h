// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplace.h                                     (C) 2000-2023 */
/*                                                                           */
/* TODO.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STRINGVARIABLEREPLACE_H
#define ARCANE_STRINGVARIABLEREPLACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/StringBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringVariableReplace
{
 public:

  static String replaceWithCmdLineArgs(const ApplicationInfo& appinfo, const String& name){
    if(name.empty()) return name;

    String name_cpy = name;

    // Permet de contourner le bug avec String::split() si le nom commence par '@'.
    if (name_cpy.startsWith("@")) {
      name_cpy = "@" + name_cpy;
    }

    StringUniqueArray string_splited;
    // On découpe la string là où se trouve les @.
    name_cpy.split(string_splited, '@');

    if(string_splited.size() == 1) return name;

    for(auto & elem : string_splited){
      String reference_input = appinfo.commandLineArguments().getParameter(elem);
      if(reference_input.null()) continue;
      elem = reference_input;
    }

    // On recombine la chaine.
    StringBuilder combined = "";
    for (const String& str : string_splited) {
      // Permet de contourner le bug avec String::split() s'il y a '@@@' dans le nom ou si le
      // nom commence par '@' (en complément des premières lignes de la méthode).
      if (str == "@")
        continue;
      combined.append(str);
    }

    return combined.toString();
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

