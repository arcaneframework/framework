// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplace.h                                     (C) 2000-2023 */
/*                                                                           */
/* Classe permettant de remplacer les symboles d'une chaine de caractères    */
/* par une autre chaine de caractères définie dans les arguments de          */
/* lancement.                                                                */
/* Un symbole est défini par une chaine de caractères entourée de @.         */
/* Exemple : @mon_symbole@                                                   */
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/StringVariableReplace.h"
#include "arcane/utils/PlatformUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String StringVariableReplace::
replaceWithCmdLineArgs(const ParameterList& parameter_list, const String& name)
{
  // Si la variable d'environnement ARCANE_REPLACE_SYMBOLS_IN_DATASET n'est pas
  // définie, on ne touche pas à la chaine de caractères.
  if (platform::getEnvironmentVariable("ARCANE_REPLACE_SYMBOLS_IN_DATASET").null()) {
    return name;
  }

  if(name.empty()) return name;

  String name_cpy = name;

  // Permet de contourner le bug avec String::split() si le nom commence par '@'.
  if (name_cpy.startsWith("@")) {
    name_cpy = "@" + name_cpy;
  }

  StringUniqueArray string_splited;
  // On découpe la string là où se trouve les @.
  name_cpy.split(string_splited, '@');

  // S'il n'y a aucun @, on retourne la chaine d'origine.
  if(string_splited.size() == 1) return name;

  for(auto & elem : string_splited){
    String reference_input = parameter_list.getParameterOrNull(elem);
    if(reference_input.null()) continue;
    elem = reference_input;
  }

  // On recombine la chaine de caractères.
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
