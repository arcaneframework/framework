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

#ifndef ARCANE_CORE_INTERNAL_STRINGVARIABLEREPLACE_H
#define ARCANE_CORE_INTERNAL_STRINGVARIABLEREPLACE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/utils/ParameterList.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/StringBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT StringVariableReplace
{
 public:
  static String replaceWithCmdLineArgs(const ParameterList& parameter_list, const String& string_with_symbols);

 private:
  static void _splitString(const String& str, UniqueArray<String>& str_array, UniqueArray<Integer>& int_array, Integer& nb_c, char c);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
