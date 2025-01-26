// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVariableReplace.h                                     (C) 2000-2025 */
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

#include "arcane/utils/ParameterList.h"

#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT StringVariableReplace
{
 public:

  static String replaceWithCmdLineArgs(const ParameterList& parameter_list, StringView string_with_symbols, bool fatal_if_not_found = false, bool fatal_if_invalid = true);

 private:

  static void _splitString(StringView str_view, ArrayView<StringView> str_view_array, char c);
  static void _countChar(StringView str_view, char c, Integer& count_c, Integer& count_c_with_escape);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
