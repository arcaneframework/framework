// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueConvertInternal.h                                      (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_VALUECONVERTINTERNAL_H
#define ARCANE_UTILS_INTERNAL_VALUECONVERTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ValueConvert.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline bool
builtInGetArrayValueFromStream(Array<T>& v, std::istream& sbuf)
{
  T read_val = T();
  if (!sbuf.eof())
    sbuf >> ws;
  while (!sbuf.eof()) {
    sbuf >> read_val;
    if (sbuf.fail() || sbuf.bad())
      return true;
    v.add(read_val);
    sbuf >> ws;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline bool
builtInGetArrayValue(Array<T>& v, StringView s)
{
  impl::StringViewInputStream svis(s);
  std::istream& sbuf = svis.stream();
  return builtInGetArrayValueFromStream(v, sbuf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
  /*!
   * \brief Indique si on utilise 'std::from_chars' pour convertir
   * les chaînes de caractères en un type numérique.
   *
   * Si on n'utilise pas 'std::from_chars', alors on utilise les fonctions
   * telles que strtod(), strtol(), ...
   *
   * Le défaut en C++20 est d'utiliser std::from_chars().
   */
  extern "C++" ARCANE_UTILS_EXPORT void
  arcaneSetIsValueConvertUseFromChars(bool v);

  //! Positionne le niveau de verbosité pour les fonctions de conversion.
  extern "C++" ARCANE_UTILS_EXPORT void
  arcaneSetValueConvertVerbosity(Int32 v);

  /*!
   * Si vrai, utilise le même mécanisme pour lire les 'RealN' que pour lire les 'Real'.
   *
   * Avant la version 3.15 de Arcane, la lecture des 'Real' se fait via std::strtod()
   * et celle des 'RealN' via std::istream. Si \a v est vrai, on utilise
   * std::strtod() pour tout le monde (ou std::from_chars()) si disponible.
   */
  extern "C++" ARCANE_UTILS_EXPORT void
  arcaneSetUseSameValueConvertForAllReal(bool v);
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

