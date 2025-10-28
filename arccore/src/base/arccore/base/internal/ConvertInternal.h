// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConvertInternal.h                                           (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INTERNAL_CONVERTINTERNAL_H
#define ARCCORE_BASE_INTERNAL_CONVERTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Informations sur le comportement des méthodes de conversion
class ARCCORE_BASE_EXPORT ConvertPolicy
{
 public:

  /*!
   * \brief Indique si on utilise 'std::from_chars' pour convertir
   * les chaînes de caractères en un type numérique.
   *
   * Si on n'utilise pas 'std::from_chars', alors on utilise les fonctions
   * telles que strtod(), strtol(), ...
   *
   * Le défaut en C++20 est d'utiliser std::from_chars().
   */
  static void setUseFromChars(bool v) { m_use_from_chars = v; }
  static bool isUseFromChars() { return m_use_from_chars; }

  //! Positionne le niveau de verbosité pour les fonctions de conversion.
  static void setVerbosity(Int32 v) { m_verbosity = v; }
  static bool verbosity() { return m_verbosity; }

  /*!
   * Si vrai, utilise le même mécanisme pour lire les 'RealN' que pour lire les 'Real'.
   *
   * Avant la version 3.15 de Arcane, la lecture des 'Real' se fait via std::strtod()
   * et celle des 'RealN' via std::istream. Si \a v est vrai, on utilise
   * std::strtod() pour tout le monde (ou std::from_chars()) si disponible.
   */
  static void setUseSameConvertForAllReal(bool v)
  {
    m_use_same_convert_for_all_real = v;
  }
  static bool isUseSameConvertForAllReal()
  {
    return m_use_same_convert_for_all_real;
  }

 private:

  static Int32 m_verbosity;
  static bool m_use_from_chars;
  static bool m_use_same_convert_for_all_real;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour convertir une 'StringView' en 'double'.
 */
class ARCCORE_BASE_EXPORT StringViewToDoubleConverter
{
 public:

  static Int64 _getDoubleValueWithFromChars(double& v, StringView s);
  static Int64 _getDoubleValue(double& v, StringView s);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT StringView
_removeLeadingSpaces(StringView s, Int64 pos = 0);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

