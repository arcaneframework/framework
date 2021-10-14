// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CppNameDemangler.h                                          (C) 2000-2021 */
/*                                                                           */
/* Classe pour 'demangler' un nom correspondant à un type C++.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CPPNAMEDEMANGLER_H
#define ARCANE_UTILS_CPPNAMEDEMANGLER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour 'demangler' un nom correspondant à un type C++.
 *
 * Cela peut être par exemple le nom retourné par std::type_info::name().
 */
class ARCANE_UTILS_EXPORT CppNameDemangler
{
  class Impl;

 public:
  CppNameDemangler();
  explicit CppNameDemangler(size_t buf_len);
  ~CppNameDemangler();
  CppNameDemangler(const CppNameDemangler&) = delete;
  CppNameDemangler(CppNameDemangler&&) = delete;
  CppNameDemangler operator=(const CppNameDemangler&) = delete;

 public:
  /*!
   * \brief 'Demangle' le nom \a mangled_name.
   *
   * Si aucun implémentation n'est disponible ou si le nom n'est pas manglé,
   * retourne \a mangled_name.
   *
   * Le pointeur retourné est une chaîne de caractères avec un '0' terminal.

   * \note Le pointeur retourné est invalidé lors du prochain appel à cette méthode
   * demangle(). Il ne faut donc pas le conserver entre deux appels.
   */
  const char* demangle(const char* mangled_name);

 private:
  Impl* m_p = nullptr;

 private:
  void _init(size_t len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

