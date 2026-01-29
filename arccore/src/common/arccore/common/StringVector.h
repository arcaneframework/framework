// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVector.h                                              (C) 2000-2026 */
/*                                                                           */
/* Liste de 'String'.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_STRINGVECTOR_H
#define ARCCORE_COMMON_STRINGVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur de 'String'.
 *
 * Cette classe à une sémantique par valeur et a le même comportement
 * qu'un UniqueArray<String>.
 */
class ARCCORE_COMMON_EXPORT StringVector
{
  class Impl;

 public:

  StringVector() = default;
  explicit StringVector(const StringList& string_list);
  StringVector(const StringVector& rhs);
  StringVector(StringVector&& rhs) noexcept;
  StringVector& operator=(const StringVector& rhs);
  ~StringVector();

 public:

  //! Nombre d'éléments
  Int32 size() const;
  //! Ajoute \a str à la liste des chaînes de caractères
  void add(const String& str);
  //! Retourne la i-ème chaîne de caractères
  String operator[](Int32 index) const;

  //! Converti l'instance en 'StringList'
  StringList toStringList() const;

 private:

  Impl* m_p = nullptr;

 private:

  inline void _checkNeedCreate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
