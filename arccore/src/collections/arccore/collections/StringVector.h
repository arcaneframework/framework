// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVector.h                                              (C) 2000-2025 */
/*                                                                           */
/* Liste de 'String'.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_STRINGVECTOR_H
#define ARCCORE_COLLECTIONS_STRINGVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de 'String'.
 *
 * Cette classe à une sémantique par valeur et a le même comportement
 * qu'un UniqueArray<String>.
 */
class ARCCORE_COLLECTIONS_EXPORT StringVector
{
  class Impl;

 public:

  StringVector() = default;
  StringVector(const StringVector& rhs);
  StringVector(StringVector&& rhs) noexcept;
  StringVector& operator=(const StringVector& rhs);
  ~StringVector();

 public:

  Int32 size() const;
  void add(const String& str);
  String operator[](Int32 index) const;

 private:

  Impl* m_p = nullptr;

 private:

  inline void _checkNeedCreate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
