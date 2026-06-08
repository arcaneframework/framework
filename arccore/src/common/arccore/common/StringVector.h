// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringVector.h                                              (C) 2000-2026 */
/*                                                                           */
/* List of 'String'.                                                         */
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
 * \brief Vector of 'String'.
 *
 * This class has a value semantics and behaves the same as
 * a UniqueArray<String>.
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

  //! Number of elements
  Int32 size() const;
  //! Adds str to the list of strings
  void add(const String& str);
  //! Returns the i-th string
  String operator[](Int32 index) const;

  //! Converts the instance to 'StringList'
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
