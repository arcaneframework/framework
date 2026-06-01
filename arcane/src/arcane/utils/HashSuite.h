// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashSuite.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Hash function for a sequence of values.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHSUITE_H
#define ARCANE_UTILS_HASHSUITE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class allowing for iterative hash calculation.
 * \warning The order in which values are provided via the add() method is important.
 */
class IntegerHashSuite
{
 public:

  /*!
   * \brief Method allowing a value to be added to the hash calculation.
   * \warning The order in which values are provided via the method
   * add() is important.
   * \param value The value to add.
   */
  template <class T>
  void add(T value)
  {
    const UInt64 next_hash = static_cast<UInt64>(IntegerHashFunctionT<T>::hashfunc(value));
    m_hash ^= next_hash + 0x9e3779b9 + (m_hash << 6) + (m_hash >> 2);
  }

  /*!
   * \brief Method allowing retrieval of the hash calculated from
   * all values passed to the add() method.
   * \return The hash.
   */
  Int64 hash() const
  {
    return static_cast<Int64>(m_hash);
  }

 private:

  UInt64 m_hash{ 0 };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
