// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.h                                                    (C) 2000-2024 */
/*                                                                           */
/* Types atomiques pour le multi-threading.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ATOMIC_H
#define ARCANE_UTILS_ATOMIC_H
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
 * \brief Type 'Int32' atomique.
 *
 * \deprecated Cette classe est obsolète. Il faut utiliser std::atomic<Int32>
 * à la place.
 */
class ARCANE_UTILS_EXPORT AtomicInt32
{
 public:

  //! Constructeur: attention, aucune initialisation
  ARCANE_DEPRECATED_REASON("Y2022: Use std::atomic<Int32> instead")
  AtomicInt32() {}
  ARCANE_DEPRECATED_REASON("Y2022: Use std::atomic<Int32> instead")
  AtomicInt32(Int32);

 public:

  Int32 operator++();
  Int32 operator--();
  Int32 value() const;
  void operator=(Int32 v);

  ARCANE_DEPRECATED_REASON("Y2022: Use std::atomic<Int32>::fetch_add(1) instead")
  static Int32 increment(volatile Int32* v);

  ARCANE_DEPRECATED_REASON("Y2022: Use std::atomic<Int32>::fetch_sub(1) instead")
  static Int32 decrement(volatile Int32* v);

  ARCANE_DEPRECATED_REASON("Y2022: Use std::atomic<Int32>::store() instead")
  static void setValue(volatile Int32* v, Int32 new_v);

  ARCANE_DEPRECATED_REASON("Y2022: Use std::atomic<Int32>::load() instead")
  static Int32 getValue(volatile Int32* v);

 private:

  mutable Int32 m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
