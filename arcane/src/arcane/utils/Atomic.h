// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.h                                                    (C) 2000-2017 */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type 'Int32' atomique.
 */
class ARCANE_UTILS_EXPORT AtomicInt32
{
 public:
  //! Constructeur: attention, aucune initialisation
  AtomicInt32(){}
 public:
  AtomicInt32(Int32);
  Int32 operator++();
  Int32 operator--();
  Int32 value() const;
  void operator=(Int32 v);
  static Int32 increment(volatile Int32* v);
  static Int32 decrement(volatile Int32* v);
  static void setValue(volatile Int32* v,Int32 new_v);
  static Int32 getValue(volatile Int32* v);
 private:
  Int32 m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
