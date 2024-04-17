// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.cc                                                   (C) 2000-2024 */
/*                                                                           */
/* Types atomiques pour le multi-threading.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Atomic.h"

#include <glib.h>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Pour l'instant, utilise les fonctions de la glib. A terme, il faudra
 * utiliser le type std::atomic de la stl de la norme C++0x.
 *
 * Pour supporter les veilles versions de la glib, il ne faut
 * utiliser que les fonctions suivantes:
 * 
 * gint g_atomic_int_exchange_and_add(gint *atomic, gint val);
 * void g_atomic_int_add(gint *atomic,gint val);
 * gboolean g_atomic_int_compare_and_exchange(gint *atomic,gint oldval, gint newval);
 * gint g_atomic_int_get(gint *atomic);
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AtomicInt32::
AtomicInt32(int v)
{
  g_atomic_int_set(&m_value, v);
}

Int32 AtomicInt32::
operator++()
{
  return g_atomic_int_add(&m_value, 1) + 1;
}

Int32 AtomicInt32::
operator--()
{
  return g_atomic_int_add(&m_value, -1) - 1;
}

void AtomicInt32::
operator=(Int32 v)
{
  g_atomic_int_set(&m_value, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AtomicInt32::
value() const
{
  return g_atomic_int_get(&m_value);
}

Int32 AtomicInt32::
increment(volatile Int32* v)
{
  return (g_atomic_int_add(v, 1) + 1);
}

Int32 AtomicInt32::decrement(volatile Int32* v)
{
  return g_atomic_int_add(v, -1) - 1;
}

void AtomicInt32::
setValue(volatile Int32* v, Int32 new_v)
{
  g_atomic_int_set(v, new_v);
}

Int32 AtomicInt32::
getValue(volatile Int32* v)
{
  return g_atomic_int_get(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
