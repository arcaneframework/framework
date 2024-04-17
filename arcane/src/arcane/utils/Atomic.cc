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

#ifdef ARCANE_HAS_CXX20
#include <atomic>
#else
#include <glib.h>
#endif

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  void _setValue(volatile Int32* ptr, Int32 value)
  {
#ifdef ARCANE_HAS_CXX20
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    r.store(value);
#else
    g_atomic_int_set(ptr, value);
#endif
  }
  Int32 _getValue(volatile Int32* ptr)
  {
#ifdef ARCANE_HAS_CXX20
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    return r.load();
#else
    return g_atomic_int_get(ptr);
#endif
  }
  Int32 _atomicAdd(volatile Int32* ptr)
  {
#ifdef ARCANE_HAS_CXX20
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    return r.fetch_add(1) + 1;
#else
    return g_atomic_int_add(ptr, 1) + 1;
#endif
  }
  Int32 _atomicSub(volatile Int32* ptr)
  {
#ifdef ARCANE_HAS_CXX20
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    return r.fetch_sub(1) - 1;
#else
    return g_atomic_int_add(ptr, -1) - 1;
#endif
  }
} // namespace

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
  _setValue(&m_value, v);
}

Int32 AtomicInt32::
operator++()
{
  return _atomicAdd(&m_value);
}

Int32 AtomicInt32::
operator--()
{
  return _atomicSub(&m_value);
}

void AtomicInt32::
operator=(Int32 v)
{
  _setValue(&m_value, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 AtomicInt32::
value() const
{
  return _getValue(&m_value);
}

Int32 AtomicInt32::
increment(volatile Int32* v)
{
  return _atomicAdd(v);
}

Int32 AtomicInt32::
decrement(volatile Int32* v)
{
  return _atomicSub(v);
}

void AtomicInt32::
setValue(volatile Int32* v, Int32 new_v)
{
  _setValue(v, new_v);
}

Int32 AtomicInt32::
getValue(volatile Int32* v)
{
  return _getValue(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
