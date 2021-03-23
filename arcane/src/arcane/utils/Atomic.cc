// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.cc                                                   (C) 2000-2016 */
/*                                                                           */
/* Types atomiques pour le multi-threading.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Atomic.h"

#include <glib.h>

// NOTE: Versions de GLIB suivant les distributions Linux.
// CentOS 6 -> 2.28
// CentOS 7 -> 2.46
#if GLIB_CHECK_VERSION(2,30,0)
#define ARCANE_GLIB_HAS_ATOMIC_ADD
#endif

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
  g_atomic_int_set(&m_value,v);
}

Int32 AtomicInt32::
operator++()
{
#ifdef ARCANE_GLIB_HAS_ATOMIC_ADD
  return g_atomic_int_add(&m_value,1)+1;
#else
  return g_atomic_int_exchange_and_add(&m_value,1)+1;
#endif
}

Int32 AtomicInt32::
operator--()
{
#ifdef ARCANE_GLIB_HAS_ATOMIC_ADD
  return g_atomic_int_add(&m_value,-1)-1;
#else
  return g_atomic_int_exchange_and_add(&m_value,-1)-1;
#endif
}

void AtomicInt32::
operator=(Int32 v)
{
  g_atomic_int_set(&m_value,v);
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
#ifdef ARCANE_GLIB_HAS_ATOMIC_ADD
  return (g_atomic_int_add(v, 1) + 1);
#else
  return g_atomic_int_exchange_and_add(v,1)+1;
#endif
}

Int32 AtomicInt32::decrement(volatile Int32* v)
{
#ifdef ARCANE_GLIB_HAS_ATOMIC_ADD
  return g_atomic_int_add(v,-1)-1;
#else
  return g_atomic_int_exchange_and_add(v,-1)-1;
#endif
}

void AtomicInt32::
setValue(volatile Int32* v,Int32 new_v)
{
  g_atomic_int_set(v,new_v);
}

Int32 AtomicInt32::
getValue(volatile Int32* v)
{
  return g_atomic_int_get(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
