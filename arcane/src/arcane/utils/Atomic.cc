// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.cc                                                   (C) 2000-2025 */
/*                                                                           */
/* Types atomiques pour le multi-threading.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Atomic.h"

#include <atomic>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  void _setValue(volatile Int32* ptr, Int32 value)
  {
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    r.store(value);
  }
  Int32 _getValue(volatile Int32* ptr)
  {
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    return r.load();
  }
  Int32 _atomicAdd(volatile Int32* ptr)
  {
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    return r.fetch_add(1) + 1;
  }
  Int32 _atomicSub(volatile Int32* ptr)
  {
    std::atomic_ref<Int32> r(*const_cast<Int32*>(ptr));
    return r.fetch_sub(1) - 1;
  }
} // namespace

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
