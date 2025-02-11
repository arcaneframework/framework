// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadPrivate.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de conserver une valeur spécifique par thread.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ThreadPrivate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ThreadPrivateStorage::
ThreadPrivateStorage()
: m_storage(nullptr)
{
}

ThreadPrivateStorage::
~ThreadPrivateStorage()
{
}

void ThreadPrivateStorage::
initialize()
{
  if (!m_storage){
    m_storage = new GlibPrivate();
    m_storage->create();
  }
}

void* ThreadPrivateStorage::
getValue()
{
  return m_storage->getValue();
}


void ThreadPrivateStorage::
setValue(void* v)
{
  m_storage->setValue(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* ThreadPrivateBase::
item()
{
  void* ptr = m_key->getValue();
  if (ptr){
    return ptr;
  }
  void* new_ptr = nullptr;
  {
    GlibMutex::Lock x(m_mutex);
    new_ptr = m_create_functor->createInstance();
  }
  m_key->setValue(new_ptr);
  return new_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
