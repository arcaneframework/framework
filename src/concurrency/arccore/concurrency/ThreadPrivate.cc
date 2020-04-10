// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadPrivate.cc                                            (C) 2000-2018 */
/*                                                                           */
/* Classe permettant de conserver une valeur spécifique par thread.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ThreadPrivate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
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
