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
/* GlibAdapter.h                                               (C) 2000-2018 */
/*                                                                           */
/* Classes utilitaires pour s'adapter aux différentes versions de la 'glib'. */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_GLIBADAPTER_H
#define ARCCORE_CONCURRENCY_GLIBADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

class GlibCond;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Encapsule un GMutex de la glib.
 */
class ARCCORE_CONCURRENCY_EXPORT GlibMutex
{
  friend class GlibCond;
 public:
  class Impl;
 public:
  class Lock
  {
   public:
    Lock(GlibMutex& x);
    ~Lock();
    Lock() = delete;
    Lock(const Lock&) = delete;
    void operator=(const Lock&) = delete;
   private:
    Impl* m_mutex;
  };
 public:
  GlibMutex() ARCCORE_NOEXCEPT;
  ~GlibMutex();
 public:
  void lock();
  void unlock();
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Encapsule un GPrivate de la glib.
 */
class ARCCORE_CONCURRENCY_EXPORT GlibPrivate
{
 public:
  class Impl;
 public:
  GlibPrivate();
  ~GlibPrivate();
  void create();
  void setValue(void* value);
  void* getValue();
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Encapsule un GCond de la glib.
 */
class ARCCORE_CONCURRENCY_EXPORT GlibCond
{
 public:
  class Impl;
 public:
  GlibCond();
  ~GlibCond();
  void broadcast();
  void wait(GlibMutex* mutex);
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
