// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlibAdapter.h                                               (C) 2000-2025 */
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

namespace Arcane
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

  Impl* m_p = nullptr;
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

  Impl* m_p = nullptr;
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

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
