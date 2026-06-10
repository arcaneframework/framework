// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalRef.h                                               (C) 2000-2025 */
/*                                                                           */
/* Management of a reference to an object external to C++.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_EXTERNALREF_H
#define ARCCORE_BASE_EXTERNALREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Management of references to an external object.
 *
 * This class allows saving references to objects that are not managed
 * directly by %Arccore. This is the case, for example, when using C# wrapping
 * and wanting to manipulate objects 'managed' by '.Net' that no longer
 * necessarily have an explicit managed reference. This class allows
 * maintaining a reference to these objects to prevent them from being
 * collected by the garbage collector.
 *
 * This class is internal to %Arccore and should in principle not be used
 * directly.
 */
class ARCCORE_BASE_EXPORT ExternalRef
{
 private:

  struct Handle
  {
    Handle()
    : handle(nullptr)
    {}
    Handle(void* h)
    : handle(h)
    {}
    ~Handle();
    void addReference() { ++m_nb_ref; }
    void removeReference()
    {
      Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
      if (v == 1)
        delete this;
    }
    void* handle;
    std::atomic<int> m_nb_ref = 0;
  };

 public:

  typedef void (*DestroyFuncType)(void* handle);

 public:

  ExternalRef() = default;
  ExternalRef(void* handle)
  : m_handle(new Handle(handle))
  {}

 public:

  bool isValid() const
  {
    Handle* p = m_handle.get();
    if (!p)
      return false;
    return _internalHandle() != nullptr;
  }
  void* _internalHandle() const { return m_handle->handle; }

 private:

  Arccore::ReferenceCounter<Handle> m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::Internal
{
using Arcane::Internal::ExternalRef;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
