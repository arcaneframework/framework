// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyGlobal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Concurrency' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/Ref.h"
#include "arccore/base/NotSupportedException.h"

#include "arccore/concurrency/ParallelFor.h"
#include "arccore/concurrency/TaskFactory.h"
#include "arccore/concurrency/ITaskImplementation.h"
#include "arccore/concurrency/Task.h"

#include "arccore/concurrency/NullThreadImplementation.h"
#include "arccore/concurrency/SpinLock.h"

#ifdef ARCCORE_HAS_GLIB
#include "arccore/concurrency/GlibThreadImplementation.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  NullThreadImplementation global_null_thread_implementation;
  ReferenceCounter<IThreadImplementation> global_thread_implementation{ &global_null_thread_implementation };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IThreadImplementation* Concurrency::
getThreadImplementation()
{
  return global_thread_implementation.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IThreadImplementation* Concurrency::
setThreadImplementation(IThreadImplementation* service)
{
  IThreadImplementation* old_service = global_thread_implementation.get();
  global_thread_implementation = service;
  if (!service)
    global_thread_implementation = &global_null_thread_implementation;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IThreadImplementation> Concurrency::
createGlibThreadImplementation()
{
#ifdef ARCCORE_HAS_GLIB
  return makeRef<IThreadImplementation>(new GlibThreadImplementation());
#else
  throw NotSupportedException(A_FUNCINFO, "GLib is not available Recompile Arccore with ARCCORE_ENABLE_GLIB=TRUE");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullThreadImplementationFactory
{
 public:

  static Ref<IThreadImplementation> create()
  {
    return makeRef<>(new NullThreadImplementation());
  }
};

Ref<IThreadImplementation> Concurrency::
createNullThreadImplementation()
{
  return NullThreadImplementationFactory::create();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IThreadImplementation::
_deprecatedCreateSpinLock(Int64* spin_lock_addr)
{
  createSpinLock(spin_lock_addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IThreadImplementation::
_deprecatedLockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr)
{
  lockSpinLock(spin_lock_addr, scoped_spin_lock_addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IThreadImplementation::
_deprecatedUnlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr)
{
  unlockSpinLock(spin_lock_addr, scoped_spin_lock_addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ITaskImplementation::nbAllowedThread() const
{
  return ConcurrencyBase::maxAllowedThread();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IThreadImplementation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
