// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* ConcurrencyGlobal.h                                         (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'Concurrency' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"
#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arccore/concurrency/NullThreadImplementation.h"
#include "arccore/concurrency/SpinLock.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace
{
NullThreadImplementation global_null_thread_implementation;
ReferenceCounter<IThreadImplementation> global_thread_implementation{&global_null_thread_implementation};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation* Concurrency::
getThreadImplementation()
{
  return global_thread_implementation.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation* Concurrency::
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
