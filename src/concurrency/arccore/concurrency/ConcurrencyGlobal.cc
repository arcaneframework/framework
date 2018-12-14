// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* ConcurrencyGlobal.h                                         (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'Concurrency' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

#include "arccore/concurrency/NullThreadImplementation.h"
#include "arccore/concurrency/SpinLock.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace Concurrency
{
NullThreadImplementation global_null_thread_implementation;
IThreadImplementation* global_thread_implementation = &global_null_thread_implementation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation* Concurrency::
getThreadImplementation()
{
  return global_thread_implementation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation* Concurrency::
setThreadImplementation(IThreadImplementation* service)
{
  IThreadImplementation* old_service = global_thread_implementation;
  global_thread_implementation = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
