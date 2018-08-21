/*---------------------------------------------------------------------------*/
/* ConcurrencyGlobal.h                                         (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'Concurrency' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace Concurrency
{
IThreadImplementation* global_thread_implementation_service = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation* Concurrency::
getThreadImplementation()
{
  return global_thread_implementation_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation* Concurrency::
setThreadImplementation(IThreadImplementation* service)
{
  IThreadImplementation* old_service = global_thread_implementation_service;
  global_thread_implementation_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
