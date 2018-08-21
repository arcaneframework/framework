/*---------------------------------------------------------------------------*/
/* ConcurrencyGlobal.h                                         (C) 2000-2018 */
/*                                                                           */
/* Définitions globales de la composante 'Concurrency' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_CONCURRENCYGLOBAL_H
#define ARCCORE_CONCURRENCY_CONCURRENCYGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_concurrency)
#define ARCCORE_CONCURRENCY_EXPORT ARCCORE_EXPORT
#define ARCCORE_CONCURRENCY_EXTERN_TPL
#else
#define ARCCORE_CONCURRENCY_EXPORT ARCCORE_IMPORT
#define ARCCORE_CONCURRENCY_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
class IThreadImplementation;
class Mutex;
class GlobalMutex;
class IThreadBarrier;

//! Classe opaque encapsulant l'implementation des threads
class ThreadImpl;
//! Classe opaque encapsulant l'implementation d'un mutex
class MutexImpl;

namespace Concurrency
{
extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation*
getThreadImplementation();
extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation*
setThreadImplementation(IThreadImplementation* impl);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

