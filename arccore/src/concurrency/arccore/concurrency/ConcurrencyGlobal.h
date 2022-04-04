﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyGlobal.h                                         (C) 2000-2020 */
/*                                                                           */
/* Définitions globales de la composante 'Concurrency' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_CONCURRENCYGLOBAL_H
#define ARCCORE_CONCURRENCY_CONCURRENCYGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/RefDeclarations.h"

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
class SpinLock;
class GlobalMutex;
class IThreadBarrier;
class NullThreadImplementation;
class NullThreadBarrier;

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

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(IThreadImplementation)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

