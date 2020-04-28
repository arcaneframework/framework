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

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(IThreadImplementation);

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

