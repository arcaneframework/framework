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

namespace Arcane
{
class IThreadImplementation;
class Mutex;
class SpinLock;
class GlobalMutex;
class IThreadBarrier;
class NullThreadImplementation;
class NullThreadBarrier;

//@{ Classe internes à Arccore/Arcane
class SpinLockImpl;
class GlibThreadImplementation;
//@}

//! Classe opaque encapsulant l'implementation des threads
class ThreadImpl;
//! Classe opaque encapsulant l'implementation d'un mutex
class MutexImpl;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IThreadImplementation)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Concurrency
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation*
getThreadImplementation();
extern "C++" ARCCORE_CONCURRENCY_EXPORT IThreadImplementation*
setThreadImplementation(IThreadImplementation* impl);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_CONCURRENCY_EXPORT Ref<IThreadImplementation>
createGlibThreadImplementation();

extern "C++" ARCCORE_CONCURRENCY_EXPORT Ref<IThreadImplementation>
createStdThreadImplementation();

extern "C++" ARCCORE_CONCURRENCY_EXPORT Ref<IThreadImplementation>
createNullThreadImplementation();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Concurrency

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arcane::GlibThreadImplementation;
using Arcane::GlobalMutex;
using Arcane::IThreadBarrier;
using Arcane::IThreadImplementation;
using Arcane::Mutex;
using Arcane::MutexImpl;
using Arcane::NullThreadBarrier;
using Arcane::NullThreadImplementation;
using Arcane::SpinLock;
using Arcane::SpinLockImpl;
using Arcane::ThreadImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::Concurrency
{
using Arcane::Concurrency::getThreadImplementation;
using Arcane::Concurrency::setThreadImplementation;
using Arcane::Concurrency::createGlibThreadImplementation;
using Arcane::Concurrency::createStdThreadImplementation;
using Arcane::Concurrency::createNullThreadImplementation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
