// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemEnumeratorTracer.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface for tracing calls to enumerators on entities.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMENUMERATORTRACER_H
#define ARCANE_CORE_IITEMENUMERATORTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for an enumerator tracer on entities.
 *
 * This interface provides methods that are called automatically
 * when using the macros that allow iteration over entities
 * such as ENUMERATE_CELL or ENUMERATE_SIMD_CELL. For performance reasons,
 * these macros are only traced if the source file using them is
 * compiled with the ARCANE_TRACE_ENUMERATOR macro.
 *
 * The singleton() method allows retrieving the current implementation.
 *
 */
class ARCANE_CORE_EXPORT IItemEnumeratorTracer
{
 public:

  static IItemEnumeratorTracer* singleton();

 public:

  virtual ~IItemEnumeratorTracer() = default;

 public:

  //! Method called before executing an ENUMERATE_
  virtual void enterEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti) = 0;

  //! Method called after executing an ENUMERATE_
  virtual void exitEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti) = 0;

  //! Method called before executing an ENUMERATE_SIMD_
  virtual void enterEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti) = 0;

  //! Method called after executing an ENUMERATE_SIMD_
  virtual void exitEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti) = 0;

 public:

  virtual void dumpStats() = 0;
  virtual IPerformanceCounterService* perfCounter() = 0;
  virtual Ref<IPerformanceCounterService> perfCounterRef() = 0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
