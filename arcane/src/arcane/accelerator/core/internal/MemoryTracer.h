// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryTracer.h                                              (C) 2000-2024 */
/*                                                                           */
/* Utilitaires pour tracer les accès mémoire entre l'accélérateur et l'hôte. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_MEMORYTRACER_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_MEMORYTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include <tuple>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe utilitaire pour tracer les accès mémoire entre l'accélérateur
 * et l'hôte.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT MemoryTracer
{
 public:

  static void notifyMemoryAllocation(Span<const std::byte> bytes, const String& name, const String& stack_trace, Int64 timestamp);
  static void notifyMemoryFree(void* ptr, const String& name, const String& stack_trace, Int64 timestamp);
  static std::pair<String, String> findMemory(const void* ptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe utilitaire pour tracer les allocations/désallocation
 */
class ARCANE_ACCELERATOR_CORE_EXPORT MemoryTracerWrapper
{
 public:

  MemoryTracerWrapper();

 public:

  bool isActive() const { return m_trace_level > 0; }
  void traceDeallocate(const AllocatedMemoryInfo& mem_info, const MemoryAllocationArgs& args);
  void traceAllocate(void* p, size_t new_size, MemoryAllocationArgs args);
  void setTraceLevel(Int32 v) { m_trace_level = v; }

 private:

  Int32 m_trace_level = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
