﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryTracer.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Utilitaires pour tracer les accès mémoire entre l'accélérateur et l'hôte. */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/MemoryTracer.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IMemoryAllocator.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

class MemoryTracerMng
{
  struct Info
  {
    Int64 length;
    String name;
    String stack;
  };

 public:

  void add(Span<const std::byte> bytes, const String& name, const String& stack_trace, [[maybe_unused]] Int64 timestamp)
  {
    const void* ptr = bytes.data();
    m_infos_map.insert(std::make_pair(ptr, Info{ bytes.size(), name, stack_trace }));
  }

  void remove(void* ptr, [[maybe_unused]] const String& name, [[maybe_unused]] const String& stack_trace, [[maybe_unused]] Int64 timestamp)
  {
    auto x = m_infos_map.find(ptr);
    if (x != m_infos_map.end())
      m_infos_map.erase(x);
  }

  std::pair<String, String> find(const void* ptr)
  {
    auto x = m_infos_map.lower_bound(ptr);
    if (x != m_infos_map.end())
      return std::make_pair(x->second.name, x->second.stack);
    return {};
  }

 private:

  std::map<const void*, Info> m_infos_map;
};

namespace
{
  MemoryTracerMng m_memory_tracer_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryTracer::
notifyMemoryAllocation(Span<const std::byte> bytes, const String& name,
                       const String& stack_trace, Int64 timestamp)
{
  // TODO: rendre thread-safe
  m_memory_tracer_mng.add(bytes, name, stack_trace, timestamp);
}

void MemoryTracer::
notifyMemoryFree(void* ptr, const String& name, const String& stack_trace, Int64 timestamp)
{
  // TODO: rendre thread-safe
  m_memory_tracer_mng.remove(ptr, name, stack_trace, timestamp);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<String, String> MemoryTracer::
findMemory(const void* ptr)
{
  return m_memory_tracer_mng.find(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryTracerWrapper::
MemoryTracerWrapper()
{
  // TODO: Utiliser une autre variable d'environnement que CUDA
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_MALLOC_TRACE", true))
    m_trace_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryTracerWrapper::
traceDeallocate(const AllocatedMemoryInfo& mem_info, const MemoryAllocationArgs& args)
{
  if (!isActive())
    return;

  void* ptr = mem_info.baseAddress();
  // Utilise un flux spécifique pour être sur que les affichages ne seront pas mélangés
  // en cas de multi-threading
  std::ostringstream ostr;
  if (m_trace_level >= 2)
    ostr << "FREE_MANAGED=" << ptr << " size=" << mem_info.capacity() << " name=" << args.arrayName();
  String s;
  if (m_trace_level >= 3) {
    s = platform::getStackTrace();
    if (m_trace_level >= 4) {
      ostr << " stack=" << s;
    }
  }
  impl::MemoryTracer::notifyMemoryFree(ptr, args.arrayName(), s, 0);
  if (m_trace_level >= 2) {
    ostr << "\n";
    std::cout << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryTracerWrapper::
traceAllocate(void* p, size_t new_size, MemoryAllocationArgs args)
{
  if (!isActive())
    return;

  // Utilise un flux spécifique pour être sur que les affichages ne seront pas mélangés
  // en cas de multi-threading
  std::ostringstream ostr;
  if (m_trace_level >= 2)
    ostr << "MALLOC_MANAGED=" << p << " size=" << new_size << " name=" << args.arrayName();
  String s;
  if (m_trace_level >= 3) {
    s = platform::getStackTrace();
    if (m_trace_level >= 4) {
      ostr << " stack=" << s;
    }
  }
  Span<const std::byte> bytes(reinterpret_cast<std::byte*>(p), new_size);
  impl::MemoryTracer::notifyMemoryAllocation(bytes, args.arrayName(), s, 0);
  if (m_trace_level >= 2) {
    ostr << "\n";
    std::cout << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
