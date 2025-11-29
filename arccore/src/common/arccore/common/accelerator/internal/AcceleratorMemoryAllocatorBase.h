// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMemoryAllocatorBase.h                            (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un allocateur spécifique pour accélérateur.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORMEMORYALLOCATORBASE_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORMEMORYALLOCATORBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/common/AllocatedMemoryInfo.h"
#include "arccore/common/AlignedMemoryAllocator.h"
#include "arccore/common/internal/MemoryPool.h"

#include "arccore/common/accelerator/internal/MemoryTracer.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe commune pour gérer l'allocation par bloc.
 *
 * Cette classe permet de garantir qu'on alloue la mémoire sur des
 * multiples de la taille d'un bloc.
 * Cela est notamment utilisé pour la mémoire unifiée ce qui permet d'éviter
 * des effets de bord entre les allocations pour les transferts
 * entre l'accélérateur CPU et l'hôte.
 *
 * Par défaut on alloue un multiple de 128 octets.
 */
class ARCCORE_COMMON_EXPORT BlockAllocatorWrapper
{
 public:

  void initialize(Int64 block_size, bool do_block_alloc)
  {
    m_block_size = block_size;
    if (m_block_size <= 0)
      m_block_size = 128;
    m_do_block_allocate = do_block_alloc;
  }

  void dumpStats(std::ostream& ostr, const String& name);

  Int64 adjustedCapacity(Int64 wanted_capacity, Int64 element_size) const
  {
    const bool do_page = m_do_block_allocate;
    if (!do_page)
      return wanted_capacity;
    // Alloue un multiple de la taille d'un bloc
    // Pour la mémoire unifiée, la taille de bloc est une page mémoire.
    // Comme les transfers de la mémoire unifiée se font par page,
    // cela permet de détecter quelles allocations provoquent le transfert.
    // On se débrouille aussi pour limiter les différentes taille
    // de bloc alloué pour éviter d'avoir trop de blocs de taille
    // différente pour que l'éventuel MemoryPool ne contienne trop
    // de valeurs.
    Int64 orig_capacity = wanted_capacity;
    Int64 new_size = orig_capacity * element_size;
    Int64 block_size = m_block_size;
    Int64 nb_iter = 4 + (4096 / block_size);
    for (Int64 i = 0; i < nb_iter; ++i) {
      if (new_size >= (4 * block_size))
        block_size *= 4;
      else
        break;
    }
    new_size = _computeNextMultiple(new_size, block_size);
    wanted_capacity = new_size / element_size;
    if (wanted_capacity < orig_capacity)
      wanted_capacity = orig_capacity;
    return wanted_capacity;
  }

  void notifyDoAllocate(void* ptr)
  {
    ++m_nb_allocate;
    if (m_do_block_allocate) {
      uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
      if ((addr % m_block_size) != 0) {
        ++m_nb_unaligned_allocate;
      }
    }
  }

 private:

  //! Taille d'un bloc. L'allocation sera un multiple de cette taille
  Int64 m_block_size = 128;
  //! Indique si l'allocation en utilisant \a m_block_size
  bool m_do_block_allocate = true;
  //! Nombre d'allocations
  std::atomic<Int32> m_nb_allocate = 0;
  //! Nombre d'allocations non alignées
  std::atomic<Int32> m_nb_unaligned_allocate = 0;

 private:

  // Calcule la plus petite valeur de \a multiple de \a multiple
  static Int64 _computeNextMultiple(Int64 n, Int64 multiple)
  {
    Int64 new_n = n / multiple;
    if ((n % multiple) != 0)
      ++new_n;
    return (new_n * multiple);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour accélérateur.
 */
class ARCCORE_COMMON_EXPORT AcceleratorMemoryAllocatorBase
: public AlignedMemoryAllocator
{
 public:

  using IMemoryPoolAllocator = Arcane::impl::IMemoryPoolAllocator;
  using BaseClass = AlignedMemoryAllocator;

 public:

  //! Liste des flags pour le pool mémoire à activer
  enum class MemoryPoolFlags
  {
    UVM = 1,
    Device = 2,
    HostPinned = 4
  };

 public:

  class IUnderlyingAllocator
  : public IMemoryPoolAllocator
  {
   public:

    virtual void doMemoryCopy(void* destination, const void* source, Int64 size) = 0;
    virtual eMemoryResource memoryResource() const = 0;
  };

 public:

  AcceleratorMemoryAllocatorBase(const String& allocator_name, IUnderlyingAllocator* underlying_allocator);

 public:

  void finalize(ITraceMng* tm);

 public:

  bool hasRealloc(MemoryAllocationArgs) const final { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) final
  {
    void* out = m_sub_allocator->allocateMemory(new_size);
    m_block_wrapper.notifyDoAllocate(out);
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCCORE_FATAL("Bad alignment for Accelerator allocator: offset={0}", (a % 128));
    m_tracer.traceAllocate(out, new_size, args);
    _applyHint(out, new_size, args);
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_info, Int64 new_size) final;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo mem_info) final
  {
    void* ptr = mem_info.baseAddress();
    size_t mem_size = mem_info.capacity();
    if (m_use_memory_pool)
      _removeHint(ptr, mem_size, args);
    // Ne lève pas d'exception en cas d'erreurs lors de la désallocation
    // car elles ont souvent lieu dans les destructeurs et cela provoque
    // un arrêt du code par std::terminate().
    m_tracer.traceDeallocate(mem_info, args);
    m_sub_allocator->freeMemory(ptr, mem_size);
  }

  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const final
  {
    wanted_capacity = AlignedMemoryAllocator::adjustedCapacity(args, wanted_capacity, element_size);
    return m_block_wrapper.adjustedCapacity(wanted_capacity, element_size);
  }
  eMemoryResource memoryResource() const final { return m_direct_sub_allocator->memoryResource(); }

 protected:

  virtual void _applyHint([[maybe_unused]] void* ptr, [[maybe_unused]] size_t new_size,
                          [[maybe_unused]] MemoryAllocationArgs args) {}
  virtual void _removeHint([[maybe_unused]] void* ptr, [[maybe_unused]] size_t new_size,
                           [[maybe_unused]] MemoryAllocationArgs args) {}

 private:

  impl::MemoryTracerWrapper m_tracer;
  std::unique_ptr<IUnderlyingAllocator> m_direct_sub_allocator;
  Arcane::impl::MemoryPool m_memory_pool;
  IMemoryPoolAllocator* m_sub_allocator = nullptr;
  bool m_use_memory_pool = false;
  String m_allocator_name;
  std::atomic<Int32> m_nb_reallocate = 0;
  std::atomic<Int64> m_reallocate_size = 0;
  Int32 m_print_level = 0;
  BlockAllocatorWrapper m_block_wrapper;

 protected:

  //! Initialisation pour la mémoire UVM
  void _doInitializeUVM(bool default_use_memory_pool = false);
  //! Initialisation pour la mémoire HostPinned
  void _doInitializeHostPinned(bool default_use_memory_pool = false);
  //! Initialisation pour la mémoire Device
  void _doInitializeDevice(bool default_use_memory_pool = false);

 protected:

  void _setTraceLevel(Int32 v) { m_tracer.setTraceLevel(v); }

 private:

  // IMPORTANT: doit être appelé avant toute allocation et ne plus être modifié ensuite.
  void _setUseMemoryPool(bool is_used);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
