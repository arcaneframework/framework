// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAccelerator.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Backend 'CUDA' pour les accélérateurs.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/cuda/CudaAccelerator.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IMemoryAllocator.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/internal/MemoryPool.h"

#include "arcane/accelerator/core/internal/MemoryTracer.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Cuda
{
using namespace Arcane::impl;
using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste des flags pour le pool mémoire à activer
enum class MemoryPoolFlags
{
  UVM = 1,
  Device = 2,
  HostPinned = 4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void arcaneCheckCudaErrors(const TraceInfo& ti, cudaError_t e)
{
  if (e != cudaSuccess)
    ARCANE_FATAL("CUDA Error trace={0} e={1} str={2}", ti, e, cudaGetErrorString(e));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void arcaneCheckCudaErrorsNoThrow(const TraceInfo& ti, cudaError_t e)
{
  if (e == cudaSuccess)
    return;
  String str = String::format("CUDA Error trace={0} e={1} str={2}", ti, e, cudaGetErrorString(e));
  FatalErrorException ex(ti, str);
  ex.write(std::cerr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
class BlockAllocatorWrapper
{
 public:

  void initialize(Int64 block_size, bool do_block_alloc)
  {
    m_block_size = block_size;
    if (m_block_size <= 0)
      m_block_size = 128;
    m_do_block_allocate = do_block_alloc;
  }

  void dumpStats(std::ostream& ostr, const String& name)
  {
    ostr << "Allocator '" << name << "' : nb_allocate=" << m_nb_allocate
         << " nb_unaligned=" << m_nb_unaligned_allocate
         << "\n";
  }

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

  void doAllocate(void* ptr, [[maybe_unused]] size_t new_size)
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
 * \brief Classe de base d'un allocateur spécifique pour 'Cuda'.
 */
class CudaMemoryAllocatorBase
: public Arccore::AlignedMemoryAllocator3
{
 public:

  using BaseClass = Arccore::AlignedMemoryAllocator3;

 public:

  class ConcreteAllocator
  {
   public:

    virtual ~ConcreteAllocator() = default;

   public:

    virtual cudaError_t _allocate(void** ptr, size_t new_size) = 0;
    virtual cudaError_t _deallocate(void* ptr) = 0;
  };

  class UnderlyingAllocator
  : public IMemoryPoolAllocator
  {
   public:

    explicit UnderlyingAllocator(CudaMemoryAllocatorBase* v)
    : m_base(v)
    {
    }

   public:

    void* allocateMemory(size_t size) override
    {
      void* out = nullptr;
      ARCANE_CHECK_CUDA(m_base->m_concrete_allocator->_allocate(&out, size));
      m_base->m_block_wrapper.doAllocate(out, size);
      return out;
    }
    void freeMemory(void* ptr, [[maybe_unused]] size_t size) override
    {
      ARCANE_CHECK_CUDA_NOTHROW(m_base->m_concrete_allocator->_deallocate(ptr));
    }

   public:

    CudaMemoryAllocatorBase* m_base = nullptr;
  };

 public:

  CudaMemoryAllocatorBase(const String& allocator_name, ConcreteAllocator* concrete_allocator)
  : AlignedMemoryAllocator3(128)
  , m_concrete_allocator(concrete_allocator)
  , m_direct_sub_allocator(this)
  , m_memory_pool(&m_direct_sub_allocator, allocator_name)
  , m_sub_allocator(&m_direct_sub_allocator)
  , m_allocator_name(allocator_name)
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_PRINT_LEVEL", true))
      m_print_level = v.value();
  }

  ~CudaMemoryAllocatorBase()
  {
  }

 public:

  void finalize(ITraceMng* tm)
  {
    if (m_print_level >= 1) {
      OStringStream ostr;
      if (m_use_memory_pool) {
        m_memory_pool.dumpStats(ostr());
        m_memory_pool.dumpFreeMap(ostr());
      }
      ostr() << "Allocator '" << m_allocator_name << "' nb_realloc=" << m_nb_reallocate
             << " realloc_copy=" << m_reallocate_size << "\n";
      m_block_wrapper.dumpStats(ostr(), m_allocator_name);
      if (tm)
        tm->info() << ostr.str();
      else
        std::cout << ostr.str();
    }

    m_memory_pool.freeCachedMemory();
  }

 public:

  bool hasRealloc(MemoryAllocationArgs) const final { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) final
  {
    void* out = m_sub_allocator->allocateMemory(new_size);
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCANE_FATAL("Bad alignment for CUDA allocator: offset={0}", (a % 128));
    m_tracer.traceAllocate(out, new_size, args);
    _applyHint(out, new_size, args);
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_info, Int64 new_size) final
  {
    ++m_nb_reallocate;
    Int64 current_size = current_info.size();
    m_reallocate_size += current_size;
    String array_name = args.arrayName();
    const bool do_print = (m_print_level >= 2);
    if (do_print) {
      std::cout << "Reallocate allocator=" << m_allocator_name
                << " current_size=" << current_size
                << " current_capacity=" << current_info.capacity()
                << " new_capacity=" << new_size
                << " ptr=" << current_info.baseAddress();
      if (array_name.null() && m_print_level >= 3) {
        std::cout << " stack=" << platform::getStackTrace();
      }
      else {
        std::cout << " name=" << array_name;
        if (m_print_level >= 4)
          std::cout << " stack=" << platform::getStackTrace();
      }
      std::cout << "\n";
    }
    if (m_use_memory_pool)
      _removeHint(current_info.baseAddress(), current_size, args);
    AllocatedMemoryInfo a = allocate(args, new_size);
    // TODO: supprimer les Hint après le deallocate car la zone mémoire peut être réutilisée.
    ARCANE_CHECK_CUDA(cudaMemcpy(a.baseAddress(), current_info.baseAddress(), current_size, cudaMemcpyDefault));
    deallocate(args, current_info);
    return a;
  }
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
    wanted_capacity = AlignedMemoryAllocator3::adjustedCapacity(args, wanted_capacity, element_size);
    return m_block_wrapper.adjustedCapacity(wanted_capacity, element_size);
  }

 protected:

  virtual void _applyHint([[maybe_unused]] void* ptr, [[maybe_unused]] size_t new_size,
                          [[maybe_unused]] MemoryAllocationArgs args) {}
  virtual void _removeHint([[maybe_unused]] void* ptr, [[maybe_unused]] size_t new_size,
                           [[maybe_unused]] MemoryAllocationArgs args) {}

 private:

  impl::MemoryTracerWrapper m_tracer;
  std::unique_ptr<ConcreteAllocator> m_concrete_allocator;
  UnderlyingAllocator m_direct_sub_allocator;
  MemoryPool m_memory_pool;
  IMemoryPoolAllocator* m_sub_allocator = nullptr;
  bool m_use_memory_pool = false;
  String m_allocator_name;
  std::atomic<Int32> m_nb_reallocate = 0;
  std::atomic<Int64> m_reallocate_size = 0;
  Int32 m_print_level = 0;

 protected:

  BlockAllocatorWrapper m_block_wrapper;

 protected:

  void _setTraceLevel(Int32 v) { m_tracer.setTraceLevel(v); }
  // IMPORTANT: doit être appelé avant toute allocation et ne plus être modifié ensuite.
  void _setUseMemoryPool(bool is_used)
  {
    IMemoryPoolAllocator* mem_pool = &m_memory_pool;
    IMemoryPoolAllocator* direct = &m_direct_sub_allocator;
    m_sub_allocator = (is_used) ? mem_pool : direct;
    m_use_memory_pool = is_used;
    if (is_used) {
      if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL_MAX_BLOCK_SIZE", true)) {
        if (v.value() < 0)
          ARCANE_FATAL("Invalid value '{0}' for memory pool max block size");
        size_t block_size = static_cast<size_t>(v.value());
        m_memory_pool.setMaxCachedBlockSize(block_size);
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur pour la mémoire unifiée.
 *
 * Pour éviter des effets de bord du driver NVIDIA qui effectue les transferts
 * entre le CPU et le GPU par page. on alloue la mémoire par bloc multiple
 * de la taille d'une page.
 */
class UnifiedMemoryCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 public:

  class Allocator
  : public ConcreteAllocator
  {
   public:

    Allocator()
    {
      if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_USE_ALLOC_ATS", true))
        m_use_ats = v.value();
      if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_MEMORY_HINT_ON_DEVICE", true))
        m_use_hint_as_mainly_device = (v.value() != 0);
    }

    cudaError_t _deallocate(void* ptr) final
    {
      if (m_use_ats) {
        ::free(ptr);
        return cudaSuccess;
      }
      //std::cout << "CUDA_MANAGED_FREE ptr=" << ptr << "\n";
      return ::cudaFree(ptr);
    }

    cudaError_t _allocate(void** ptr, size_t new_size) final
    {
      if (m_use_ats) {
        *ptr = ::aligned_alloc(128, new_size);
      }
      else {
        auto r = ::cudaMallocManaged(ptr, new_size, cudaMemAttachGlobal);
        //std::cout << "CUDA_MANAGED_MALLOC ptr=" << (*ptr) << " size=" << new_size << "\n";
        //if (new_size < 4000)
        //std::cout << "STACK=" << platform::getStackTrace() << "\n";

        if (r != cudaSuccess)
          return r;

        // Si demandé, indique qu'on préfère allouer sur le GPU.
        // NOTE: Dans ce cas, on récupère le device actuel pour positionner la localisation
        // préférée. Dans le cas où on utilise MemoryPool, cette allocation ne sera effectuée
        // qu'une seule fois. Si le device par défaut pour un thread change au cours du calcul
        // il y aura une incohérence. Pour éviter cela, on pourrait faire un cudaMemAdvise()
        // pour chaque allocation (via _applyHint()) mais ces opérations sont assez couteuses
        // et s'il y a beaucoup d'allocation il peut en résulter une perte de performance.
        if (m_use_hint_as_mainly_device) {
          int device_id = 0;
          void* p = *ptr;
          cudaGetDevice(&device_id);
          ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, _getMemoryLocation(device_id)));
          ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, _getMemoryLocation(cudaCpuDeviceId)));
        }
      }

      return cudaSuccess;
    }

   public:

    bool m_use_ats = false;
    //! Si vrai, par défaut on considère toutes les allocations comme eMemoryLocationHint::MainlyDevice
    bool m_use_hint_as_mainly_device = false;
  };

 public:

  UnifiedMemoryCudaMemoryAllocator()
  : CudaMemoryAllocatorBase("UnifiedMemoryCudaMemory", new Allocator())
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_MALLOC_TRACE", true))
      _setTraceLevel(v.value());
  }

  void initialize()
  {
    bool do_page_allocate = true;
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_UM_PAGE_ALLOC", true))
      do_page_allocate = (v.value() != 0);
    Int64 page_size = platform::getPageSize();
    m_block_wrapper.initialize(page_size, do_page_allocate);

    bool use_memory_pool = false;
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL", true))
      use_memory_pool = (v.value() & static_cast<int>(MemoryPoolFlags::UVM)) != 0;
    _setUseMemoryPool(use_memory_pool);
  }

 public:

  void notifyMemoryArgsChanged([[maybe_unused]] MemoryAllocationArgs old_args,
                               MemoryAllocationArgs new_args, AllocatedMemoryInfo ptr) final
  {
    void* p = ptr.baseAddress();
    Int64 s = ptr.capacity();
    if (p && s > 0)
      _applyHint(ptr.baseAddress(), ptr.size(), new_args);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::UnifiedMemory; }

 protected:

  void _applyHint(void* p, size_t new_size, MemoryAllocationArgs args)
  {
    eMemoryLocationHint hint = args.memoryLocationHint();
    // Utilise le device actif pour positionner le GPU par défaut
    // On ne le fait que si le \a hint le nécessite pour éviter d'appeler
    // cudaGetDevice() à chaque fois.
    int device_id = 0;
    if (hint == eMemoryLocationHint::MainlyDevice || hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      cudaGetDevice(&device_id);
    }
    auto device_memory_location = _getMemoryLocation(device_id);
    auto cpu_memory_location = _getMemoryLocation(cudaCpuDeviceId);

    //std::cout << "SET_MEMORY_HINT name=" << args.arrayName() << " size=" << new_size << " hint=" << (int)hint << "\n";
    if (hint == eMemoryLocationHint::MainlyDevice || hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, device_memory_location));
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, cpu_memory_location));
    }
    if (hint == eMemoryLocationHint::MainlyHost) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetPreferredLocation, cpu_memory_location));
      //ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetAccessedBy, 0));
    }
    if (hint == eMemoryLocationHint::HostAndDeviceMostlyRead) {
      ARCANE_CHECK_CUDA(cudaMemAdvise(p, new_size, cudaMemAdviseSetReadMostly, device_memory_location));
    }
  }
  void _removeHint(void* p, size_t size, MemoryAllocationArgs args)
  {
    eMemoryLocationHint hint = args.memoryLocationHint();
    if (hint == eMemoryLocationHint::None)
      return;
    int device_id = 0;
    ARCANE_CHECK_CUDA(cudaMemAdvise(p, size, cudaMemAdviseUnsetReadMostly, _getMemoryLocation(device_id)));
  }

 private:

  bool m_use_ats = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{
 public:

  class Allocator
  : public ConcreteAllocator
  {
   public:

    cudaError_t _allocate(void** ptr, size_t new_size) final
    {
      return ::cudaMallocHost(ptr, new_size);
    }
    cudaError_t _deallocate(void* ptr) final
    {
      return ::cudaFreeHost(ptr);
    }
  };

 public:

  HostPinnedCudaMemoryAllocator()
  : CudaMemoryAllocatorBase("HostPinnedCudaMemory", new Allocator())
  {
  }

 public:

  void initialize()
  {
    bool use_memory_pool = false;
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL", true))
      use_memory_pool = (v.value() & static_cast<int>(MemoryPoolFlags::HostPinned)) != 0;
    _setUseMemoryPool(use_memory_pool);
    m_block_wrapper.initialize(128, use_memory_pool);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceCudaMemoryAllocator
: public CudaMemoryAllocatorBase
{

  class Allocator
  : public ConcreteAllocator
  {
   public:

    Allocator()
    {
      if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_USE_ALLOC_ATS", true))
        m_use_ats = v.value();
    }

    cudaError_t _allocate(void** ptr, size_t new_size) final
    {
      if (m_use_ats) {
        // FIXME: it does not work on WIN32
        *ptr = std::aligned_alloc(128, new_size);
        if (*ptr)
          return cudaSuccess;
        return cudaErrorMemoryAllocation;
      }
      cudaError_t r = ::cudaMalloc(ptr, new_size);
      //std::cout << "ALLOCATE_DEVICE ptr=" << (*ptr) << " size=" << new_size << " r=" << (int)r << "\n";
      return r;
    }
    cudaError_t _deallocate(void* ptr) final
    {
      if (m_use_ats) {
        std::free(ptr);
        return cudaSuccess;
      }
      //std::cout << "FREE_DEVICE ptr=" << ptr << "\n";
      return ::cudaFree(ptr);
    }

   private:

    bool m_use_ats = false;
  };

 public:

  DeviceCudaMemoryAllocator()
  : CudaMemoryAllocatorBase("DeviceCudaMemoryAllocator", new Allocator())
  {
  }

 public:

  void initialize()
  {
    bool use_memory_pool = false;
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_MEMORY_POOL", true))
      use_memory_pool = (v.value() & static_cast<int>(MemoryPoolFlags::Device)) != 0;
    _setUseMemoryPool(use_memory_pool);
    m_block_wrapper.initialize(128, use_memory_pool);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::Device; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  UnifiedMemoryCudaMemoryAllocator unified_memory_cuda_memory_allocator;
  HostPinnedCudaMemoryAllocator host_pinned_cuda_memory_allocator;
  DeviceCudaMemoryAllocator device_cuda_memory_allocator;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::IMemoryAllocator*
getCudaMemoryAllocator()
{
  return &unified_memory_cuda_memory_allocator;
}

Arccore::IMemoryAllocator*
getCudaDeviceMemoryAllocator()
{
  return &device_cuda_memory_allocator;
}

Arccore::IMemoryAllocator*
getCudaUnifiedMemoryAllocator()
{
  return &unified_memory_cuda_memory_allocator;
}

Arccore::IMemoryAllocator*
getCudaHostPinnedMemoryAllocator()
{
  return &host_pinned_cuda_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void initializeCudaMemoryAllocators()
{
  unified_memory_cuda_memory_allocator.initialize();
  device_cuda_memory_allocator.initialize();
  host_pinned_cuda_memory_allocator.initialize();
}

void finalizeCudaMemoryAllocators(ITraceMng* tm)
{
  unified_memory_cuda_memory_allocator.finalize(tm);
  device_cuda_memory_allocator.finalize(tm);
  host_pinned_cuda_memory_allocator.finalize(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
