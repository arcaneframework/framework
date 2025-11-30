// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAcceleratorRuntime.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Runtime pour 'Cuda'.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator_native/CudaAccelerator.h"

#include "arccore/base/CheckedConvert.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/common/internal/MemoryUtilsInternal.h"
#include "arccore/common/internal/IMemoryResourceMngInternal.h"

#include "arccore/common/accelerator/RunQueueBuildInfo.h"
#include "arccore/common/accelerator/Memory.h"
#include "arccore/common/accelerator/DeviceInfoList.h"
#include "arccore/common/accelerator/KernelLaunchArgs.h"
#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/accelerator/DeviceMemoryInfo.h"
#include "arccore/common/accelerator/NativeStream.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"
#include "arccore/common/accelerator/internal/RegisterRuntimeInfo.h"
#include "arccore/common/accelerator/internal/RunCommandImpl.h"
#include "arccore/common/accelerator/internal/IRunQueueStream.h"
#include "arccore/common/accelerator/internal/IRunQueueEventImpl.h"
#include "arccore/common/accelerator/internal/AcceleratorMemoryAllocatorBase.h"

#include "arccore/accelerator_native/runtime/Cupti.h"

#include <sstream>
#include <unordered_map>
#include <mutex>

#include <cuda.h>

// Pour std::memset
#include <cstring>

#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
#include <nvtx3/nvToolsExt.h>
#endif

namespace Arcane::Accelerator::Cuda
{
using Impl::KernelLaunchArgs;

namespace
{
  Int32 global_cupti_flush = 0;
  CuptiInfo global_cupti_info;
} // namespace


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// A partir de CUDA 13, il y a un nouveau type cudaMemLocation
// pour les méthodes telles cudeMemAdvise ou cudaMemPrefetch
#if defined(ARCANE_USING_CUDA13_OR_GREATER)
inline cudaMemLocation
_getMemoryLocation(int device_id)
{
  cudaMemLocation mem_location;
  mem_location.type = cudaMemLocationTypeDevice;
  mem_location.id = device_id;
  if (device_id == cudaCpuDeviceId)
    mem_location.type = cudaMemLocationTypeHost;
  else {
    mem_location.type = cudaMemLocationTypeDevice;
    mem_location.id = device_id;
  }
  return mem_location;
}
#else
inline int
_getMemoryLocation(int device_id)
{
  return device_id;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConcreteAllocator
{
 public:

  virtual ~ConcreteAllocator() = default;

 public:

  virtual cudaError_t _allocate(void** ptr, size_t new_size) = 0;
  virtual cudaError_t _deallocate(void* ptr) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ConcreteAllocatorType>
class UnderlyingAllocator
: public AcceleratorMemoryAllocatorBase::IUnderlyingAllocator
{
 public:

  UnderlyingAllocator() = default;

 public:

  void* allocateMemory(size_t size) final
  {
    void* out = nullptr;
    ARCANE_CHECK_CUDA(m_concrete_allocator._allocate(&out, size));
    return out;
  }
  void freeMemory(void* ptr, [[maybe_unused]] size_t size) final
  {
    ARCANE_CHECK_CUDA_NOTHROW(m_concrete_allocator._deallocate(ptr));
  }

  void doMemoryCopy(void* destination, const void* source, Int64 size) final
  {
    ARCANE_CHECK_CUDA(cudaMemcpy(destination, source, size, cudaMemcpyDefault));
  }

  eMemoryResource memoryResource() const final
  {
    return m_concrete_allocator.memoryResource();
  }

 public:

  ConcreteAllocatorType m_concrete_allocator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryConcreteAllocator
: public ConcreteAllocator
{
 public:

  UnifiedMemoryConcreteAllocator()
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

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::UnifiedMemory; }

 public:

  bool m_use_ats = false;
  //! Si vrai, par défaut on considère toutes les allocations comme eMemoryLocationHint::MainlyDevice
  bool m_use_hint_as_mainly_device = false;
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
: public AcceleratorMemoryAllocatorBase
{
 public:
 public:

  UnifiedMemoryCudaMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("UnifiedMemoryCudaMemory", new UnderlyingAllocator<UnifiedMemoryConcreteAllocator>())
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUDA_MALLOC_TRACE", true))
      _setTraceLevel(v.value());
  }

  void initialize()
  {
    _doInitializeUVM(true);
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

class HostPinnedConcreteAllocator
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
  constexpr eMemoryResource memoryResource() const { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedCudaMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:
 public:

  HostPinnedCudaMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("HostPinnedCudaMemory", new UnderlyingAllocator<HostPinnedConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    _doInitializeHostPinned(true);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceConcreteAllocator
: public ConcreteAllocator
{
 public:

  DeviceConcreteAllocator()
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

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::Device; }

 private:

  bool m_use_ats = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceCudaMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{

 public:

  DeviceCudaMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("DeviceCudaMemoryAllocator", new UnderlyingAllocator<DeviceConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    _doInitializeDevice(true);
  }
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

void
initializeCudaMemoryAllocators()
{
  unified_memory_cuda_memory_allocator.initialize();
  device_cuda_memory_allocator.initialize();
  host_pinned_cuda_memory_allocator.initialize();
}

void
finalizeCudaMemoryAllocators(ITraceMng* tm)
{
  unified_memory_cuda_memory_allocator.finalize(tm);
  device_cuda_memory_allocator.finalize(tm);
  host_pinned_cuda_memory_allocator.finalize(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
arcaneCheckCudaErrors(const TraceInfo& ti, CUresult e)
{
  if (e == CUDA_SUCCESS)
    return;
  const char* error_name = nullptr;
  CUresult e2 = cuGetErrorName(e, &error_name);
  if (e2 != CUDA_SUCCESS)
    error_name = "Unknown";

  const char* error_message = nullptr;
  CUresult e3 = cuGetErrorString(e, &error_message);
  if (e3 != CUDA_SUCCESS)
    error_message = "Unknown";

  ARCCORE_FATAL("CUDA Error trace={0} e={1} name={2} message={3}",
                ti, e, error_name, error_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Map contenant l'occupation idéale pour un kernel donné.
 *
 * \note Pour l'instant, on ne supporte pas d'avoir une valeur non nulle
 * pour la quantité de mémoire partagée.
 *
 * En cas d'erreur dans le calcul, on retourne une valeur de zéro.
 */
class OccupancyMap
{
 public:

  Int32 getNbThreadPerBlock(const void* kernel_ptr)
  {
    std::scoped_lock lock(m_mutex);
    auto x = m_nb_thread_per_block_map.find(kernel_ptr);
    if (x != m_nb_thread_per_block_map.end())
      return x->second;
    int min_grid_size = 0;
    int computed_block_size = 0;
    int wanted_shared_memory = 0;
    cudaError_t r = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &computed_block_size, kernel_ptr, wanted_shared_memory);
    if (r != cudaSuccess)
      computed_block_size = 0;
    int num_block_0 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_block_0, kernel_ptr, 256, wanted_shared_memory);
    int num_block_1 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_block_1, kernel_ptr, 1024, wanted_shared_memory);

    cudaFuncAttributes func_attr;
    cudaFuncGetAttributes(&func_attr, kernel_ptr);
    m_nb_thread_per_block_map[kernel_ptr] = computed_block_size;
    std::cout << "ComputedBlockSize=" << computed_block_size << " n0=" << num_block_0 << " n1=" << num_block_1
              << " min_grid_size=" << min_grid_size << " nb_reg=" << func_attr.numRegs;

#if CUDART_VERSION >= 12040
    // cudaFuncGetName is only available in 12.4
    const char* func_name = nullptr;
    cudaFuncGetName(&func_name, kernel_ptr);
    std::cout << " name=" << func_name << "\n";
#endif

    return computed_block_size;
  }

 private:

  std::unordered_map<const void*, Int32> m_nb_thread_per_block_map;
  std::mutex m_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaRunQueueStream
: public impl::IRunQueueStream
{
 public:

  CudaRunQueueStream(impl::IRunnerRuntime* runtime, const RunQueueBuildInfo& bi)
  : m_runtime(runtime)
  {
    if (bi.isDefault())
      ARCANE_CHECK_CUDA(cudaStreamCreate(&m_cuda_stream));
    else {
      int priority = bi.priority();
      ARCANE_CHECK_CUDA(cudaStreamCreateWithPriority(&m_cuda_stream, cudaStreamDefault, priority));
    }
  }
  ~CudaRunQueueStream() override
  {
    ARCANE_CHECK_CUDA_NOTHROW(cudaStreamDestroy(m_cuda_stream));
  }

 public:

  void notifyBeginLaunchKernel([[maybe_unused]] impl::RunCommandImpl& c) override
  {
#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
    auto kname = c.kernelName();
    if (kname.empty())
      nvtxRangePush(c.traceInfo().name());
    else
      nvtxRangePush(kname.localstr());
#endif
    return m_runtime->notifyBeginLaunchKernel();
  }
  void notifyEndLaunchKernel(impl::RunCommandImpl&) override
  {
#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
    nvtxRangePop();
#endif
    return m_runtime->notifyEndLaunchKernel();
  }
  void barrier() override
  {
    ARCANE_CHECK_CUDA(cudaStreamSynchronize(m_cuda_stream));
    if (global_cupti_flush > 0)
      global_cupti_info.flush();
  }
  bool _barrierNoException() override
  {
    return (cudaStreamSynchronize(m_cuda_stream) != cudaSuccess);
  }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    auto source_bytes = args.source().bytes();
    auto r = cudaMemcpyAsync(args.destination().data(), source_bytes.data(),
                             source_bytes.size(), cudaMemcpyDefault, m_cuda_stream);
    ARCANE_CHECK_CUDA(r);
    if (!args.isAsync())
      barrier();
  }
  void prefetchMemory(const MemoryPrefetchArgs& args) override
  {
    auto src = args.source().bytes();
    if (src.size() == 0)
      return;
    DeviceId d = args.deviceId();
    int device = cudaCpuDeviceId;
    if (!d.isHost())
      device = d.asInt32();
    //std::cout << "PREFETCH device=" << device << " host(id)=" << cudaCpuDeviceId
    //          << " size=" << args.source().size() << " data=" << src.data() << "\n";
    auto mem_location = _getMemoryLocation(device);
#if defined(ARCANE_USING_CUDA13_OR_GREATER)
    auto r = cudaMemPrefetchAsync(src.data(), src.size(), mem_location, 0, m_cuda_stream);
#else
    auto r = cudaMemPrefetchAsync(src.data(), src.size(), mem_location, m_cuda_stream);
#endif
    ARCANE_CHECK_CUDA(r);
    if (!args.isAsync())
      barrier();
  }
  Impl::NativeStream nativeStream() override
  {
    return Impl::NativeStream(&m_cuda_stream);
  }

 public:

  cudaStream_t trueStream() const
  {
    return m_cuda_stream;
  }

 private:

  impl::IRunnerRuntime* m_runtime = nullptr;
  cudaStream_t m_cuda_stream = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaRunQueueEvent
: public impl::IRunQueueEventImpl
{
 public:

  explicit CudaRunQueueEvent(bool has_timer)
  {
    if (has_timer)
      ARCANE_CHECK_CUDA(cudaEventCreate(&m_cuda_event));
    else
      ARCANE_CHECK_CUDA(cudaEventCreateWithFlags(&m_cuda_event, cudaEventDisableTiming));
  }
  ~CudaRunQueueEvent() override
  {
    ARCANE_CHECK_CUDA_NOTHROW(cudaEventDestroy(m_cuda_event));
  }

 public:

  // Enregistre l'événement au sein d'une RunQueue
  void recordQueue(impl::IRunQueueStream* stream) final
  {
    auto* rq = static_cast<CudaRunQueueStream*>(stream);
    ARCANE_CHECK_CUDA(cudaEventRecord(m_cuda_event, rq->trueStream()));
  }

  void wait() final
  {
    ARCANE_CHECK_CUDA(cudaEventSynchronize(m_cuda_event));
  }

  void waitForEvent(impl::IRunQueueStream* stream) final
  {
    auto* rq = static_cast<CudaRunQueueStream*>(stream);
    ARCANE_CHECK_CUDA(cudaStreamWaitEvent(rq->trueStream(), m_cuda_event, cudaEventWaitDefault));
  }

  Int64 elapsedTime(IRunQueueEventImpl* start_event) final
  {
    // NOTE: Les évènements doivent avoir été créé avec le timer actif
    ARCCORE_CHECK_POINTER(start_event);
    auto* true_start_event = static_cast<CudaRunQueueEvent*>(start_event);
    float time_in_ms = 0.0;

    // TODO: regarder si nécessaire
    // ARCANE_CHECK_CUDA(cudaEventSynchronize(m_cuda_event));

    ARCANE_CHECK_CUDA(cudaEventElapsedTime(&time_in_ms, true_start_event->m_cuda_event, m_cuda_event));
    double x = time_in_ms * 1.0e6;
    Int64 nano_time = static_cast<Int64>(x);
    return nano_time;
  }

  bool hasPendingWork() final
  {
    cudaError_t v = cudaEventQuery(m_cuda_event);
    if (v == cudaErrorNotReady)
      return true;
    ARCANE_CHECK_CUDA(v);
    return false;
  }

 private:

  cudaEvent_t m_cuda_event;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaRunnerRuntime
: public impl::IRunnerRuntime
{
 public:

  ~CudaRunnerRuntime() override = default;

 public:

  void notifyBeginLaunchKernel() override
  {
    ++m_nb_kernel_launched;
    if (m_is_verbose)
      std::cout << "BEGIN CUDA KERNEL!\n";
  }
  void notifyEndLaunchKernel() override
  {
    ARCANE_CHECK_CUDA(cudaGetLastError());
    if (m_is_verbose)
      std::cout << "END CUDA KERNEL!\n";
  }
  void barrier() override
  {
    ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
  }
  eExecutionPolicy executionPolicy() const override
  {
    return eExecutionPolicy::CUDA;
  }
  impl::IRunQueueStream* createStream(const RunQueueBuildInfo& bi) override
  {
    return new CudaRunQueueStream(this, bi);
  }
  impl::IRunQueueEventImpl* createEventImpl() override
  {
    return new CudaRunQueueEvent(false);
  }
  impl::IRunQueueEventImpl* createEventImplWithTimer() override
  {
    return new CudaRunQueueEvent(true);
  }
  void setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    auto v = buffer.bytes();
    const void* ptr = v.data();
    size_t count = v.size();
    int device = device_id.asInt32();
    cudaMemoryAdvise cuda_advise;

    if (advice == eMemoryAdvice::MostlyRead)
      cuda_advise = cudaMemAdviseSetReadMostly;
    else if (advice == eMemoryAdvice::PreferredLocationDevice)
      cuda_advise = cudaMemAdviseSetPreferredLocation;
    else if (advice == eMemoryAdvice::AccessedByDevice)
      cuda_advise = cudaMemAdviseSetAccessedBy;
    else if (advice == eMemoryAdvice::PreferredLocationHost) {
      cuda_advise = cudaMemAdviseSetPreferredLocation;
      device = cudaCpuDeviceId;
    }
    else if (advice == eMemoryAdvice::AccessedByHost) {
      cuda_advise = cudaMemAdviseSetAccessedBy;
      device = cudaCpuDeviceId;
    }
    else
      return;
    //std::cout << "MEMADVISE p=" << ptr << " size=" << count << " advise = " << cuda_advise << " id = " << device << "\n";
    ARCANE_CHECK_CUDA(cudaMemAdvise(ptr, count, cuda_advise, _getMemoryLocation(device)));
  }
  void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    auto v = buffer.bytes();
    const void* ptr = v.data();
    size_t count = v.size();
    int device = device_id.asInt32();
    cudaMemoryAdvise cuda_advise;

    if (advice == eMemoryAdvice::MostlyRead)
      cuda_advise = cudaMemAdviseUnsetReadMostly;
    else if (advice == eMemoryAdvice::PreferredLocationDevice)
      cuda_advise = cudaMemAdviseUnsetPreferredLocation;
    else if (advice == eMemoryAdvice::AccessedByDevice)
      cuda_advise = cudaMemAdviseUnsetAccessedBy;
    else if (advice == eMemoryAdvice::PreferredLocationHost) {
      cuda_advise = cudaMemAdviseUnsetPreferredLocation;
      device = cudaCpuDeviceId;
    }
    else if (advice == eMemoryAdvice::AccessedByHost) {
      cuda_advise = cudaMemAdviseUnsetAccessedBy;
      device = cudaCpuDeviceId;
    }
    else
      return;
    ARCANE_CHECK_CUDA(cudaMemAdvise(ptr, count, cuda_advise, _getMemoryLocation(device)));
  }

  void setCurrentDevice(DeviceId device_id) final
  {
    Int32 id = device_id.asInt32();
    if (!device_id.isAccelerator())
      ARCCORE_FATAL("Device {0} is not an accelerator device", id);
    ARCANE_CHECK_CUDA(cudaSetDevice(id));
  }

  const IDeviceInfoList* deviceInfoList() final { return &m_device_info_list; }

  void startProfiling() override
  {
    global_cupti_info.start();
  }

  void stopProfiling() override
  {
    global_cupti_info.stop();
  }

  bool isProfilingActive() override
  {
    return global_cupti_info.isActive();
  }

  void getPointerAttribute(PointerAttribute& attribute, const void* ptr) override
  {
    cudaPointerAttributes ca;
    ARCANE_CHECK_CUDA(cudaPointerGetAttributes(&ca, ptr));
    // NOTE: le type Arcane 'ePointerMemoryType' a normalememt les mêmes valeurs
    // que le type CUDA correspondant donc on peut faire un cast simple.
    auto mem_type = static_cast<ePointerMemoryType>(ca.type);
    _fillPointerAttribute(attribute, mem_type, ca.device,
                          ptr, ca.devicePointer, ca.hostPointer);
  }

  DeviceMemoryInfo getDeviceMemoryInfo(DeviceId device_id) override
  {
    int d = 0;
    int wanted_d = device_id.asInt32();
    ARCANE_CHECK_CUDA(cudaGetDevice(&d));
    if (d != wanted_d)
      ARCANE_CHECK_CUDA(cudaSetDevice(wanted_d));
    size_t free_mem = 0;
    size_t total_mem = 0;
    ARCANE_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    if (d != wanted_d)
      ARCANE_CHECK_CUDA(cudaSetDevice(d));
    DeviceMemoryInfo dmi;
    dmi.setFreeMemory(free_mem);
    dmi.setTotalMemory(total_mem);
    return dmi;
  }

  void pushProfilerRange(const String& name, Int32 color_rgb) override
  {
#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
    if (color_rgb >= 0) {
      // NOTE: Il faudrait faire: nvtxEventAttributes_t eventAttrib = { 0 };
      // mais cela provoque pleins d'avertissement de type 'missing initializer for member'
      nvtxEventAttributes_t eventAttrib;
      std::memset(&eventAttrib, 0, sizeof(nvtxEventAttributes_t));
      eventAttrib.version = NVTX_VERSION;
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = color_rgb;
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
      eventAttrib.message.ascii = name.localstr();
      nvtxRangePushEx(&eventAttrib);
    }
    else
      nvtxRangePush(name.localstr());
#endif
  }
  void popProfilerRange() override
  {
#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
    nvtxRangePop();
#endif
  }

  void finalize(ITraceMng* tm) override
  {
    finalizeCudaMemoryAllocators(tm);
  }

  KernelLaunchArgs computeKernalLaunchArgs(const KernelLaunchArgs& orig_args,
                                           const void* kernel_ptr,
                                           Int64 total_loop_size,
                                           Int32 wanted_shared_memory) override
  {
    if (!m_use_computed_occupancy)
      return orig_args;
    if (wanted_shared_memory < 0)
      wanted_shared_memory = 0;
    // Pour l'instant, on ne fait pas de calcul si la mémoire partagée est non nulle.
    if (wanted_shared_memory != 0)
      return orig_args;
    Int32 computed_block_size = m_occupancy_map.getNbThreadPerBlock(kernel_ptr);
    if (computed_block_size == 0)
      return orig_args;
    Int64 big_b = (total_loop_size + computed_block_size - 1) / computed_block_size;
    int blocks_per_grid = CheckedConvert::toInt32(big_b);
    return { blocks_per_grid, computed_block_size };
  }

 public:

  void fillDevices(bool is_verbose);
  void build()
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_COMPUTED_OCCUPANCY", true))
      m_use_computed_occupancy = v.value();
  }

 private:

  Int64 m_nb_kernel_launched = 0;
  bool m_is_verbose = false;
  bool m_use_computed_occupancy = false;
  impl::DeviceInfoList m_device_info_list;
  OccupancyMap m_occupancy_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CudaRunnerRuntime::
fillDevices(bool is_verbose)
{
  int nb_device = 0;
  ARCANE_CHECK_CUDA(cudaGetDeviceCount(&nb_device));
  std::ostream& omain = std::cout;
  if (is_verbose)
    omain << "ArcaneCUDA: Initialize Arcane CUDA runtime nb_available_device=" << nb_device << "\n";
  for (int i = 0; i < nb_device; ++i) {
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, i);
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    std::ostringstream ostr;
    std::ostream& o = ostr;
    o << "Device " << i << " name=" << dp.name << "\n";
    o << " Driver version = " << (driver_version / 1000) << "." << (driver_version % 1000) << "\n";
    o << " Runtime version = " << (runtime_version / 1000) << "." << (runtime_version % 1000) << "\n";
    o << " computeCapability = " << dp.major << "." << dp.minor << "\n";
    o << " totalGlobalMem = " << dp.totalGlobalMem << "\n";
    o << " sharedMemPerBlock = " << dp.sharedMemPerBlock << "\n";
    o << " regsPerBlock = " << dp.regsPerBlock << "\n";
    o << " warpSize = " << dp.warpSize << "\n";
    o << " memPitch = " << dp.memPitch << "\n";
    o << " maxThreadsPerBlock = " << dp.maxThreadsPerBlock << "\n";
    o << " maxThreadsPerMultiProcessor = " << dp.maxThreadsPerMultiProcessor << "\n";
    o << " totalConstMem = " << dp.totalConstMem << "\n";
    o << " cooperativeLaunch = " << dp.cooperativeLaunch << "\n";
    o << " multiProcessorCount = " << dp.multiProcessorCount << "\n";
    o << " integrated = " << dp.integrated << "\n";
    o << " canMapHostMemory = " << dp.canMapHostMemory << "\n";
    o << " directManagedMemAccessFromHost = " << dp.directManagedMemAccessFromHost << "\n";
    o << " hostNativeAtomicSupported = " << dp.hostNativeAtomicSupported << "\n";
    o << " pageableMemoryAccess = " << dp.pageableMemoryAccess << "\n";
    o << " concurrentManagedAccess = " << dp.concurrentManagedAccess << "\n";
    o << " pageableMemoryAccessUsesHostPageTables = " << dp.pageableMemoryAccessUsesHostPageTables << "\n";
    o << " hostNativeAtomicSupported = " << dp.hostNativeAtomicSupported << "\n";
    o << " maxThreadsDim = " << dp.maxThreadsDim[0] << " " << dp.maxThreadsDim[1]
      << " " << dp.maxThreadsDim[2] << "\n";
    o << " maxGridSize = " << dp.maxGridSize[0] << " " << dp.maxGridSize[1]
      << " " << dp.maxGridSize[2] << "\n";
#if !defined(ARCANE_USING_CUDA13_OR_GREATER)
    o << " clockRate = " << dp.clockRate << "\n";
    o << " deviceOverlap = " << dp.deviceOverlap << "\n";
    o << " computeMode = " << dp.computeMode << "\n";
    o << " kernelExecTimeoutEnabled = " << dp.kernelExecTimeoutEnabled << "\n";
#endif

    {
      int least_val = 0;
      int greatest_val = 0;
      ARCANE_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&least_val, &greatest_val));
      o << " leastPriority = " << least_val << " greatestPriority = " << greatest_val << "\n";
    }
    {
      CUdevice device;
      ARCANE_CHECK_CUDA(cuDeviceGet(&device, i));
      CUuuid device_uuid;
      ARCANE_CHECK_CUDA(cuDeviceGetUuid(&device_uuid, device));
      o << " deviceUuid=";
      impl::printUUID(o, device_uuid.bytes);
      o << "\n";
    }
    String description(ostr.str());
    if (is_verbose)
      omain << description;

    DeviceInfo device_info;
    device_info.setDescription(description);
    device_info.setDeviceId(DeviceId(i));
    device_info.setName(dp.name);
    device_info.setWarpSize(dp.warpSize);
    m_device_info_list.addDevice(device_info);
  }

  Int32 global_cupti_level = 0;

  // Regarde si on active Cupti
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUPTI_LEVEL", true))
    global_cupti_level = v.value();
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUPTI_FLUSH", true))
    global_cupti_flush = v.value();
  bool do_print_cupti = true;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUPTI_PRINT", true))
    do_print_cupti = (v.value() != 0);

  if (global_cupti_level > 0) {
#ifndef ARCANE_HAS_CUDA_CUPTI
    ARCANE_FATAL("Trying to enable CUPTI but Arcane is not compiled with cupti support");
#endif
    global_cupti_info.init(global_cupti_level, do_print_cupti);
    global_cupti_info.start();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaMemoryCopier
: public IMemoryCopier
{
  void copy(ConstMemoryView from, [[maybe_unused]] eMemoryResource from_mem,
            MutableMemoryView to, [[maybe_unused]] eMemoryResource to_mem,
            const RunQueue* queue) override
  {
    if (queue) {
      queue->copyMemory(MemoryCopyArgs(to.bytes(), from.bytes()).addAsync(queue->isAsync()));
      return;
    }
    // 'cudaMemcpyDefault' sait automatiquement ce qu'il faut faire en tenant
    // uniquement compte de la valeur des pointeurs. Il faudrait voir si
    // utiliser \a from_mem et \a to_mem peut améliorer les performances.
    ARCANE_CHECK_CUDA(cudaMemcpy(to.data(), from.data(), from.bytes().size(), cudaMemcpyDefault));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Cuda

namespace
{
Arcane::Accelerator::Cuda::CudaRunnerRuntime global_cuda_runtime;
Arcane::Accelerator::Cuda::CudaMemoryCopier global_cuda_memory_copier;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette fonction est le point d'entrée utilisé lors du chargement
// dynamique de cette bibliothèque
extern "C" ARCCORE_EXPORT void
arcaneRegisterAcceleratorRuntimecuda(Arcane::Accelerator::RegisterRuntimeInfo& init_info)
{
  using namespace Arcane;
  using namespace Arcane::Accelerator::Cuda;
  global_cuda_runtime.build();
  Arcane::Accelerator::impl::setUsingCUDARuntime(true);
  Arcane::Accelerator::impl::setCUDARunQueueRuntime(&global_cuda_runtime);
  initializeCudaMemoryAllocators();
  MemoryUtils::setDefaultDataMemoryResource(eMemoryResource::UnifiedMemory);
  MemoryUtils::setAcceleratorHostMemoryAllocator(&unified_memory_cuda_memory_allocator);
  IMemoryResourceMngInternal* mrm = MemoryUtils::getDataMemoryResourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryResource::UnifiedMemory, &unified_memory_cuda_memory_allocator);
  mrm->setAllocator(eMemoryResource::HostPinned, &host_pinned_cuda_memory_allocator);
  mrm->setAllocator(eMemoryResource::Device, &device_cuda_memory_allocator);
  mrm->setCopier(&global_cuda_memory_copier);
  global_cuda_runtime.fillDevices(init_info.isVerbose());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
