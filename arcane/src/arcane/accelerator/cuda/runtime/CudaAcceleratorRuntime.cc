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

#include "arcane/accelerator/cuda/CudaAccelerator.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/internal/MemoryUtilsInternal.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/DeviceInfoList.h"
#include "arcane/accelerator/core/KernelLaunchArgs.h"

#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/RegisterRuntimeInfo.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/DeviceMemoryInfo.h"
#include "arcane/accelerator/core/NativeStream.h"

#include "arcane/accelerator/cuda/runtime/internal/Cupti.h"

#include <iostream>
#include <unordered_map>
#include <mutex>

#include <cuda.h>

#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
#include <nvtx3/nvToolsExt.h>
#endif

using namespace Arccore;

namespace Arcane::Accelerator::Cuda
{
using impl::KernelLaunchArgs;

namespace
{
  Int32 global_cupti_flush = 0;
  CuptiInfo global_cupti_info;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void arcaneCheckCudaErrors(const TraceInfo& ti, CUresult e)
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

  ARCANE_FATAL("CUDA Error trace={0} e={1} name={2} message={3}",
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
    const char* func_name = nullptr;
    cudaFuncGetName(&func_name, kernel_ptr);
    m_nb_thread_per_block_map[kernel_ptr] = computed_block_size;
    std::cout << "ComputedBlockSize=" << computed_block_size << " n0=" << num_block_0 << " n1=" << num_block_1
              << " min_grid_size=" << min_grid_size << " nb_reg=" << func_attr.numRegs
              << " name=" << func_name << "\n";
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
  impl::NativeStream nativeStream() override
  {
    return impl::NativeStream(&m_cuda_stream);
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
    ARCANE_CHECK_POINTER(start_event);
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
      ARCANE_FATAL("Device {0} is not an accelerator device", id);
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
    OStringStream ostr;
    std::ostream& o = ostr.stream();
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
  void copy(ConstMemoryView from, [[maybe_unused]] eMemoryRessource from_mem,
            MutableMemoryView to, [[maybe_unused]] eMemoryRessource to_mem,
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
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimecuda(Arcane::Accelerator::RegisterRuntimeInfo& init_info)
{
  using namespace Arcane;
  using namespace Arcane::Accelerator::Cuda;
  global_cuda_runtime.build();
  Arcane::Accelerator::impl::setUsingCUDARuntime(true);
  Arcane::Accelerator::impl::setCUDARunQueueRuntime(&global_cuda_runtime);
  initializeCudaMemoryAllocators();
  MemoryUtils::setAcceleratorHostMemoryAllocator(getCudaMemoryAllocator());
  IMemoryRessourceMngInternal* mrm = MemoryUtils::getDataMemoryResourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryRessource::UnifiedMemory, getCudaUnifiedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::HostPinned, getCudaHostPinnedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::Device, getCudaDeviceMemoryAllocator());
  mrm->setCopier(&global_cuda_memory_copier);
  global_cuda_runtime.fillDevices(init_info.isVerbose());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
