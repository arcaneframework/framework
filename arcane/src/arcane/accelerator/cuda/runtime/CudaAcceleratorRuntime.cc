﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAcceleratorRuntime.cc                                   (C) 2000-2024 */
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
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/DeviceInfoList.h"

#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/RunQueue.h"

#include <iostream>

#include <cuda.h>

#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
#include <nvToolsExt.h>
#endif

using namespace Arccore;

namespace Arcane::Accelerator::Cuda
{
namespace
{
  Int32 global_cupti_level = 0;
  Int32 global_cupti_flush = 0;
} // namespace
extern "C++" void
initCupti(Int32 level, bool do_print);
extern "C++" void
flushCupti();
extern "C++" void
startCupti();
extern "C++" void
stopCupti();

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

void _printUUID(std::ostream& o, char bytes[16])
{
  static const char hexa_chars[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

  for (int i = 0; i < 16; ++i) {
    o << hexa_chars[(bytes[i] >> 4) & 0xf];
    o << hexa_chars[bytes[i] & 0xf];
    if (i == 4 || i == 6 || i == 8 || i == 10)
      o << '-';
  }
}

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
      flushCupti();
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
    if (src.size()==0)
      return;
    DeviceId d = args.deviceId();
    int device = cudaCpuDeviceId;
    if (!d.isHost())
      device = d.asInt32();
    //std::cout << "PREFETCH device=" << device << " host=" << cudaCpuDeviceId << " size=" << args.source().length() << "\n";
    auto r = cudaMemPrefetchAsync(src.data(), src.size(), device, m_cuda_stream);
    ARCANE_CHECK_CUDA(r);
    if (!args.isAsync())
      barrier();
  }
  void* _internalImpl() override
  {
    return &m_cuda_stream;
  }

 public:

  cudaStream_t trueStream() const
  {
    return m_cuda_stream;
  }

 private:

  impl::IRunnerRuntime* m_runtime;
  cudaStream_t m_cuda_stream;
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
    ARCANE_CHECK_CUDA(cudaMemAdvise(ptr, count, cuda_advise, device));
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
    ARCANE_CHECK_CUDA(cudaMemAdvise(ptr, count, cuda_advise, device));
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
    startCupti();
  }

  void stopProfiling() override
  {
    stopCupti();
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

 public:

  void fillDevices();

 private:

  Int64 m_nb_kernel_launched = 0;
  bool m_is_verbose = false;
  impl::DeviceInfoList m_device_info_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CudaRunnerRuntime::
fillDevices()
{
  int nb_device = 0;
  ARCANE_CHECK_CUDA(cudaGetDeviceCount(&nb_device));
  std::ostream& omain = std::cout;
  omain << "ArcaneCUDA: Initialize Arcane CUDA runtime nb_available_device=" << nb_device << "\n";
  for (int i = 0; i < nb_device; ++i) {
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, i);
    OStringStream ostr;
    std::ostream& o = ostr.stream();
    o << "Device " << i << " name=" << dp.name << "\n";
    o << " computeCapability = " << dp.major << "." << dp.minor << "\n";
    o << " totalGlobalMem = " << dp.totalGlobalMem << "\n";
    o << " sharedMemPerBlock = " << dp.sharedMemPerBlock << "\n";
    o << " regsPerBlock = " << dp.regsPerBlock << "\n";
    o << " warpSize = " << dp.warpSize << "\n";
    o << " memPitch = " << dp.memPitch << "\n";
    o << " maxThreadsPerBlock = " << dp.maxThreadsPerBlock << "\n";
    o << " totalConstMem = " << dp.totalConstMem << "\n";
    o << " clockRate = " << dp.clockRate << "\n";
    o << " deviceOverlap = " << dp.deviceOverlap << "\n";
    o << " multiProcessorCount = " << dp.multiProcessorCount << "\n";
    o << " kernelExecTimeoutEnabled = " << dp.kernelExecTimeoutEnabled << "\n";
    o << " integrated = " << dp.integrated << "\n";
    o << " canMapHostMemory = " << dp.canMapHostMemory << "\n";
    o << " computeMode = " << dp.computeMode << "\n";
    o << " maxThreadsDim = " << dp.maxThreadsDim[0] << " " << dp.maxThreadsDim[1]
      << " " << dp.maxThreadsDim[2] << "\n";
    o << " maxGridSize = " << dp.maxGridSize[0] << " " << dp.maxGridSize[1]
      << " " << dp.maxGridSize[2] << "\n";
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
      _printUUID(o, device_uuid.bytes);
      o << "\n";
    }
    String description(ostr.str());
    omain << description;

    DeviceInfo device_info;
    device_info.setDescription(description);
    device_info.setDeviceId(DeviceId(i));
    device_info.setName(dp.name);
    m_device_info_list.addDevice(device_info);
  }

  // Regarde si on active Cupti
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUPTI_LEVEL", true))
    global_cupti_level = v.value();
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUPTI_FLUSH", true))
    global_cupti_flush = v.value();
  bool do_print_cupti = true;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CUPTI_PRINT", true))
    do_print_cupti = (v.value() != 0);

  if (global_cupti_level > 0)
    initCupti(global_cupti_level, do_print_cupti);
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
    if (queue){
      queue->copyMemory(MemoryCopyArgs(to.bytes(),from.bytes()).addAsync(queue->isAsync()));
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
arcaneRegisterAcceleratorRuntimecuda()
{
  using namespace Arcane;
  using namespace Arcane::Accelerator::Cuda;
  Arcane::Accelerator::impl::setUsingCUDARuntime(true);
  Arcane::Accelerator::impl::setCUDARunQueueRuntime(&global_cuda_runtime);
  initializeCudaMemoryAllocators();
  Arcane::platform::setAcceleratorHostMemoryAllocator(getCudaMemoryAllocator());
  IMemoryRessourceMngInternal* mrm = platform::getDataMemoryRessourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryRessource::UnifiedMemory, getCudaUnifiedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::HostPinned, getCudaHostPinnedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::Device, getCudaDeviceMemoryAllocator());
  mrm->setCopier(&global_cuda_memory_copier);
  global_cuda_runtime.fillDevices();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
