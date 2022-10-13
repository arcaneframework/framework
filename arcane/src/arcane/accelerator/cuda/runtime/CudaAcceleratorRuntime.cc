// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaAcceleratorRuntime.cc                                   (C) 2000-2022 */
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
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"

#include "arcane/accelerator/core/IRunQueueRuntime.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"

#include <iostream>

#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
#include <nvToolsExt.h>
#endif

using namespace Arccore;

namespace Arcane::Accelerator::Cuda
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void checkDevices()
{
  int nb_device = 0;
  ARCANE_CHECK_CUDA(cudaGetDeviceCount(&nb_device));
  std::ostream& o = std::cout;
  o << "Initialize Arcane CUDA runtime\n";
  o << "Available device = " << nb_device << "\n";
  for (int i = 0; i < nb_device; ++i) {
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, i);

    o << "\nDevice " << i << " name=" << dp.name << "\n";
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
    int least_val = 0;
    int greatest_val = 0;
    ARCANE_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&least_val, &greatest_val));
    o << " leastPriority = " << least_val << " greatestPriority = " << greatest_val << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaRunQueueStream
: public impl::IRunQueueStream
{
 public:

  CudaRunQueueStream(impl::IRunQueueRuntime* runtime, const RunQueueBuildInfo& bi)
  : m_runtime(runtime)
  {
    if (bi.isDefault())
      ARCANE_CHECK_CUDA(cudaStreamCreate(&m_cuda_stream));
    else {
      int priority = bi.priority();
      ARCANE_CHECK_CUDA(cudaStreamCreateWithPriority(&m_cuda_stream, cudaStreamDefault, priority));
    }
  }
  ~CudaRunQueueStream() noexcept(false) override
  {
    ARCANE_CHECK_CUDA(cudaStreamDestroy(m_cuda_stream));
  }

 public:

  void notifyBeginKernel([[maybe_unused]] RunCommand& c) override
  {
#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
    auto kname = c.kernelName();
    if (kname.empty())
      nvtxRangePush(c.traceInfo().name());
    else
      nvtxRangePush(kname.localstr());
#endif
    return m_runtime->notifyBeginKernel();
  }
  void notifyEndKernel(RunCommand&) override
  {
#ifdef ARCANE_HAS_CUDA_NVTOOLSEXT
    nvtxRangePop();
#endif
    return m_runtime->notifyEndKernel();
  }
  void barrier() override
  {
    ARCANE_CHECK_CUDA(cudaStreamSynchronize(m_cuda_stream));
  }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    auto r = cudaMemcpyAsync(args.destination().data(), args.source().data(),
                             args.source().length(), cudaMemcpyDefault, m_cuda_stream);
    ARCANE_CHECK_CUDA(r);
    if (!args.isAsync())
      barrier();
  }
  void prefetchMemory(const MemoryPrefetchArgs& args) override
  {
    DeviceId d = args.deviceId();
    int device = cudaCpuDeviceId;
    if (!d.isHost())
      device = d.asInt32();
    //std::cout << "PREFETCH device=" << device << " host=" << cudaCpuDeviceId << " size=" << args.source().length() << "\n";
    auto r = cudaMemPrefetchAsync(args.source().data(), args.source().length(), device, m_cuda_stream);
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

  impl::IRunQueueRuntime* m_runtime;
  cudaStream_t m_cuda_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaRunQueueEvent
: public impl::IRunQueueEventImpl
{
 public:

  CudaRunQueueEvent()
  {
    ARCANE_CHECK_CUDA(cudaEventCreateWithFlags(&m_cuda_event, cudaEventDisableTiming));
  }
  ~CudaRunQueueEvent() noexcept(false) override
  {
    ARCANE_CHECK_CUDA(cudaEventDestroy(m_cuda_event));
  }

 public:

  // Enregistre l'événement au sein d'une RunQueue
  void recordQueue(impl::IRunQueueStream* stream) override
  {
    auto* rq = static_cast<CudaRunQueueStream*>(stream);
    ARCANE_CHECK_CUDA(cudaEventRecord(m_cuda_event, rq->trueStream()));
  }

  void wait() override
  {
    ARCANE_CHECK_CUDA(cudaEventSynchronize(m_cuda_event));
  }

  void waitForEvent(impl::IRunQueueStream* stream) override
  {
    auto* rq = static_cast<CudaRunQueueStream*>(stream);
    ARCANE_CHECK_CUDA(cudaStreamWaitEvent(rq->trueStream(), m_cuda_event, cudaEventWaitDefault));
  }

 private:

  cudaEvent_t m_cuda_event;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaRunQueueRuntime
: public impl::IRunQueueRuntime
{
 public:

  ~CudaRunQueueRuntime() override = default;

 public:

  void notifyBeginKernel() override
  {
    ++m_nb_kernel_launched;
    if (m_is_verbose)
      std::cout << "BEGIN CUDA KERNEL!\n";
  }
  void notifyEndKernel() override
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
    return new CudaRunQueueEvent();
  }
  void setMemoryAdvice(MemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    auto v = buffer.span();
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
    cudaMemAdvise(ptr, count, cuda_advise, device);
  }
  void unsetMemoryAdvice(MemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    auto v = buffer.span();
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
    cudaMemAdvise(ptr, count, cuda_advise, device);
  }

 private:

  Int64 m_nb_kernel_launched = 0;
  bool m_is_verbose = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CudaMemoryCopier
: public IMemoryCopier
{
  void copy(MemoryView from, [[maybe_unused]] eMemoryRessource from_mem,
            MutableMemoryView to, [[maybe_unused]] eMemoryRessource to_mem) override
  {
    // 'cudaMemcpyDefault' sait automatiquement ce qu'il faut faire en tenant
    // uniquement compte de la valeur des pointeurs. Il faudrait voir si
    // utiliser \a from_mem et \a to_mem peut améliorer les performances.
    ARCANE_CHECK_CUDA(cudaMemcpy(to.span().data(), from.span().data(), from.size(), cudaMemcpyDefault));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Cuda

namespace
{
Arcane::Accelerator::Cuda::CudaRunQueueRuntime global_cuda_runtime;
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
  Arcane::platform::setAcceleratorHostMemoryAllocator(getCudaMemoryAllocator());
  IMemoryRessourceMngInternal* mrm = platform::getDataMemoryRessourceMng()->_internal();
  mrm->setAllocator(eMemoryRessource::UnifiedMemory,getCudaUnifiedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::HostPinned,getCudaHostPinnedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::Device,getCudaDeviceMemoryAllocator());
  mrm->setCopier(&global_cuda_memory_copier);
  checkDevices();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
