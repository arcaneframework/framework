// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAcceleratorRuntime.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Runtime pour 'HIP'.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/hip/HipAccelerator.h"

#include "arccore/base/MemoryView.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotImplementedException.h"

#include "arccore/common/IMemoryResourceMng.h"
#include "arccore/common/internal/IMemoryResourceMngInternal.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/internal/MemoryUtilsInternal.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/DeviceInfoList.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/DeviceMemoryInfo.h"
#include "arcane/accelerator/core/NativeStream.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/RegisterRuntimeInfo.h"
#include "arcane/accelerator/core/internal/IRunQueueStream.h"
#include "arcane/accelerator/core/internal/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"

#include <iostream>

#ifdef ARCANE_HAS_ROCTX
#include <roctx.h>
#endif

using namespace Arccore;

namespace Arcane::Accelerator::Hip
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunQueueStream
: public impl::IRunQueueStream
{
 public:

  HipRunQueueStream(impl::IRunnerRuntime* runtime, const RunQueueBuildInfo& bi)
  : m_runtime(runtime)
  {
    if (bi.isDefault())
      ARCANE_CHECK_HIP(hipStreamCreate(&m_hip_stream));
    else {
      int priority = bi.priority();
      ARCANE_CHECK_HIP(hipStreamCreateWithPriority(&m_hip_stream, hipStreamDefault, priority));
    }
  }
  ~HipRunQueueStream() override
  {
    ARCANE_CHECK_HIP_NOTHROW(hipStreamDestroy(m_hip_stream));
  }

 public:

  void notifyBeginLaunchKernel([[maybe_unused]] impl::RunCommandImpl& c) override
  {
#ifdef ARCANE_HAS_ROCTX
    auto kname = c.kernelName();
    if (kname.empty())
      roctxRangePush(c.traceInfo().name());
    else
      roctxRangePush(kname.localstr());
#endif
    return m_runtime->notifyBeginLaunchKernel();
  }
  void notifyEndLaunchKernel(impl::RunCommandImpl&) override
  {
#ifdef ARCANE_HAS_ROCTX
    roctxRangePop();
#endif
    return m_runtime->notifyEndLaunchKernel();
  }
  void barrier() override
  {
    ARCANE_CHECK_HIP(hipStreamSynchronize(m_hip_stream));
  }
  bool _barrierNoException() override
  {
    return hipStreamSynchronize(m_hip_stream) != hipSuccess;
  }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    auto r = hipMemcpyAsync(args.destination().data(), args.source().data(),
                            args.source().bytes().size(), hipMemcpyDefault, m_hip_stream);
    ARCANE_CHECK_HIP(r);
    if (!args.isAsync())
      barrier();
  }
  void prefetchMemory(const MemoryPrefetchArgs& args) override
  {
    auto src = args.source().bytes();
    if (src.size()==0)
      return;
    DeviceId d = args.deviceId();
    int device = hipCpuDeviceId;
    if (!d.isHost())
      device = d.asInt32();
    auto r = hipMemPrefetchAsync(src.data(), src.size(), device, m_hip_stream);
    ARCANE_CHECK_HIP(r);
    if (!args.isAsync())
      barrier();
  }
  Impl::NativeStream nativeStream() override
  {
    return Impl::NativeStream(&m_hip_stream);
  }

 public:

  hipStream_t trueStream() const
  {
    return m_hip_stream;
  }

 private:

  impl::IRunnerRuntime* m_runtime;
  hipStream_t m_hip_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunQueueEvent
: public impl::IRunQueueEventImpl
{
 public:

  explicit HipRunQueueEvent(bool has_timer)
  {
    if (has_timer)
      ARCANE_CHECK_HIP(hipEventCreate(&m_hip_event));
    else
      ARCANE_CHECK_HIP(hipEventCreateWithFlags(&m_hip_event, hipEventDisableTiming));
  }
  ~HipRunQueueEvent() override
  {
    ARCANE_CHECK_HIP_NOTHROW(hipEventDestroy(m_hip_event));
  }

 public:

  // Enregistre l'événement au sein d'une RunQueue
  void recordQueue(impl::IRunQueueStream* stream) final
  {
    auto* rq = static_cast<HipRunQueueStream*>(stream);
    ARCANE_CHECK_HIP(hipEventRecord(m_hip_event, rq->trueStream()));
  }

  void wait() final
  {
    ARCANE_CHECK_HIP(hipEventSynchronize(m_hip_event));
  }

  void waitForEvent(impl::IRunQueueStream* stream) final
  {
    auto* rq = static_cast<HipRunQueueStream*>(stream);
    ARCANE_CHECK_HIP(hipStreamWaitEvent(rq->trueStream(), m_hip_event, 0));
  }

  Int64 elapsedTime(IRunQueueEventImpl* from_event) final
  {
    auto* true_from_event = static_cast<HipRunQueueEvent*>(from_event);
    ARCANE_CHECK_POINTER(true_from_event);
    float time_in_ms = 0.0;
    ARCANE_CHECK_HIP(hipEventElapsedTime(&time_in_ms, true_from_event->m_hip_event, m_hip_event));
    double x = time_in_ms * 1.0e6;
    Int64 nano_time = static_cast<Int64>(x);
    return nano_time;
  }

  bool hasPendingWork() final
  {
    hipError_t v = hipEventQuery(m_hip_event);
    if (v == hipErrorNotReady)
      return true;
    ARCANE_CHECK_HIP(v);
    return false;
  }

 private:

  hipEvent_t m_hip_event;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunnerRuntime
: public impl::IRunnerRuntime
{
 public:

  ~HipRunnerRuntime() override = default;

 public:

  void notifyBeginLaunchKernel() override
  {
    ++m_nb_kernel_launched;
    if (m_is_verbose)
      std::cout << "BEGIN HIP KERNEL!\n";
  }
  void notifyEndLaunchKernel() override
  {
    ARCANE_CHECK_HIP(hipGetLastError());
    if (m_is_verbose)
      std::cout << "END HIP KERNEL!\n";
  }
  void barrier() override
  {
    ARCANE_CHECK_HIP(hipDeviceSynchronize());
  }
  eExecutionPolicy executionPolicy() const override
  {
    return eExecutionPolicy::HIP;
  }
  impl::IRunQueueStream* createStream(const RunQueueBuildInfo& bi) override
  {
    return new HipRunQueueStream(this, bi);
  }
  impl::IRunQueueEventImpl* createEventImpl() override
  {
    return new HipRunQueueEvent(false);
  }
  impl::IRunQueueEventImpl* createEventImplWithTimer() override
  {
    return new HipRunQueueEvent(true);
  }
  void setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    auto v = buffer.bytes();
    const void* ptr = v.data();
    size_t count = v.size();
    int device = device_id.asInt32();
    hipMemoryAdvise hip_advise;

    if (advice == eMemoryAdvice::MostlyRead)
      hip_advise = hipMemAdviseSetReadMostly;
    else if (advice == eMemoryAdvice::PreferredLocationDevice)
      hip_advise = hipMemAdviseSetPreferredLocation;
    else if (advice == eMemoryAdvice::AccessedByDevice)
      hip_advise = hipMemAdviseSetAccessedBy;
    else if (advice == eMemoryAdvice::PreferredLocationHost) {
      hip_advise = hipMemAdviseSetPreferredLocation;
      device = hipCpuDeviceId;
    }
    else if (advice == eMemoryAdvice::AccessedByHost) {
      hip_advise = hipMemAdviseSetAccessedBy;
      device = hipCpuDeviceId;
    }
    else
      return;
    //std::cout << "MEMADVISE p=" << ptr << " size=" << count << " advise = " << hip_advise << " id = " << device << "\n";
    ARCANE_CHECK_HIP(hipMemAdvise(ptr, count, hip_advise, device));
  }
  void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    auto v = buffer.bytes();
    const void* ptr = v.data();
    size_t count = v.size();
    int device = device_id.asInt32();
    hipMemoryAdvise hip_advise;

    if (advice == eMemoryAdvice::MostlyRead)
      hip_advise = hipMemAdviseUnsetReadMostly;
    else if (advice == eMemoryAdvice::PreferredLocationDevice)
      hip_advise = hipMemAdviseUnsetPreferredLocation;
    else if (advice == eMemoryAdvice::AccessedByDevice)
      hip_advise = hipMemAdviseUnsetAccessedBy;
    else if (advice == eMemoryAdvice::PreferredLocationHost) {
      hip_advise = hipMemAdviseUnsetPreferredLocation;
      device = hipCpuDeviceId;
    }
    else if (advice == eMemoryAdvice::AccessedByHost) {
      hip_advise = hipMemAdviseUnsetAccessedBy;
      device = hipCpuDeviceId;
    }
    else
      return;
    ARCANE_CHECK_HIP(hipMemAdvise(ptr, count, hip_advise, device));
  }

  void setCurrentDevice(DeviceId device_id) final
  {
    Int32 id = device_id.asInt32();
    if (!device_id.isAccelerator())
      ARCANE_FATAL("Device {0} is not an accelerator device", id);
    ARCANE_CHECK_HIP(hipSetDevice(id));
  }
  const IDeviceInfoList* deviceInfoList() override { return &m_device_info_list; }

  void getPointerAttribute(PointerAttribute& attribute, const void* ptr) override
  {
    hipPointerAttribute_t pa;
    hipError_t ret_value = hipPointerGetAttributes(&pa, ptr);
    auto mem_type = ePointerMemoryType::Unregistered;
    // Si \a ptr n'a pas été alloué dynamiquement (i.e: il est sur la pile),
    // hipPointerGetAttribute() retourne une erreur. Dans ce cas on considère
    // la mémoire comme non enregistrée.
    if (ret_value==hipSuccess){
#if HIP_VERSION_MAJOR >= 6
      auto rocm_memory_type = pa.type;
#else
      auto rocm_memory_type = pa.memoryType;
#endif
      if (pa.isManaged)
        mem_type = ePointerMemoryType::Managed;
      else if (rocm_memory_type == hipMemoryTypeHost)
        mem_type = ePointerMemoryType::Host;
      else if (rocm_memory_type == hipMemoryTypeDevice)
        mem_type = ePointerMemoryType::Device;
    }

    //std::cout << "HIP Info: hip_memory_type=" << (int)pa.memoryType << " is_managed?=" << pa.isManaged
    //          << " flags=" << pa.allocationFlags
    //          << " my_memory_type=" << (int)mem_type
    //          << "\n";
    _fillPointerAttribute(attribute, mem_type, pa.device,
                          ptr, pa.devicePointer, pa.hostPointer);
  }

  DeviceMemoryInfo getDeviceMemoryInfo(DeviceId device_id) override
  {
    int d = 0;
    int wanted_d = device_id.asInt32();
    ARCANE_CHECK_HIP(hipGetDevice(&d));
    if (d != wanted_d)
      ARCANE_CHECK_HIP(hipSetDevice(wanted_d));
    size_t free_mem = 0;
    size_t total_mem = 0;
    ARCANE_CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    if (d != wanted_d)
      ARCANE_CHECK_HIP(hipSetDevice(d));
    DeviceMemoryInfo dmi;
    dmi.setFreeMemory(free_mem);
    dmi.setTotalMemory(total_mem);
    return dmi;
  }

  void pushProfilerRange(const String& name, [[maybe_unused]] Int32 color) override
  {
#ifdef ARCANE_HAS_ROCTX
    roctxRangePush(name.localstr());
#endif
  }
  void popProfilerRange() override
  {
#ifdef ARCANE_HAS_ROCTX
    roctxRangePop();
#endif
  }

  void finalize(ITraceMng* tm) override
  {
    finalizeHipMemoryAllocators(tm);
  }

 public:

  void fillDevices(bool is_verbose);

 private:

  Int64 m_nb_kernel_launched = 0;
  bool m_is_verbose = false;
  impl::DeviceInfoList m_device_info_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HipRunnerRuntime::
fillDevices(bool is_verbose)
{
  int nb_device = 0;
  ARCANE_CHECK_HIP(hipGetDeviceCount(&nb_device));
  std::ostream& omain = std::cout;
  if (is_verbose)
    omain << "ArcaneHIP: Initialize Arcane HIP runtime nb_available_device=" << nb_device << "\n";
  for (int i = 0; i < nb_device; ++i) {
    OStringStream ostr;
    std::ostream& o = ostr.stream();

    hipDeviceProp_t dp;
    ARCANE_CHECK_HIP(hipGetDeviceProperties(&dp, i));

    int has_managed_memory = 0;
    ARCANE_CHECK_HIP(hipDeviceGetAttribute(&has_managed_memory, hipDeviceAttributeManagedMemory, i));

    // Le format des versions dans HIP est:
    // HIP_VERSION  =  (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)

    int runtime_version = 0;
    ARCANE_CHECK_HIP(hipRuntimeGetVersion(&runtime_version));
    //runtime_version /= 10000;
    int runtime_major = runtime_version / 10000000;
    int runtime_minor = (runtime_version / 100000) % 100;

    int driver_version = 0;
    ARCANE_CHECK_HIP(hipDriverGetVersion(&driver_version));
    //driver_version /= 10000;
    int driver_major = driver_version / 10000000;
    int driver_minor = (driver_version / 100000) % 100;

    o << "\nDevice " << i << " name=" << dp.name << "\n";
    o << " Driver version = " << driver_major << "." << (driver_minor) << "." << (driver_version % 100000) << "\n";
    o << " Runtime version = " << runtime_major << "." << (runtime_minor) << "." << (runtime_version % 100000) << "\n";
    o << " computeCapability = " << dp.major << "." << dp.minor << "\n";
    o << " totalGlobalMem = " << dp.totalGlobalMem << "\n";
    o << " sharedMemPerBlock = " << dp.sharedMemPerBlock << "\n";
    o << " regsPerBlock = " << dp.regsPerBlock << "\n";
    o << " warpSize = " << dp.warpSize << "\n";
    o << " memPitch = " << dp.memPitch << "\n";
    o << " maxThreadsPerBlock = " << dp.maxThreadsPerBlock << "\n";
    o << " totalConstMem = " << dp.totalConstMem << "\n";
    o << " clockRate = " << dp.clockRate << "\n";
    //o << " deviceOverlap = " << dp.deviceOverlap<< "\n";
    o << " multiProcessorCount = " << dp.multiProcessorCount << "\n";
    o << " kernelExecTimeoutEnabled = " << dp.kernelExecTimeoutEnabled << "\n";
    o << " integrated = " << dp.integrated << "\n";
    o << " canMapHostMemory = " << dp.canMapHostMemory << "\n";
    o << " computeMode = " << dp.computeMode << "\n";
    o << " maxThreadsDim = " << dp.maxThreadsDim[0] << " " << dp.maxThreadsDim[1]
      << " " << dp.maxThreadsDim[2] << "\n";
    o << " maxGridSize = " << dp.maxGridSize[0] << " " << dp.maxGridSize[1]
      << " " << dp.maxGridSize[2] << "\n";
    o << " concurrentManagedAccess = " << dp.concurrentManagedAccess << "\n";
    o << " directManagedMemAccessFromHost = " << dp.directManagedMemAccessFromHost << "\n";
    o << " gcnArchName = " << dp.gcnArchName << "\n";
    o << " pageableMemoryAccess = " << dp.pageableMemoryAccess << "\n";
    o << " pageableMemoryAccessUsesHostPageTables = " << dp.pageableMemoryAccessUsesHostPageTables << "\n";
    o << " hasManagedMemory = " << has_managed_memory << "\n";
#if HIP_VERSION_MAJOR >= 6
    o << " gpuDirectRDMASupported = " << dp.gpuDirectRDMASupported << "\n";
    o << " hostNativeAtomicSupported = " << dp.hostNativeAtomicSupported << "\n";
    o << " unifiedFunctionPointers = " << dp.unifiedFunctionPointers << "\n";
#endif
    {
      hipDevice_t device;
      ARCANE_CHECK_HIP(hipDeviceGet(&device, i));
      hipUUID device_uuid;
      ARCANE_CHECK_HIP(hipDeviceGetUuid(&device_uuid, device));
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipMemoryCopier
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
    // 'hipMemcpyDefault' sait automatiquement ce qu'il faut faire en tenant
    // uniquement compte de la valeur des pointeurs. Il faudrait voir si
    // utiliser \a from_mem et \a to_mem peut améliorer les performances.
    ARCANE_CHECK_HIP(hipMemcpy(to.data(), from.data(), from.bytes().size(), hipMemcpyDefault));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Hip

namespace 
{
Arcane::Accelerator::Hip::HipRunnerRuntime global_hip_runtime;
Arcane::Accelerator::Hip::HipMemoryCopier global_hip_memory_copier;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette fonction est le point d'entrée utilisé lors du chargement
// dynamique de cette bibliothèque
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimehip(Arcane::Accelerator::RegisterRuntimeInfo& init_info)
{
  using namespace Arcane;
  using namespace Arcane::Accelerator::Hip;
  Arcane::Accelerator::impl::setUsingHIPRuntime(true);
  Arcane::Accelerator::impl::setHIPRunQueueRuntime(&global_hip_runtime);
  initializeHipMemoryAllocators();
  MemoryUtils::setDefaultDataMemoryResource(eMemoryResource::UnifiedMemory);
  MemoryUtils::setAcceleratorHostMemoryAllocator(getHipMemoryAllocator());
  IMemoryResourceMngInternal* mrm = MemoryUtils::getDataMemoryResourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryRessource::UnifiedMemory, getHipUnifiedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::HostPinned, getHipHostPinnedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::Device, getHipDeviceMemoryAllocator());
  mrm->setCopier(&global_hip_memory_copier);
  global_hip_runtime.fillDevices(init_info.isVerbose());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
