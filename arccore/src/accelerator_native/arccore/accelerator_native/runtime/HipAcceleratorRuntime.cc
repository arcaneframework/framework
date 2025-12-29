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

#include "arccore/accelerator_native/HipAccelerator.h"

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

#include <sstream>

#ifdef ARCCORE_HAS_ROCTX
#include <roctx.h>
#endif

using namespace Arccore;

namespace Arcane::Accelerator::Hip
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConcreteAllocator
{
 public:

  virtual ~ConcreteAllocator() = default;

 public:

  virtual hipError_t _allocate(void** ptr, size_t new_size) = 0;
  virtual hipError_t _deallocate(void* ptr) = 0;
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
    ARCCORE_CHECK_HIP(m_concrete_allocator._allocate(&out, size));
    return out;
  }
  void freeMemory(void* ptr, [[maybe_unused]] size_t size) final
  {
    ARCCORE_CHECK_HIP_NOTHROW(m_concrete_allocator._deallocate(ptr));
  }

  void doMemoryCopy(void* destination, const void* source, Int64 size) final
  {
    ARCCORE_CHECK_HIP(hipMemcpy(destination, source, size, hipMemcpyDefault));
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

  hipError_t _deallocate(void* ptr) final
  {
    return ::hipFree(ptr);
  }

  hipError_t _allocate(void** ptr, size_t new_size) final
  {
    auto r = ::hipMallocManaged(ptr, new_size, hipMemAttachGlobal);
    return r;
  }

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::UnifiedMemory; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemoryHipMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:

  UnifiedMemoryHipMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("UnifiedMemoryHipMemory", new UnderlyingAllocator<UnifiedMemoryConcreteAllocator>())
  {
  }

 public:

  void initialize()
  {
    _doInitializeUVM(true);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedConcreteAllocator
: public ConcreteAllocator
{
 public:

  hipError_t _allocate(void** ptr, size_t new_size) final
  {
    return ::hipHostMalloc(ptr, new_size);
  }
  hipError_t _deallocate(void* ptr) final
  {
    return ::hipHostFree(ptr);
  }
  constexpr eMemoryResource memoryResource() const { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedHipMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{
 public:
 public:

  HostPinnedHipMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("HostPinnedHipMemory", new UnderlyingAllocator<HostPinnedConcreteAllocator>())
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
  }

  hipError_t _allocate(void** ptr, size_t new_size) final
  {
    hipError_t r = ::hipMalloc(ptr, new_size);
    return r;
  }
  hipError_t _deallocate(void* ptr) final
  {
    return ::hipFree(ptr);
  }

  constexpr eMemoryResource memoryResource() const { return eMemoryResource::Device; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceHipMemoryAllocator
: public AcceleratorMemoryAllocatorBase
{

 public:

  DeviceHipMemoryAllocator()
  : AcceleratorMemoryAllocatorBase("DeviceHipMemoryAllocator", new UnderlyingAllocator<DeviceConcreteAllocator>())
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
  UnifiedMemoryHipMemoryAllocator unified_memory_hip_memory_allocator;
  HostPinnedHipMemoryAllocator host_pinned_hip_memory_allocator;
  DeviceHipMemoryAllocator device_hip_memory_allocator;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void initializeHipMemoryAllocators()
{
  unified_memory_hip_memory_allocator.initialize();
  device_hip_memory_allocator.initialize();
  host_pinned_hip_memory_allocator.initialize();
}

void finalizeHipMemoryAllocators(ITraceMng* tm)
{
  unified_memory_hip_memory_allocator.finalize(tm);
  device_hip_memory_allocator.finalize(tm);
  host_pinned_hip_memory_allocator.finalize(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunQueueStream
: public Impl::IRunQueueStream
{
 public:

  HipRunQueueStream(Impl::IRunnerRuntime* runtime, const RunQueueBuildInfo& bi)
  : m_runtime(runtime)
  {
    if (bi.isDefault())
      ARCCORE_CHECK_HIP(hipStreamCreate(&m_hip_stream));
    else {
      int priority = bi.priority();
      ARCCORE_CHECK_HIP(hipStreamCreateWithPriority(&m_hip_stream, hipStreamDefault, priority));
    }
  }
  ~HipRunQueueStream() override
  {
    ARCCORE_CHECK_HIP_NOTHROW(hipStreamDestroy(m_hip_stream));
  }

 public:

  void notifyBeginLaunchKernel([[maybe_unused]] Impl::RunCommandImpl& c) override
  {
#ifdef ARCCORE_HAS_ROCTX
    auto kname = c.kernelName();
    if (kname.empty())
      roctxRangePush(c.traceInfo().name());
    else
      roctxRangePush(kname.localstr());
#endif
    return m_runtime->notifyBeginLaunchKernel();
  }
  void notifyEndLaunchKernel(Impl::RunCommandImpl&) override
  {
#ifdef ARCCORE_HAS_ROCTX
    roctxRangePop();
#endif
    return m_runtime->notifyEndLaunchKernel();
  }
  void barrier() override
  {
    ARCCORE_CHECK_HIP(hipStreamSynchronize(m_hip_stream));
  }
  bool _barrierNoException() override
  {
    return hipStreamSynchronize(m_hip_stream) != hipSuccess;
  }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    auto r = hipMemcpyAsync(args.destination().data(), args.source().data(),
                            args.source().bytes().size(), hipMemcpyDefault, m_hip_stream);
    ARCCORE_CHECK_HIP(r);
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
    ARCCORE_CHECK_HIP(r);
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

  Impl::IRunnerRuntime* m_runtime;
  hipStream_t m_hip_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunQueueEvent
: public Impl::IRunQueueEventImpl
{
 public:

  explicit HipRunQueueEvent(bool has_timer)
  {
    if (has_timer)
      ARCCORE_CHECK_HIP(hipEventCreate(&m_hip_event));
    else
      ARCCORE_CHECK_HIP(hipEventCreateWithFlags(&m_hip_event, hipEventDisableTiming));
  }
  ~HipRunQueueEvent() override
  {
    ARCCORE_CHECK_HIP_NOTHROW(hipEventDestroy(m_hip_event));
  }

 public:

  // Enregistre l'événement au sein d'une RunQueue
  void recordQueue(Impl::IRunQueueStream* stream) final
  {
    auto* rq = static_cast<HipRunQueueStream*>(stream);
    ARCCORE_CHECK_HIP(hipEventRecord(m_hip_event, rq->trueStream()));
  }

  void wait() final
  {
    ARCCORE_CHECK_HIP(hipEventSynchronize(m_hip_event));
  }

  void waitForEvent(Impl::IRunQueueStream* stream) final
  {
    auto* rq = static_cast<HipRunQueueStream*>(stream);
    ARCCORE_CHECK_HIP(hipStreamWaitEvent(rq->trueStream(), m_hip_event, 0));
  }

  Int64 elapsedTime(IRunQueueEventImpl* from_event) final
  {
    auto* true_from_event = static_cast<HipRunQueueEvent*>(from_event);
    ARCCORE_CHECK_POINTER(true_from_event);
    float time_in_ms = 0.0;
    ARCCORE_CHECK_HIP(hipEventElapsedTime(&time_in_ms, true_from_event->m_hip_event, m_hip_event));
    double x = time_in_ms * 1.0e6;
    Int64 nano_time = static_cast<Int64>(x);
    return nano_time;
  }

  bool hasPendingWork() final
  {
    hipError_t v = hipEventQuery(m_hip_event);
    if (v == hipErrorNotReady)
      return true;
    ARCCORE_CHECK_HIP(v);
    return false;
  }

 private:

  hipEvent_t m_hip_event;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunnerRuntime
: public Impl::IRunnerRuntime
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
    ARCCORE_CHECK_HIP(hipGetLastError());
    if (m_is_verbose)
      std::cout << "END HIP KERNEL!\n";
  }
  void barrier() override
  {
    ARCCORE_CHECK_HIP(hipDeviceSynchronize());
  }
  eExecutionPolicy executionPolicy() const override
  {
    return eExecutionPolicy::HIP;
  }
  Impl::IRunQueueStream* createStream(const RunQueueBuildInfo& bi) override
  {
    return new HipRunQueueStream(this, bi);
  }
  Impl::IRunQueueEventImpl* createEventImpl() override
  {
    return new HipRunQueueEvent(false);
  }
  Impl::IRunQueueEventImpl* createEventImplWithTimer() override
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
    ARCCORE_CHECK_HIP(hipMemAdvise(ptr, count, hip_advise, device));
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
    ARCCORE_CHECK_HIP(hipMemAdvise(ptr, count, hip_advise, device));
  }

  void setCurrentDevice(DeviceId device_id) final
  {
    Int32 id = device_id.asInt32();
    ARCCORE_FATAL_IF(!device_id.isAccelerator(), "Device {0} is not an accelerator device", id);
    ARCCORE_CHECK_HIP(hipSetDevice(id));
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
    ARCCORE_CHECK_HIP(hipGetDevice(&d));
    if (d != wanted_d)
      ARCCORE_CHECK_HIP(hipSetDevice(wanted_d));
    size_t free_mem = 0;
    size_t total_mem = 0;
    ARCCORE_CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    if (d != wanted_d)
      ARCCORE_CHECK_HIP(hipSetDevice(d));
    DeviceMemoryInfo dmi;
    dmi.setFreeMemory(free_mem);
    dmi.setTotalMemory(total_mem);
    return dmi;
  }

  void pushProfilerRange(const String& name, [[maybe_unused]] Int32 color) override
  {
#ifdef ARCCORE_HAS_ROCTX
    roctxRangePush(name.localstr());
#endif
  }
  void popProfilerRange() override
  {
#ifdef ARCCORE_HAS_ROCTX
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
  Impl::DeviceInfoList m_device_info_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HipRunnerRuntime::
fillDevices(bool is_verbose)
{
  int nb_device = 0;
  ARCCORE_CHECK_HIP(hipGetDeviceCount(&nb_device));
  std::ostream& omain = std::cout;
  if (is_verbose)
    omain << "ArcaneHIP: Initialize Arcane HIP runtime nb_available_device=" << nb_device << "\n";
  for (int i = 0; i < nb_device; ++i) {
    std::ostringstream ostr;
    std::ostream& o = ostr;

    hipDeviceProp_t dp;
    ARCCORE_CHECK_HIP(hipGetDeviceProperties(&dp, i));

    int has_managed_memory = 0;
    ARCCORE_CHECK_HIP(hipDeviceGetAttribute(&has_managed_memory, hipDeviceAttributeManagedMemory, i));

    // Le format des versions dans HIP est:
    // HIP_VERSION  =  (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)

    int runtime_version = 0;
    ARCCORE_CHECK_HIP(hipRuntimeGetVersion(&runtime_version));
    //runtime_version /= 10000;
    int runtime_major = runtime_version / 10000000;
    int runtime_minor = (runtime_version / 100000) % 100;

    int driver_version = 0;
    ARCCORE_CHECK_HIP(hipDriverGetVersion(&driver_version));
    //driver_version /= 10000;
    int driver_major = driver_version / 10000000;
    int driver_minor = (driver_version / 100000) % 100;

    o << "\nDevice " << i << " name=" << dp.name << "\n";
    o << " Driver version = " << driver_major << "." << (driver_minor) << "." << (driver_version % 100000) << "\n";
    o << " Runtime version = " << runtime_major << "." << (runtime_minor) << "." << (runtime_version % 100000) << "\n";
    o << " computeCapability = " << dp.major << "." << dp.minor << "\n";
    o << " totalGlobalMem = " << dp.totalGlobalMem << "\n";
    o << " sharedMemPerBlock = " << dp.sharedMemPerBlock << "\n";
    o << " sharedMemPerMultiprocessor = " << dp.sharedMemPerMultiprocessor << "\n";
    o << " sharedMemPerBlockOptin = " << dp.sharedMemPerBlockOptin << "\n";
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
    o << " pciInfo = " << dp.pciDomainID << " " << dp.pciBusID << " " << dp.pciDeviceID << "\n";
#if HIP_VERSION_MAJOR >= 6
    o << " gpuDirectRDMASupported = " << dp.gpuDirectRDMASupported << "\n";
    o << " hostNativeAtomicSupported = " << dp.hostNativeAtomicSupported << "\n";
    o << " unifiedFunctionPointers = " << dp.unifiedFunctionPointers << "\n";
#endif
    std::ostringstream device_uuid_ostr;
    {
      hipDevice_t device;
      ARCCORE_CHECK_HIP(hipDeviceGet(&device, i));
      hipUUID device_uuid;
      ARCCORE_CHECK_HIP(hipDeviceGetUuid(&device_uuid, device));
      o << " deviceUuid=";
      Impl::printUUID(device_uuid_ostr, device_uuid.bytes);
      o << device_uuid_ostr.str();
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
    device_info.setUUIDAsString(device_uuid_ostr.str());
    device_info.setSharedMemoryPerBlock(static_cast<Int32>(dp.sharedMemPerBlock));
    device_info.setSharedMemoryPerMultiprocessor(static_cast<Int32>(dp.sharedMemPerMultiprocessor));
    device_info.setSharedMemoryPerBlockOptin(static_cast<Int32>(dp.sharedMemPerBlockOptin));
    device_info.setTotalConstMemory(static_cast<Int32>(dp.totalConstMem));
    device_info.setPCIDomainID(dp.pciDomainID);
    device_info.setPCIBusID(dp.pciBusID);
    device_info.setPCIDeviceID(dp.pciDeviceID);
    m_device_info_list.addDevice(device_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipMemoryCopier
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
    // 'hipMemcpyDefault' sait automatiquement ce qu'il faut faire en tenant
    // uniquement compte de la valeur des pointeurs. Il faudrait voir si
    // utiliser \a from_mem et \a to_mem peut améliorer les performances.
    ARCCORE_CHECK_HIP(hipMemcpy(to.data(), from.data(), from.bytes().size(), hipMemcpyDefault));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Hip

using namespace Arcane;

namespace
{
Arcane::Accelerator::Hip::HipRunnerRuntime global_hip_runtime;
Arcane::Accelerator::Hip::HipMemoryCopier global_hip_memory_copier;

void _setAllocator(Accelerator::AcceleratorMemoryAllocatorBase* allocator)
{
  IMemoryResourceMngInternal* mrm = MemoryUtils::getDataMemoryResourceMng()->_internal();
  eMemoryResource mem = allocator->memoryResource();
  mrm->setAllocator(mem, allocator);
  mrm->setMemoryPool(mem, allocator->memoryPool());
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette fonction est le point d'entrée utilisé lors du chargement
// dynamique de cette bibliothèque
extern "C" ARCCORE_EXPORT void
arcaneRegisterAcceleratorRuntimehip(Arcane::Accelerator::RegisterRuntimeInfo& init_info)
{
  using namespace Arcane::Accelerator::Hip;
  Arcane::Accelerator::Impl::setUsingHIPRuntime(true);
  Arcane::Accelerator::Impl::setHIPRunQueueRuntime(&global_hip_runtime);
  initializeHipMemoryAllocators();
  MemoryUtils::setDefaultDataMemoryResource(eMemoryResource::UnifiedMemory);
  MemoryUtils::setAcceleratorHostMemoryAllocator(&unified_memory_hip_memory_allocator);
  IMemoryResourceMngInternal* mrm = MemoryUtils::getDataMemoryResourceMng()->_internal();
  mrm->setIsAccelerator(true);
  _setAllocator(&unified_memory_hip_memory_allocator);
  _setAllocator(&host_pinned_hip_memory_allocator);
  _setAllocator(&device_hip_memory_allocator);
  mrm->setCopier(&global_hip_memory_copier);
  global_hip_runtime.fillDevices(init_info.isVerbose());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
