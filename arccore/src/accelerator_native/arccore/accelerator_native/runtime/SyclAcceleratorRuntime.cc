// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SyclAcceleratorRuntime.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Runtime pour 'SYCL'.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator_native/SyclAccelerator.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/NotSupportedException.h"

#include "arccore/common/AlignedMemoryAllocator.h"
#include "arccore/common/AllocatedMemoryInfo.h"
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

namespace Arcane::Accelerator::Sycl
{

#define ARCCORE_SYCL_FUNC_NOT_HANDLED \
  std::cout << "WARNING: SYCL: function not handled " << A_FUNCINFO << "\n"

class SyclRunnerRuntime;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette file est utilisée pour les allocations.
// Elle doit donc toujours exister car on ne sait pas quand aura lieu
// la dernière désallocation.
sycl::queue global_default_queue;
namespace
{
  sycl::queue& _defaultQueue()
  {
    return global_default_queue;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un allocateur spécifique pour 'Sycl'.
 */
class SyclMemoryAllocatorBase
: public AlignedMemoryAllocator
{
 public:

  SyclMemoryAllocatorBase()
  : AlignedMemoryAllocator(128)
  {}

  bool hasRealloc(MemoryAllocationArgs) const override { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override
  {
    sycl::queue& q = _defaultQueue();
    void* out = nullptr;
    _allocate(&out, new_size, args, q);
    if (!out)
      ARCCORE_FATAL("Can not allocate memory size={0}", new_size);
    Int64 a = reinterpret_cast<Int64>(out);
    if ((a % 128) != 0)
      ARCCORE_FATAL("Bad alignment for SYCL allocator: offset={0}", (a % 128));
    return { out, new_size };
  }
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override
  {
    sycl::queue& q = _defaultQueue();
    AllocatedMemoryInfo a = allocate(args, new_size);
    q.submit([&](sycl::handler& cgh) {
      cgh.memcpy(a.baseAddress(), current_ptr.baseAddress(), current_ptr.size());
    });
    q.wait();

    deallocate(args, current_ptr);
    return a;
  }
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override
  {
    sycl::queue& q = _defaultQueue();
    _deallocate(ptr.baseAddress(), args, q);
  }

 protected:

  virtual void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) = 0;
  virtual void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnifiedMemorySyclMemoryAllocator
: public SyclMemoryAllocatorBase
{
 protected:

  void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) override
  {
    *ptr = sycl::malloc_shared(new_size, q);
  }
  void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) override
  {
    sycl::free(ptr, q);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::UnifiedMemory; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostPinnedSyclMemoryAllocator
: public SyclMemoryAllocatorBase
{
 protected:

  void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) override
  {
    // TODO: Faire host-pinned
    *ptr = sycl::malloc_host(new_size, q);
  }
  void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) override
  {
    sycl::free(ptr, q);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::HostPinned; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DeviceSyclMemoryAllocator
: public SyclMemoryAllocatorBase
{
 protected:

  void _allocate(void** ptr, size_t new_size, MemoryAllocationArgs, sycl::queue& q) override
  {
    *ptr = sycl::malloc_device(new_size, q);
  }
  void _deallocate(void* ptr, MemoryAllocationArgs, sycl::queue& q) override
  {
    sycl::free(ptr, q);
  }
  eMemoryResource memoryResource() const override { return eMemoryResource::Device; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  UnifiedMemorySyclMemoryAllocator unified_memory_sycl_memory_allocator;
  HostPinnedSyclMemoryAllocator host_pinned_sycl_memory_allocator;
  DeviceSyclMemoryAllocator device_sycl_memory_allocator;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SyclRunQueueStream
: public impl::IRunQueueStream
{
 public:

  SyclRunQueueStream(SyclRunnerRuntime* runtime, const RunQueueBuildInfo& bi);
  ~SyclRunQueueStream() override
  {
  }

 public:

  void notifyBeginLaunchKernel([[maybe_unused]] impl::RunCommandImpl& c) override
  {
    return m_runtime->notifyBeginLaunchKernel();
  }
  void notifyEndLaunchKernel(impl::RunCommandImpl&) override
  {
    return m_runtime->notifyEndLaunchKernel();
  }
  void barrier() override
  {
    m_sycl_stream->wait_and_throw();
  }
  bool _barrierNoException() override
  {
    m_sycl_stream->wait();
    return false;
  }
  void copyMemory(const MemoryCopyArgs& args) override
  {
    auto source_bytes = args.source().bytes();
    m_sycl_stream->memcpy(args.destination().data(), source_bytes.data(),
                          source_bytes.size());
    if (!args.isAsync())
      this->barrier();
  }
  void prefetchMemory([[maybe_unused]] const MemoryPrefetchArgs& args) override
  {
    auto source_bytes = args.source().bytes();
    Int64 nb_byte = source_bytes.size();
    if (nb_byte == 0)
      return;
    m_sycl_stream->prefetch(source_bytes.data(), nb_byte);
    if (!args.isAsync())
      this->barrier();
  }
  Impl::NativeStream nativeStream() override
  {
    return Impl::NativeStream(m_sycl_stream.get());
  }

  void _setSyclLastCommandEvent([[maybe_unused]] void* sycl_event_ptr) override
  {
    sycl::event last_event;
    if (sycl_event_ptr)
      last_event = *(reinterpret_cast<sycl::event*>(sycl_event_ptr));
    m_last_command_event = last_event;
  }

 public:

  static sycl::async_handler _getAsyncHandler()
  {
    auto err_handler = [](const sycl::exception_list& exceptions) {
      std::ostringstream ostr;
      ostr << "Error in SYCL runtime\n";
      for (const std::exception_ptr& e : exceptions) {
        try {
          std::rethrow_exception(e);
        }
        catch (const sycl::exception& e) {
          ostr << "SYCL exception: " << e.what() << "\n";
        }
      }
      ARCCORE_FATAL(ostr.str());
    };
    return err_handler;
  }

  //! Évènement correspondant à la dernière commande
  sycl::event lastCommandEvent() { return m_last_command_event; }

 public:

  sycl::queue& trueStream() const
  {
    return *m_sycl_stream;
  }

 private:

  impl::IRunnerRuntime* m_runtime;
  std::unique_ptr<sycl::queue> m_sycl_stream;
  sycl::event m_last_command_event;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SyclRunQueueEvent
: public impl::IRunQueueEventImpl
{
 public:

  explicit SyclRunQueueEvent([[maybe_unused]] bool has_timer)
  {
  }
  ~SyclRunQueueEvent() override
  {
  }

 public:

  // Enregistre l'événement au sein d'une RunQueue
  void recordQueue([[maybe_unused]] impl::IRunQueueStream* stream) final
  {
    ARCCORE_CHECK_POINTER(stream);
    auto* rq = static_cast<SyclRunQueueStream*>(stream);
    m_sycl_event = rq->lastCommandEvent();
#if defined(__ADAPTIVECPP__)
    m_recorded_stream = stream;
    // TODO: Vérifier s'il faut faire quelque chose
#elif defined(__INTEL_LLVM_COMPILER)
    //m_sycl_event = rq->trueStream().ext_oneapi_submit_barrier();
#else
    ARCCORE_THROW(NotSupportedException, "Only supported for AdaptiveCpp and Intel DPC++ implementation");
#endif
  }

  void wait() final
  {
    //ARCCORE_SYCL_FUNC_NOT_HANDLED;
    // TODO: Vérifier ce que cela signifie exactement
    m_sycl_event.wait();
  }

  void waitForEvent([[maybe_unused]] impl::IRunQueueStream* stream) final
  {
#if defined(__ADAPTIVECPP__)
    auto* rq = static_cast<SyclRunQueueStream*>(stream);
    m_sycl_event.wait(rq->trueStream().get_wait_list());
#elif defined(__INTEL_LLVM_COMPILER)
    std::vector<sycl::event> events;
    events.push_back(m_sycl_event);
    auto* rq = static_cast<SyclRunQueueStream*>(stream);
    rq->trueStream().ext_oneapi_submit_barrier(events);
#else
    ARCCORE_THROW(NotSupportedException, "Only supported for AdaptiveCpp and Intel DPC++ implementation");
#endif
  }

  Int64 elapsedTime([[maybe_unused]] IRunQueueEventImpl* start_event) final
  {
    ARCCORE_CHECK_POINTER(start_event);
    // Il faut prendre l'évènement de début car on est certain qu'il contient
    // la bonne valeur de 'sycl::event'.
    sycl::event event = (static_cast<SyclRunQueueEvent*>(start_event))->m_sycl_event;
    // Si pas d'évènement associé, on ne fait rien pour éviter une exception
    if (event==sycl::event())
      return 0;

    bool is_submitted = event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
    if (!is_submitted)
      return 0;
    Int64 start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    Int64 end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (end - start);
  }

  bool hasPendingWork() final
  {
    ARCCORE_THROW(NotImplementedException,"hasPendingWork()");
  }

 private:

  sycl::event m_sycl_event;
  impl::IRunQueueStream* m_recorded_stream = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SyclRunnerRuntime
: public impl::IRunnerRuntime
{
  friend class SyclRunQueueStream;

 public:

  void notifyBeginLaunchKernel() override
  {
  }
  void notifyEndLaunchKernel() override
  {
  }
  void barrier() override
  {
    // TODO Faire le wait sur la file par défaut n'est pas strictement équivalent
    // à la barrière en CUDA qui synchronize tout le device.
    m_default_queue->wait();
  }
  eExecutionPolicy executionPolicy() const override
  {
    return eExecutionPolicy::SYCL;
  }
  impl::IRunQueueStream* createStream(const RunQueueBuildInfo& bi) override
  {
    return new SyclRunQueueStream(this, bi);
  }
  impl::IRunQueueEventImpl* createEventImpl() override
  {
    return new SyclRunQueueEvent(false);
  }
  impl::IRunQueueEventImpl* createEventImplWithTimer() override
  {
    return new SyclRunQueueEvent(true);
  }
  void setMemoryAdvice([[maybe_unused]] ConstMemoryView buffer, [[maybe_unused]] eMemoryAdvice advice,
                       [[maybe_unused]] DeviceId device_id) override
  {
  }
  void unsetMemoryAdvice([[maybe_unused]] ConstMemoryView buffer,
                         [[maybe_unused]] eMemoryAdvice advice, [[maybe_unused]] DeviceId device_id) override
  {
  }

  void setCurrentDevice([[maybe_unused]] DeviceId device_id) final
  {
    ARCCORE_SYCL_FUNC_NOT_HANDLED;
  }
  const IDeviceInfoList* deviceInfoList() override { return &m_device_info_list; }

  void getPointerAttribute(PointerAttribute& attribute, const void* ptr) override
  {
    sycl::usm::alloc sycl_mem_type = sycl::get_pointer_type(ptr, *m_default_context);
    ePointerMemoryType mem_type = ePointerMemoryType::Unregistered;
    const void* host_ptr = nullptr;
    const void* device_ptr = nullptr;
    if (sycl_mem_type == sycl::usm::alloc::host) {
      // HostPinned. Doit être accessible depuis le device mais
      //
      mem_type = ePointerMemoryType::Host;
      host_ptr = ptr;
      // TODO: Regarder comment récupérer la valeur
      device_ptr = ptr;
    }
    else if (sycl_mem_type == sycl::usm::alloc::device) {
      mem_type = ePointerMemoryType::Device;
      device_ptr = ptr;
    }
    else if (sycl_mem_type == sycl::usm::alloc::shared) {
      mem_type = ePointerMemoryType::Managed;
      // TODO: pour l'instant on remplit avec le pointeur car on ne sait
      // pas comment récupérer l'info.
      host_ptr = ptr;
      device_ptr = ptr;
    }
    // TODO: à corriger
    Int32 device_id = 0;
    _fillPointerAttribute(attribute, mem_type, device_id, ptr, device_ptr, host_ptr);
  }

  DeviceMemoryInfo getDeviceMemoryInfo([[maybe_unused]] DeviceId device_id) override
  {
    return {};
  }

  void fillDevicesAndSetDefaultQueue(bool is_verbose);
  sycl::queue& defaultQueue() const { return *m_default_queue; }
  sycl::device& defaultDevice() const { return *m_default_device; }

  void finalize(ITraceMng*) override
  {
    // Supprime la queue globale utilisée pour les allocations.
    global_default_queue = sycl::queue{};
  }

 private:

  impl::DeviceInfoList m_device_info_list;
  std::unique_ptr<sycl::device> m_default_device;
  std::unique_ptr<sycl::context> m_default_context;
  std::unique_ptr<sycl::queue> m_default_queue;

 private:

  void _init(sycl::device& device)
  {
    m_default_device = std::make_unique<sycl::device>(device);
    m_default_queue = std::make_unique<sycl::queue>(device);
    m_default_context = std::make_unique<sycl::context>(device);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SyclRunQueueStream::
SyclRunQueueStream(SyclRunnerRuntime* runtime, const RunQueueBuildInfo& bi)
: m_runtime(runtime)
{
  sycl::device& d = runtime->defaultDevice();
  // Indique que les commandes lancées sont implicitement exécutées les
  // unes derrière les autres.
  auto queue_property = sycl::property::queue::in_order();
  // Pour le profiling
  auto profiling_property = sycl::property::queue::enable_profiling();
  sycl::property_list queue_properties(queue_property, profiling_property);

  // Gestionnaire d'erreur.
  sycl::async_handler err_handler;
  err_handler = _getAsyncHandler();
  if (bi.isDefault())
    m_sycl_stream = std::make_unique<sycl::queue>(d, err_handler, queue_properties);
  else {
    ARCCORE_SYCL_FUNC_NOT_HANDLED;
    m_sycl_stream = std::make_unique<sycl::queue>(d, err_handler, queue_properties);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SyclRunnerRuntime::
fillDevicesAndSetDefaultQueue(bool is_verbose)
{
  if (is_verbose){
    for (auto platform : sycl::platform::get_platforms()) {
      std::cout << "Platform: "
                << platform.get_info<sycl::info::platform::name>()
                << std::endl;
    }
  }

  sycl::device device{ sycl::gpu_selector_v };
  if (is_verbose)
    std::cout << "\nDevice: " << device.get_info<sycl::info::device::name>()
              << "\nVersion=" << device.get_info<sycl::info::device::version>()
              << std::endl;
  // Pour l'instant, on prend comme file par défaut la première trouvée
  // et on ne considère qu'un seul device accessible.
  _init(device);

  DeviceInfo device_info;
  device_info.setDescription("No description info");
  device_info.setDeviceId(DeviceId(0));
  device_info.setName(device.get_info<sycl::info::device::name>());
  m_device_info_list.addDevice(device_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SyclMemoryCopier
: public IMemoryCopier
{
  void copy(ConstMemoryView from, eMemoryResource from_mem,
            MutableMemoryView to, eMemoryResource to_mem,
            const RunQueue* queue) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Sycl

namespace
{
Arcane::Accelerator::Sycl::SyclRunnerRuntime global_sycl_runtime;
Arcane::Accelerator::Sycl::SyclMemoryCopier global_sycl_memory_copier;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Sycl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SyclMemoryCopier::
copy(ConstMemoryView from, [[maybe_unused]] eMemoryResource from_mem,
     MutableMemoryView to, [[maybe_unused]] eMemoryResource to_mem,
     const RunQueue* queue)
{
  if (queue) {
    queue->copyMemory(MemoryCopyArgs(to.bytes(), from.bytes()).addAsync(queue->isAsync()));
    return;
  }
  sycl::queue& q = global_sycl_runtime.defaultQueue();
  q.memcpy(to.data(), from.data(), from.bytes().size()).wait();
}

} // namespace Arcane::Accelerator::Sycl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette fonction est le point d'entrée utilisé lors du chargement
// dynamique de cette bibliothèque
extern "C" ARCCORE_EXPORT void
arcaneRegisterAcceleratorRuntimesycl(Arcane::Accelerator::RegisterRuntimeInfo& init_info)
{
  using namespace Arcane;
  using namespace Arcane::Accelerator::Sycl;
  Arcane::Accelerator::impl::setUsingSYCLRuntime(true);
  Arcane::Accelerator::impl::setSYCLRunQueueRuntime(&global_sycl_runtime);
  MemoryUtils::setAcceleratorHostMemoryAllocator(&unified_memory_sycl_memory_allocator);
  MemoryUtils::setDefaultDataMemoryResource(eMemoryResource::UnifiedMemory);
  IMemoryResourceMngInternal* mrm = MemoryUtils::getDataMemoryResourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryResource::UnifiedMemory, &unified_memory_sycl_memory_allocator);
  mrm->setAllocator(eMemoryResource::HostPinned, &host_pinned_sycl_memory_allocator);
  mrm->setAllocator(eMemoryResource::Device, &device_sycl_memory_allocator);
  mrm->setCopier(&global_sycl_memory_copier);
  global_sycl_runtime.fillDevicesAndSetDefaultQueue(init_info.isVerbose());
  global_default_queue = global_sycl_runtime.defaultQueue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
