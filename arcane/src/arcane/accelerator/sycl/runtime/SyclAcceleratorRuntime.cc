﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SyclAcceleratorRuntime.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Runtime pour 'SYCL'.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/sycl/SyclAccelerator.h"
#include "arcane/accelerator/sycl/internal/SyclAcceleratorInternal.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/DeviceInfoList.h"
#include "arcane/accelerator/core/RunQueue.h"

#include <iostream>

namespace Arcane::Accelerator::Sycl
{

using namespace Arccore;

#define ARCANE_SYCL_FUNC_NOT_HANDLED \
  std::cout << "WARNING: SYCL: function not handled " << A_FUNCINFO << "\n"

class SyclRunnerRuntime;

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
    m_sycl_stream->wait();
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
      m_sycl_stream->wait();
  }
  void prefetchMemory([[maybe_unused]] const MemoryPrefetchArgs& args) override
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }
  void* _internalImpl() override
  {
    return m_sycl_stream.get();
  }

 public:

  sycl::queue& trueStream() const
  {
    return *m_sycl_stream;
  }

 private:

  impl::IRunnerRuntime* m_runtime;
  std::unique_ptr<sycl::queue> m_sycl_stream;
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
#if defined(__ADAPTIVECPP__)
    m_recorded_stream = stream;
    // TODO: Vérifier s'il faut faire quelque chose
#elif defined(__INTEL_LLVM_COMPILER)
    auto* rq = static_cast<SyclRunQueueStream*>(stream);
    m_sycl_event = rq->trueStream().ext_oneapi_submit_barrier();
#else
    ARCANE_THROW(NotSupportedException, "Only supported for AdaptiveCpp and Intel DPC++ implementation");
#endif
  }

  void wait() final
  {
    //ARCANE_SYCL_FUNC_NOT_HANDLED;
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
    ARCANE_THROW(NotSupportedException, "Only supported for AdaptiveCpp and Intel DPC++ implementation");
#endif
  }

  Int64 elapsedTime([[maybe_unused]] IRunQueueEventImpl* start_event) final
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
    return 0;
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
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }
  void unsetMemoryAdvice([[maybe_unused]] ConstMemoryView buffer,
                         [[maybe_unused]] eMemoryAdvice advice, [[maybe_unused]] DeviceId device_id) override
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }

  void setCurrentDevice([[maybe_unused]] DeviceId device_id) final
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
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

  void fillDevicesAndSetDefaultQueue();
  sycl::queue& defaultQueue() const { return *m_default_queue; }
  sycl::device& defaultDevice() const { return *m_default_device; }

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
  if (bi.isDefault())
    m_sycl_stream = std::make_unique<sycl::queue>(runtime->defaultDevice(), sycl::property::queue::in_order());
  else {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
    m_sycl_stream = std::make_unique<sycl::queue>(runtime->defaultDevice(), sycl::property::queue::in_order());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SyclRunnerRuntime::
fillDevicesAndSetDefaultQueue()
{
  for (auto platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: "
              << platform.get_info<sycl::info::platform::name>()
              << std::endl;
  }

  sycl::device device{ sycl::gpu_selector_v };
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
  void copy(ConstMemoryView from, eMemoryRessource from_mem,
            MutableMemoryView to, eMemoryRessource to_mem,
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
copy(ConstMemoryView from, [[maybe_unused]] eMemoryRessource from_mem,
     MutableMemoryView to, [[maybe_unused]] eMemoryRessource to_mem,
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
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimesycl()
{
  using namespace Arcane;
  using namespace Arcane::Accelerator::Sycl;
  Arcane::Accelerator::impl::setUsingSYCLRuntime(true);
  Arcane::Accelerator::impl::setSYCLRunQueueRuntime(&global_sycl_runtime);
  Arcane::platform::setAcceleratorHostMemoryAllocator(getSyclMemoryAllocator());
  IMemoryRessourceMngInternal* mrm = platform::getDataMemoryRessourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryRessource::UnifiedMemory, getSyclUnifiedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::HostPinned, getSyclHostPinnedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::Device, getSyclDeviceMemoryAllocator());
  mrm->setCopier(&global_sycl_memory_copier);
  global_sycl_runtime.fillDevicesAndSetDefaultQueue();
  setSyclMemoryQueue(global_sycl_runtime.defaultQueue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
