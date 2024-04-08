// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/DeviceInfoList.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/internal/RunCommandImpl.h"

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
    ARCANE_FATAL("NYI");
  }
  void prefetchMemory(const MemoryPrefetchArgs& args) override
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
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }
  eExecutionPolicy executionPolicy() const override
  {
    ARCANE_FATAL("NYI");
  }
  impl::IRunQueueStream* createStream(const RunQueueBuildInfo& bi) override
  {
    return new SyclRunQueueStream(this, bi);
  }
  impl::IRunQueueEventImpl* createEventImpl() override
  {
    ARCANE_FATAL("NYI");
  }
  impl::IRunQueueEventImpl* createEventImplWithTimer() override
  {
    ARCANE_FATAL("NYI");
  }
  void setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }
  void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }

  void setCurrentDevice(DeviceId device_id) final
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }
  const IDeviceInfoList* deviceInfoList() override { return &m_device_info_list; }

  void getPointerAttribute(PointerAttribute& attribute, const void* ptr) override
  {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
  }

  void fillDevicesAndSetDefaultQueue();
  sycl::queue& defaultQueue() const { return *m_default_queue; }
  sycl::device& defaultDevice() const { return *m_default_device; }

 private:

  bool m_is_verbose = false;
  impl::DeviceInfoList m_device_info_list;
  std::unique_ptr<sycl::device> m_default_device;
  std::unique_ptr<sycl::queue> m_default_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SyclRunQueueStream::
SyclRunQueueStream(SyclRunnerRuntime* runtime, const RunQueueBuildInfo& bi)
: m_runtime(runtime)
{
  if (bi.isDefault())
    m_sycl_stream = std::make_unique<sycl::queue>(runtime->defaultDevice());
  else {
    ARCANE_SYCL_FUNC_NOT_HANDLED;
    m_sycl_stream = std::make_unique<sycl::queue>(runtime->defaultDevice());
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
  m_default_device = std::make_unique<sycl::device>(device);
  m_default_queue = std::make_unique<sycl::queue>(device);

  DeviceInfo device_info;
  device_info.setDescription("No description info");
  device_info.setDeviceId(DeviceId(0));
  device_info.setName(device.get_info<sycl::info::device::name>());
  m_device_info_list.addDevice(device_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Sycl

namespace
{
Arcane::Accelerator::Sycl::SyclRunnerRuntime global_sycl_runtime;
} // namespace

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
  global_sycl_runtime.fillDevicesAndSetDefaultQueue();
  setSyclMemoryQueue(global_sycl_runtime.defaultQueue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
