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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SyclRunnerRuntime
: public impl::IRunnerRuntime
{
 public:

  void notifyBeginLaunchKernel() override
  {
  }
  void notifyEndLaunchKernel() override
  {
  }
  void barrier() override
  {
    ARCANE_FATAL("NYI");
  }
  eExecutionPolicy executionPolicy() const override
  {
    ARCANE_FATAL("NYI");
  }
  impl::IRunQueueStream* createStream(const RunQueueBuildInfo& bi) override
  {
    ARCANE_FATAL("NYI");
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
    ARCANE_FATAL("NYI");
  }
  void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) override
  {
    ARCANE_FATAL("NYI");
  }

  void setCurrentDevice(DeviceId device_id) final
  {
    ARCANE_FATAL("NYI");
  }
  const IDeviceInfoList* deviceInfoList() override { return &m_device_info_list; }

  void getPointerAttribute(PointerAttribute& attribute, const void* ptr) override
  {
    ARCANE_FATAL("NYI");
  }

  void fillDevices();

 private:

  bool m_is_verbose = false;
  impl::DeviceInfoList m_device_info_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SyclRunnerRuntime::
fillDevices()
{
  std::cout << "FILLING DEVICES\n";
  for (auto platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: "
              << platform.get_info<sycl::info::platform::name>()
              << std::endl;

    for (auto device : platform.get_devices()) {
      std::cout << "\tDevice: "
                << device.get_info<sycl::info::device::name>()
                << std::endl;
    }
  }
  std::cout << "END FILLING DEVICES\n";
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
  Arcane::platform::setAcceleratorHostMemoryAllocator(getSyclMemoryAllocator());
  IMemoryRessourceMngInternal* mrm = platform::getDataMemoryRessourceMng()->_internal();
  mrm->setIsAccelerator(true);
  mrm->setAllocator(eMemoryRessource::UnifiedMemory, getSyclUnifiedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::HostPinned, getSyclHostPinnedMemoryAllocator());
  mrm->setAllocator(eMemoryRessource::Device, getSyclDeviceMemoryAllocator());
  global_sycl_runtime.fillDevices();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
