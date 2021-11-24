// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HipAcceleratorRuntime.cc                                    (C) 2000-2021 */
/*                                                                           */
/* Runtime pour 'HIP'.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/hip/HipAccelerator.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/IRunQueueRuntime.h"
#include "arcane/accelerator/IRunQueueStream.h"
#include "arcane/accelerator/RunCommand.h"

#include <iostream>

using namespace Arccore;

namespace Arcane::Accelerator::Hip
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void checkDevices()
{
  int nb_device = 0;
  ARCANE_CHECK_HIP(hipGetDeviceCount(&nb_device));
  std::ostream& o = std::cout;
  o << "Initialize Arcane CUDA runtime\n";
  o << "Available device = " << nb_device << "\n";
  for( int i=0; i<nb_device; ++i ){
    hipDeviceProp_t dp;
    ARCANE_CHECK_HIP(hipGetDeviceProperties(&dp, i));

    o << "\nDevice " << i << " name=" << dp.name << "\n";
    o << " computeCapability = " << dp.major << "." << dp.minor << "\n";
    o << " totalGlobalMem = " << dp.totalGlobalMem << "\n";
    o << " sharedMemPerBlock = " << dp.sharedMemPerBlock << "\n";
    o << " regsPerBlock = " << dp.regsPerBlock << "\n";
    o << " warpSize = " << dp.warpSize<< "\n";
    o << " memPitch = " << dp.memPitch<< "\n";
    o << " maxThreadsPerBlock = " << dp.maxThreadsPerBlock<< "\n";
    o << " totalConstMem = " << dp.totalConstMem<< "\n";
    o << " clockRate = " << dp.clockRate<< "\n";
    //o << " deviceOverlap = " << dp.deviceOverlap<< "\n";
    o << " multiProcessorCount = " << dp.multiProcessorCount<< "\n";
    o << " kernelExecTimeoutEnabled = " << dp.kernelExecTimeoutEnabled<< "\n";
    o << " integrated = " << dp.integrated<< "\n";
    o << " canMapHostMemory = " << dp.canMapHostMemory<< "\n";
    o << " computeMode = " << dp.computeMode<< "\n";
    o << " maxThreadsDim = "<< dp.maxThreadsDim[0] << " " << dp.maxThreadsDim[1]
      << " " << dp.maxThreadsDim[2] << "\n";
    o << " maxGridSize = "<< dp.maxGridSize[0] << " " << dp.maxGridSize[1]
      << " " << dp.maxGridSize[2] << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunQueueStream
: public IRunQueueStream
{
 public:
  HipRunQueueStream(IRunQueueRuntime* runtime)
  : m_runtime(runtime)
  {
    ARCANE_CHECK_HIP(hipStreamCreate(&m_hip_stream));
  }
  ~HipRunQueueStream() noexcept(false) override
  {
    ARCANE_CHECK_HIP(hipStreamDestroy(m_hip_stream));
  }
 public:
  void notifyBeginKernel([[maybe_unused]] RunCommand& c) override
  {
    return m_runtime->notifyBeginKernel();
  }
  void notifyEndKernel(RunCommand&) override
  {
    return m_runtime->notifyEndKernel();
  }
  void barrier() override
  {
    ARCANE_CHECK_HIP(hipStreamSynchronize(m_hip_stream));
  }
  void* _internalImpl() override { return &m_hip_stream; }
 private:
  IRunQueueRuntime* m_runtime;
  hipStream_t m_hip_stream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HipRunQueueRuntime
: public IRunQueueRuntime
{
 public:
  ~HipRunQueueRuntime() override = default;
 public:
  void notifyBeginKernel() override
  {
    ++m_nb_kernel_launched;
    if (m_is_verbose)
      std::cout << "BEGIN HIP KERNEL!\n";
  }
  void notifyEndKernel() override
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
  IRunQueueStream* createStream() override
  {
    return new HipRunQueueStream(this);
  }
 private:
  Int64 m_nb_kernel_launched = 0;
  bool m_is_verbose = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Hip

namespace 
{
Arcane::Accelerator::Hip::HipRunQueueRuntime global_hip_runtime;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette fonction est le point d'entrée utilisé lors du chargement
// dynamique de cette bibliothèque
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimehip()
{
  using namespace Arcane::Accelerator::Hip;
  Arcane::Accelerator::impl::setUsingHIPRuntime(true);
  Arcane::Accelerator::impl::setHIPRunQueueRuntime(&global_hip_runtime);
  Arcane::platform::setAcceleratorHostMemoryAllocator(getHipMemoryAllocator());
  checkDevices();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
