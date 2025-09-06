// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunnerRuntime.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface du runtime associé à une RunQueue.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_INTERNAL_IRUNQUEUERUNTIME_H
#define ARCANE_ACCELERATOR_INTERNAL_IRUNQUEUERUNTIME_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface du runtime associé à un accélérateur.
 *
 * Il existe actuellement trois implémentations accélérateurs pour cette
 * interface: CUDA, ROCM, SYCL.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime
{
 public:

  virtual ~IRunnerRuntime() = default;

 public:

  virtual void notifyBeginLaunchKernel() = 0;
  virtual void notifyEndLaunchKernel() = 0;
  virtual void barrier() = 0;
  virtual eExecutionPolicy executionPolicy() const = 0;
  virtual IRunQueueStream* createStream(const RunQueueBuildInfo& bi) = 0;
  virtual impl::IRunQueueEventImpl* createEventImpl() = 0;
  virtual impl::IRunQueueEventImpl* createEventImplWithTimer() = 0;
  virtual void setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) = 0;
  virtual void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice, DeviceId device_id) = 0;
  virtual void setCurrentDevice(DeviceId device_id) = 0;
  virtual const IDeviceInfoList* deviceInfoList() = 0;
  virtual void getPointerAttribute(PointerAttribute& attribute, const void* ptr) = 0;
  virtual DeviceMemoryInfo getDeviceMemoryInfo(DeviceId device_id) =0;

  virtual void startProfiling() {}
  virtual void stopProfiling() {}
  virtual bool isProfilingActive() { return false; }
  virtual void pushProfilerRange([[maybe_unused]] const String& name, [[maybe_unused]] Int32 color_gdb) {}
  virtual void popProfilerRange() {}
  virtual void finalize(ITraceMng*) {}
  virtual KernelLaunchArgs computeKernalLaunchArgs(const KernelLaunchArgs& orig_args,
                                                   const void* kernel_ptr,
                                                   Int64 total_loop_size,
                                                   Int32 wanted_shared_memory);

 protected:

  void _fillPointerAttribute(PointerAttribute& attribute,
                             ePointerMemoryType mem_type,
                             int device, const void* pointer, const void* device_pointer,
                             const void* host_pointer);
  void _fillPointerAttribute(PointerAttribute& attribute, const void* pointer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
