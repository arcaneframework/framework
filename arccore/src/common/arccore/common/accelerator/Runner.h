// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Runner.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Execution management on accelerator.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNNER_H
#define ARCCORE_COMMON_ACCELERATOR_RUNNER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Ref.h"

#include "arccore/common/accelerator/RunQueue.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution manager for accelerator.
 *
 * This class uses reference semantics
 *
 * An instance of this class represents an execution backend. It must be
 * initialized using initialize() before its methods can be used, or one
 * of the constructors other than the default constructor must be called.
 * The backend used is chosen via the eExecutionPolicy enumeration. The
 * backends are of two types:
 * - backends that run on the host: eExecutionPolicy::Sequential
 * and eExecutionPolicy::Thread,
 * - backends that run on accelerators: eExecutionPolicy::CUDA,
 * eExecutionPolicy::HIP and eExecutionPolicy::SYCL.
 *
 * The function \arcaneacc{isAcceleratorPolicy()} allows you to know if an
 * eExecutionPolicy is associated with an accelerator.
 *
 * If an instance of this class is associated with an accelerator, that
 * accelerator is not necessarily the one used by default for the current thread.
 * To ensure that the kernels associated with this runner are executed on the
 * correct device, it is necessary to call the setAsCurrentDevice() method at
 * least once, and to do so again if another part of the code or an external
 * library changes the default accelerator.
 *
 * The Runner class allows creating execution queues (RunQueue) via the
 * makeQueue() function. These queues can then be used to launch commands
 * (RunCommand). The page \ref arcanedoc_acceleratorapi describes the
 * operation of the accelerator API.
 */
class ARCCORE_COMMON_EXPORT Runner
{
  friend Impl::RunQueueImpl;
  friend Impl::RunCommandImpl;
  friend RunQueue;
  friend RunQueueEvent;
  friend Impl::RunnerImpl;

  friend RunQueue makeQueue(const Runner& runner);
  friend RunQueue makeQueue(const Runner* runner);
  friend RunQueue makeQueue(const Runner& runner, const RunQueueBuildInfo& bi);
  friend RunQueue makeQueue(const Runner* runner, const RunQueueBuildInfo& bi);
  friend Ref<RunQueue> makeQueueRef(const Runner& runner);
  friend Ref<RunQueue> makeQueueRef(Runner& runner, const RunQueueBuildInfo& bi);
  friend Ref<RunQueue> makeQueueRef(Runner* runner);

 public:

  /*!
   * \brief Creates an uninitialized execution manager.
   *
   * initialize() must be called before the instance can be used
   */
  Runner();
  //! Creates and initializes a manager for the accelerator \a p
  explicit Runner(eExecutionPolicy p);
  //! Creates and initializes a manager for the accelerator \a p and the device \a device
  Runner(eExecutionPolicy p, DeviceId device);

 public:

  //! Associated execution policy
  eExecutionPolicy executionPolicy() const;

  //! Initializes the instance. This method must be called only once.
  void initialize(eExecutionPolicy v);

  //! Initializes the instance. This method must be called only once.
  void initialize(eExecutionPolicy v, DeviceId device);

  //! Indicates whether the instance has been initialized
  bool isInitialized() const;

  /*!
   * \brief Indicates whether multiple threads are allowed to create RunQueues.
   *
   * \deprecated Queue creation is always thread-safe since Arcane version
   * 3.15.
   */
  ARCCORE_DEPRECATED_REASON("Y2025: this method is a no op. Concurrent queue creation is always thread-safe")
  void setConcurrentQueueCreation(bool v);

  //! Indicates whether concurrent creation of multiple RunQueues is allowed
  bool isConcurrentQueueCreation() const;

  /*!
   * \brief Total time spent in commands associated with this instance.
   *
   * This time is only meaningful if the RunQueues are synchronous.
   */
  double cumulativeCommandTime() const;

  //! Sets the execution policy for reductions
  ARCCORE_DEPRECATED_REASON("Y2025: this method is a no op. reduce policy is always eDeviceReducePolicy::Grid")
  void setDeviceReducePolicy(eDeviceReducePolicy v);

  //! Reduction execution policy
  eDeviceReducePolicy deviceReducePolicy() const;

  //! Sets memory advice for a memory region
  void setMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice);

  //! Unsets memory advice for a memory region
  void unsetMemoryAdvice(ConstMemoryView buffer, eMemoryAdvice advice);

  //! Device associated with this instance.
  DeviceId deviceId() const;

  /*!
   * \brief Sets the device associated with this instance as the default
   * context device.
   *
   * This call is equivalent to cudaSetDevice() or hipSetDevice();
   */
  void setAsCurrentDevice();

  //! Information about the device associated with this instance.
  const DeviceInfo& deviceInfo() const;

  //! Information about the device associated with this instance.
  DeviceMemoryInfo deviceMemoryInfo() const;

  //! Fills \a attr with information concerning the memory region pointed to by \a ptr
  void fillPointerAttribute(PointerAttribute& attr, const void* ptr);

 public:

  /*!
   * \brief List of devices for the execution policy \a policy.
   *
   * If the associated runtime has not yet been initialized, this method
   * returns \a nullptr.
   */
  static const IDeviceInfoList* deviceInfoList(eExecutionPolicy policy);

 private:

  // Creation is reserved for the global methods makeQueue()
  static RunQueue _makeQueue(const Runner& runner)
  {
    return RunQueue(runner, true);
  }
  static RunQueue _makeQueue(const Runner& runner, const RunQueueBuildInfo& bi)
  {
    return RunQueue(runner, bi, true);
  }
  static Ref<RunQueue> _makeQueueRef(const Runner& runner)
  {
    return makeRef(new RunQueue(runner, true));
  }
  static Ref<RunQueue> _makeQueueRef(Runner& runner, const RunQueueBuildInfo& bi)
  {
    return makeRef(new RunQueue(runner, bi, true));
  }

 public:

  //! Internal API for %Arcane
  RunnerInternal* _internalApi();

 private:

  Impl::IRunnerRuntime* _internalRuntime() const;
  Impl::RunnerImpl* _impl() const { return m_p.get(); }

 private:

  std::shared_ptr<Impl::RunnerImpl> m_p;

 private:

  void _checkIsInit() const;
  bool _isAutoPrefetchCommand() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a queue associated with \a runner.
 *
 * This call is thread-safe.
 */
inline RunQueue
makeQueue(const Runner& runner)
{
  return Runner::_makeQueue(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a queue associated with \a runner.
 *
 * This call is thread-safe.
 */
inline RunQueue
makeQueue(const Runner* runner)
{
  ARCCORE_CHECK_POINTER(runner);
  return Runner::_makeQueue(*runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a queue associated with \a runner with properties \a bi.
 *
 * This call is thread-safe.
 */
inline RunQueue
makeQueue(const Runner& runner, const RunQueueBuildInfo& bi)
{
  return Runner::_makeQueue(runner, bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a queue associated with \a runner with properties \a bi.
 *
 * This call is thread-safe.
 */
inline RunQueue
makeQueue(const Runner* runner, const RunQueueBuildInfo& bi)
{
  ARCCORE_CHECK_POINTER(runner);
  return Runner::_makeQueue(*runner, bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a reference to a queue with the default execution policy
 * of \a runner.
 *
 * If the queue is temporary, it is preferable to use makeQueue() instead
 * to avoid unnecessary allocation.
 */
inline Ref<RunQueue>
makeQueueRef(const Runner& runner)
{
  return Runner::_makeQueueRef(runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a reference to a queue with the default execution policy
 * of \a runner.
 *
 * If the queue is temporary, it is preferable to use makeQueue() instead
 * to avoid unnecessary allocation.
 */
inline Ref<RunQueue>
makeQueueRef(Runner& runner, const RunQueueBuildInfo& bi)
{
  return Runner::_makeQueueRef(runner, bi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a reference to a queue with the default execution policy
 * of \a runner.
 *
 * If the queue is temporary, it is preferable to use makeQueue() instead
 * to avoid unnecessary allocation.
 */
inline Ref<RunQueue>
makeQueueRef(Runner* runner)
{
  ARCCORE_CHECK_POINTER(runner);
  return Runner::_makeQueueRef(*runner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
