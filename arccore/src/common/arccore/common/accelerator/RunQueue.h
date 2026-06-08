// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueue.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Execution queue management on an accelerator.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNQUEUE_H
#define ARCCORE_COMMON_ACCELERATOR_RUNQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/AutoRef2.h"

#include "arccore/common/accelerator/RunCommand.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution queue for an accelerator.
 *
 * This class uses a reference semantics. The execution queue is
 * destroyed when the last reference to it is destroyed.
 *
 * A queue is attached to a Runner instance and allows executing
 * commands (RunCommand) on an accelerator or on the CPU. The method
 * executionPolicy() allows knowing where the commands from
 * the queue will be executed.
 *
 * Instances of this class are created by calling makeQueue(Runner).
 * Calculation kernels (RunCommand) can then be created via the call
 * to makeCommand().
 *
 * The default constructor builds a null queue that cannot be
 * used to launch commands. The only operations allowed on
 * the null queue are isNull(), executionPolicy(), isAcceleratorPolicy(),
 * barrier(), allocationOptions(), and memoryResource().
 *
 * The methods of this class are not thread-safe for the same instance.
 */
class ARCCORE_COMMON_EXPORT RunQueue
{
  friend RunCommand;
  friend ProfileRegion;
  friend Runner;
  friend ViewBuildInfo;
  friend class Impl::RunCommandLaunchInfo;
  friend RunCommand makeCommand(const RunQueue& run_queue);
  friend RunCommand makeCommand(const RunQueue* run_queue);
  // For _internalNativeStream()
  friend class Impl::CudaUtils;
  friend class Impl::HipUtils;
  friend class Impl::SyclUtils;

 public:

  //! Allows modifying the queue's asynchronous state during the
  //! instance's lifetime
  class ScopedAsync
  {
   public:

    explicit ScopedAsync(RunQueue* queue)
    : m_queue(queue)
    {
      // Makes the queue asynchronous
      if (m_queue) {
        m_is_async = m_queue->isAsync();
        m_queue->setAsync(true);
      }
    }
    ~ScopedAsync() noexcept(false)
    {
      // Restores the queue to its original state when the
      // constructor is called
      if (m_queue)
        m_queue->setAsync(m_is_async);
    }

   private:

    RunQueue* m_queue = nullptr;
    bool m_is_async = false;
  };

 public:

  //! Creates a null queue.
  RunQueue();
  ~RunQueue();

 public:

  //! Creates a queue associated with \a runner with default parameters
  ARCCORE_DEPRECATED_REASON("Y2024: Use makeQueue(runner) instead")
  explicit RunQueue(const Runner& runner);
  //! Creates a queue associated with \a runner with parameters \a bi
  ARCCORE_DEPRECATED_REASON("Y2024: Use makeQueue(runner,bi) instead")
  RunQueue(const Runner& runner, const RunQueueBuildInfo& bi);

 public:

  RunQueue(const RunQueue&);
  RunQueue& operator=(const RunQueue&);
  RunQueue(RunQueue&&) noexcept;
  RunQueue& operator=(RunQueue&&) noexcept;

 public:

  //! Indicates if the RunQueue is null
  bool isNull() const { return !m_p; }

  //! Execution policy of the queue.
  eExecutionPolicy executionPolicy() const;
  //! Indicates if the instance is associated with an accelerator
  bool isAcceleratorPolicy() const;

  /*!
   * \brief Sets the instance's asynchronous state.
   *
   * If the instance is asynchronous, the different commands
   * associated are non-blocking and you must explicitly call barrier()
   * to wait for the commands to finish execution.
   *
   * \pre !isNull()
   */
  void setAsync(bool v);
  //! Indicates if the execution queue is asynchronous.
  bool isAsync() const;

  /*!
   * \brief Sets the instance's asynchronous state.
   *
   * Returns the instance.
   *
   * \pre !isNull()
   * \sa setAsync().
   */
  const RunQueue& addAsync(bool is_async) const;

  //! Blocks until all commands associated with the queue are finished.
  void barrier() const;

  //! Copies information between two memory regions
  void copyMemory(const MemoryCopyArgs& args) const;
  //! Performs a memory prefetch
  void prefetchMemory(const MemoryPrefetchArgs& args) const;

  /*!
   * \name Event Management
   * \pre !isNull()
   */
  //!@{
  //! Records the instance's state in \a event.
  void recordEvent(RunQueueEvent& event);
  //! Records the instance's state in \a event.
  void recordEvent(Ref<RunQueueEvent>& event);
  //! Blocks execution on the instance until the jobs recorded in \a event are finished
  void waitEvent(RunQueueEvent& event);
  //! Blocks execution on the instance until the jobs recorded in \a event are finished
  void waitEvent(Ref<RunQueueEvent>& event);
  //!@}

  //! \name Memory Management
  //!@{
  /*!
   * \brief Allocation options associated with this queue.
   *
   * It is possible to change the memory resource and thus the allocator used
   * via setMemoryRessource().
   */
  MemoryAllocationOptions allocationOptions() const;

  /*!
   * \brief Sets the memory resource used for allocations with this instance.
   *
   * The default value is eMemoryRessource::UnifiedMemory
   * if isAcceleratorPolicy()==true and eMemoryRessource::Host otherwise.
   *
   * \sa memoryResource()
   * \sa allocationOptions()
   *
   * \pre !isNull()
   */
  void setMemoryRessource(eMemoryResource mem);

  //! Memory resource used for allocations with this instance.
  eMemoryResource memoryRessource() const;
  //! Memory resource used for allocations with this instance.
  eMemoryResource memoryResource() const;
  //!@}

 public:

  /*!
   * \brief Indicates if the creation of RunCommand for this instance
   * is allowed from multiple threads.
   *
   * This requires using a lock (like std::mutex) and may degrade
   * performance. The default is \a false.
   *
   * This method is not supported for queues that are associated
   * with accelerators (isAcceleratorPolicy()==true)
   */
  void setConcurrentCommandCreation(bool v);
  //! Indicates if concurrent creation of multiple RunCommands is allowed
  bool isConcurrentCommandCreation() const;

 public:

  /*!
   * \brief Pointer to the internal structure dependent on the implementation.
   *
   * This method is reserved for advanced usage.
   * The returned queue must not be kept beyond the life of the instance.
   *
   * With CUDA, the returned pointer is a 'cudaStream_t*'. With HIP, it
   * is a 'hipStream_t*'.
   *
   * \deprecated Use toCudaNativeStream(), toHipNativeStream()
   * or toSyclNativeStream() instead
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use toCudaNativeStream(), toHipNativeStream() or toSyclNativeStream() instead")
  void* platformStream() const;

 public:

  friend bool operator==(const RunQueue& q1, const RunQueue& q2)
  {
    return q1.m_p.get() == q2.m_p.get();
  }
  friend bool operator!=(const RunQueue& q1, const RunQueue& q2)
  {
    return q1.m_p.get() != q2.m_p.get();
  }

 public:

  Impl::RunQueueImpl* _internalImpl() const;

 private:

  // Creation methods are reserved for Runner.
  // An extra unused argument is added to avoid using
  // the deprecated constructor.
  RunQueue(const Runner& runner, bool);
  //! Creates a queue associated with \a runner with parameters \a bi
  RunQueue(const Runner& runner, const RunQueueBuildInfo& bi, bool);
  explicit RunQueue(Impl::RunQueueImpl* p);

 private:

  Impl::IRunnerRuntime* _internalRuntime() const;
  Impl::IRunQueueStream* _internalStream() const;
  Impl::RunCommandImpl* _getCommandImpl() const;
  Impl::NativeStream _internalNativeStream() const;
  void _checkNotNull() const;

  // For VariableViewBase
  friend class VariableViewBase;
  friend class NumArrayViewBase;

 private:

  AutoRef2<Impl::RunQueueImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a command associated with the queue \a run_queue.
 */
inline RunCommand
makeCommand(const RunQueue& run_queue)
{
  run_queue._checkNotNull();
  return RunCommand(run_queue);
}

/*!
 * \brief Creates a command associated with the queue \a run_queue.
 */
inline RunCommand
makeCommand(const RunQueue* run_queue)
{
  ARCCORE_CHECK_POINTER(run_queue);
  run_queue->_checkNotNull();
  return RunCommand(*run_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
