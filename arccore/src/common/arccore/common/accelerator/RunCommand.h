// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.h                                                (C) 2000-2026 */
/*                                                                           */
/* Management of an accelerator command.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNCOMMAND_H
#define ARCCORE_COMMON_ACCELERATOR_RUNCOMMAND_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
namespace Impl
{
extern "C++" ARCCORE_COMMON_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Management of an accelerator command.
 *
 * A command is associated with an execution queue (RunQueue) and its lifespan
 * must not exceed that of the latter.
 *
 * A command is an operation that will be executed on the accelerator
 * associated with the RunQueue instance used when calling makeCommand().
 * On a GPU, this corresponds to a kernel.
 *
 * For more information, refer to the section
 * \ref arcanedoc_parallel_accelerator_runcommand.
 */
class ARCCORE_COMMON_EXPORT RunCommand
{
  friend Impl::IReduceMemoryImpl* Impl::internalGetOrCreateReduceMemoryImpl(RunCommand* command);
  friend Impl::RunCommandLaunchInfo;
  friend Impl::RunQueueImpl;
  friend class ViewBuildInfo;
  template <typename T, Int32 Extent> friend class LocalMemory;

  friend RunCommand makeCommand(const RunQueue& run_queue);
  friend RunCommand makeCommand(const RunQueue* run_queue);

 public:

  ~RunCommand();

 protected:

  explicit RunCommand(const RunQueue& run_queue);

 public:

  RunCommand(RunCommand&& command) = delete;
  RunCommand(const RunCommand&) = delete;
  RunCommand& operator=(const RunCommand&) = delete;
  RunCommand& operator=(RunCommand&&) = delete;

 public:

  //! Command execution policy
  eExecutionPolicy executionPolicy() const;

  /*!
   * \brief Sets the trace information.
   *
   * This information is used for tracing or debugging.
   * The RUNCOMMAND_LOOP or RUNCOMMAND_ENUMERATE macros automatically call
   * this method.
   */
  RunCommand& addTraceInfo(const TraceInfo& ti);

  /*!
   * \brief Sets the kernel name.
   *
   * This name is used for tracing or debugging.
   */
  RunCommand& addKernelName(const String& v);

  /*!
   * \brief Sets the number of threads per block for accelerators.
   *
   * If the value \a v is zero, the default choice is used.
   * If the value \a v is positive, its minimum valid value depends
   * on the accelerator. Generally, it is at least 32.
   */
  RunCommand& addNbThreadPerBlock(Int32 v);

  /*!
   * \brief Sets the number of strides for loop decomposition
   * on accelerator/
   *
   * The default value is 1, which indicates that the loop is not decomposed.
   * This method does nothing if the command is not executed
   * on an accelerator. This value is only used for classical loops
   * (RUNCOMMAND_LOOP()) or on entities (RUNCOMMAND_ENUMERATE()).
   *
   * \warning EXPERIMENTAL API. TO BE USED ONLY IN ARCANE
   */
  RunCommand& addNbStride(Int32 v);

  //! Number of loop decomposition strides
  Int32 nbStride() const;

  //! Trace information
  const TraceInfo& traceInfo() const;

  //! Kernel name
  const String& kernelName() const;

  /*
   * \brief Number of threads per block or 0 for the default value.
   *
   * This value is used only if running on an accelerator.
   */
  Int32 nbThreadPerBlock() const;

  //! Sets the multi-thread loop configuration
  void setParallelLoopOptions(const ParallelLoopOptions& opt);

  //! Multi-thread loop configuration
  const ParallelLoopOptions& parallelLoopOptions() const;

  //! Displaying command information
  friend ARCCORE_COMMON_EXPORT RunCommand&
  operator<<(RunCommand& command, const TraceInfo& trace_info);

 private:

  // For RunCommandLaunchInfo
  void _internalNotifyBeginLaunchKernel();
  void _internalNotifyEndLaunchKernel();
  void _internalNotifyBeginLaunchKernelSyclEvent(void* sycl_event_ptr);
  ForLoopOneExecStat* _internalCommandExecStat();

 private:

  //! \internal
  Impl::RunQueueImpl* _internalQueueImpl() const;
  Impl::NativeStream _internalNativeStream() const;
  static Impl::RunCommandImpl* _internalCreateImpl(Impl::RunQueueImpl* queue);
  static void _internalDestroyImpl(Impl::RunCommandImpl* p);
  Int32 _addSharedMemory(Int32 size);
  Int32 _sharedMemory() const;

 private:

  void _allocateReduceMemory(Int32 nb_grid);

 private:

  Impl::RunCommandImpl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
