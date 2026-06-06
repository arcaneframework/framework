// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeLoopMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Time loop manager interface.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMELOOPMNG_H
#define ARCANE_CORE_ITIMELOOPMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IBackwardMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eTimeLoopEventType
{
  BeginEntryPoint,
  EndEntryPoint,
  BeginIteration,
  EndIteration
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reason for stopping the code.
 */
enum class eTimeLoopStopReason
{
  //! Indicates that the code is not yet in stop mode.
  NoStop = 0,
  //! No specific reason
  NoReason = 1,
  //! Stop due to an error
  Error = 2,
  //! Stop because final time was reached
  FinalTimeReached = 3,
  //! Stop because maximum number of iterations specified was reached.
  MaxIterationReached = 4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for the time loop manager.
 *
 The time loop consists of three parts, executed in the following order:
 \arg initialization (Init)
 \arg compute loop (ComputeLoop)
 \arg termination (Exit)

 Initialization and termination are called only once.
 However, the compute loop is executed as long as no one requests
 an explicit stop via the stopComputeLoop() method.
 */
class ITimeLoopMng
{
 public:

  virtual ~ITimeLoopMng() {} //!< Frees resources.

 public:

  virtual void build() = 0;

  //virtual void initialize() =0;

 public:

  //!< Returns the sub-domain manager
  virtual ISubDomain* subDomain() const = 0;

  //! Executes the exit entry points
  virtual void execExitEntryPoints() = 0;

  //! Executes the build entry points
  virtual void execBuildEntryPoints() = 0;

  /*! \brief Executes the initialization entry points.
   * \param is_continue is true if resuming */
  virtual void execInitEntryPoints(bool is_continue) = 0;

  //! Executes the entry points after load balancing
  //virtual void execLoadBalanceEntryPoints() =0;

  //! Executes the entry points after load balancing
  virtual void execOnMeshChangedEntryPoints() = 0;

  //! Executes the entry points after refinement
  virtual void execOnMeshRefinementEntryPoints() = 0;

  /*!
   * \brief Indicates that the compute loop must stop.
   *
   * If \a is_final_time is true, it indicates that the final time has been reached.
   * If \a has_error is true, it indicates that the calculation stopped due to an
   * error. In this case, the application return code will be different from 0.
   */
  virtual void stopComputeLoop(bool is_final_time, bool has_error = false) = 0;

  //! Returns \a true if the final time has been reached.
  virtual bool finalTimeReached() const = 0;

  //! Returns the CPU time used in seconds.
  virtual Real cpuTimeUsed() const = 0;

  //! Returns the list of 'ComputeLoop' type entry points in the time loop.
  virtual EntryPointCollection loopEntryPoints() = 0;

  //! List of all entry points for the current time loop.
  virtual EntryPointCollection usedTimeLoopEntryPoints() = 0;

  /*!
   * Executes the next entry point.
   *
   * Returns in \a is_last \e true if the entry point that was just
   * executed is the last one of the iteration.
   */
  virtual void doExecNextEntryPoint(bool& is_last) = 0;

  //! Returns the next entry point to execute or 0 if there is none
  virtual IEntryPoint* nextEntryPoint() = 0;

  //! Returns the entry point currently being executed or 0 if there is none
  virtual IEntryPoint* currentEntryPoint() = 0;

  /*!
   * \brief Starts the execution of a compute loop iteration.
   *
   * \retval 0 if the code should continue.
   * \retval >0 if the calculation stops normally.
   * \retval <0 if the calculation stops due to an error.
   */
  virtual int doOneIteration() = 0;

  /*!
   * \brief Executes the compute loop.
   *
   * The compute loop is executed until the stopComputeLoop() method is called
   * or the number of loops performed equals \a max_loop if \a max_loop is not 0.
   * \retval 1 if the code stops normally due to final time reached
   * \retval 2 if the code stops normally due to \a max_loop reached
   * \retval <0 if the calculation stops due to an error.
   */
  virtual int doComputeLoop(Integer max_loop = 0) = 0;

  //@{
  //! Registration and selection of the time loop.
  /*!
   * \brief Registers a time loop.
   * Registers the time loop \a time_loop.
   *
   * If a time loop with the same name as \a time_loop is already referenced,
   * the new one replaces the old one.
   */
  virtual void registerTimeLoop(ITimeLoop* time_loop) = 0;

  /*! \brief Positions the time loop to be executed.
   * Selects the time loop named \a name as the one to be
   * executed. This method performs the following operations:
   * <ul>
   * <li>Starting from the name \a name, it searches for the time loop to use.
   * This time loop must have been referenced by the call to
   * registerTimeLoop()</li>
   * <li>For each entry point name of the time loop,
   * it searches for the corresponding entry point (IEntryPoint) registered in
   * the architecture</li>
   * <li>It constructs the list of entry points to be called during
   * initialization, in the compute loop, and during termination,
   * taking into account the entry points that are automatically loaded.</li>
   * <li>It determines the list of modules used by considering that a module
   * is used if and only if one of its entry points is used</li>
   * </ul>
   *
   * The operation fails and causes a fatal error in one of the following cases:
   * \arg this method has already been called,
   * \arg no time loop named \a name is registered,
   * \arg one of the entry point names in the list does not correspond to
   * any referenced entry point.
   *
   * If \a name is null, the time loop used is the default loop which contains no explicit entry points. It only contains
   * the automatically registered entry points.
   *
   * \retval true in case of error,
   * \retval false otherwise.
   */
  virtual void setUsedTimeLoop(const String& name) = 0;
  //@}

  //! Returns the time loop used
  virtual ITimeLoop* usedTimeLoop() const = 0;

  virtual void setBackwardMng(IBackwardMng* backward_mng) = 0;

  virtual IBackwardMng* getBackwardMng() const = 0;

  /*!
   * \brief Performs a backward step.
   *
   * This method just positions a marker. The backward step actually
   * takes place when the currently executing entry point finishes.
   *
   * After backward step, the backward entry points are called.
   *
   * \warning During parallel execution, this method must be
   * called by all sub-domains.
   */
  virtual void goBackward() = 0;

  /*! \brief True if currently in a backward step.
   *
   * A backward step is active as long as the physical time is less than
   * the physical time reached before the backward step trigger.
   */
  virtual bool isDoingBackward() = 0;

  /*!
   * \brief Schedules a mesh partitioning using the partition tool
   * \a mesh_partitioner.
   *
   * This method just positions a marker. The partitioning actually
   * takes place when the last entry point of the compute loop is finished (end of an iteration).
   *
   * After partitioning, the mesh change entry points are called.
   *
   * \warning During parallel execution, this method must be
   * called by all sub-domains.
   */
  virtual void registerActionMeshPartition(IMeshPartitionerBase* mesh_partitioner) = 0;

  /*!
   * \brief Positions the period between two saves for backward step.
   * If this value is null, backward step is disabled.
   */
  virtual void setBackwardSavePeriod(Integer n) = 0;

  /*!
   * \brief Positions the state of the verification mode
   */
  virtual void setVerificationActive(bool is_active) = 0;

  /*!
   * \brief Performs a verification.
   *
   * This operation is collective.
   *
   * This operation allows manually performing a verification operation,
   * whose name is \a name. This name \a name must be unique for
   * given iteration.
   */
  virtual void doVerification(const String& name) = 0;

  /*!
   * \brief Returns in \a names the list of time loop names.
   */
  virtual void timeLoopsName(StringCollection& names) const = 0;

  //! Returns in \a time_loops the list of time loops.
  virtual void timeLoops(TimeLoopCollection& time_loops) const = 0;

  //! Creates a time loop named \a name.
  virtual ITimeLoop* createTimeLoop(const String& name) = 0;

  //! Number of compute loops performed.
  virtual Integer nbLoop() const = 0;

  /*!
   * \brief Observable on the instance.
   *
   * The type of the observable is given by \a type
   */
  virtual IObservable* observable(eTimeLoopEventType type) = 0;

  //! Positions the reason for stopping the code
  virtual void setStopReason(eTimeLoopStopReason reason) = 0;

  /*!
   * \brief Reason for stopping the code.
   *
   * If the value is eTimeLoopStopReason::NoStop, then the code
   * is not stopping.
   */
  virtual eTimeLoopStopReason stopReason() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
