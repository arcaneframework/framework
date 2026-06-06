// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelSuperMng.h                                         (C) 2000-2025 */
/*                                                                           */
/* Parallelism supervisor interface.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELSUPERMNG_H
#define ARCANE_CORE_IPARALLELSUPERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;
class IParallelMng;
class IThreadMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Abstract class of the parallelism supervisor.
 */
class ARCANE_CORE_EXPORT IParallelSuperMng
{
 public:

  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;

 public:
	
  virtual ~IParallelSuperMng() {} //!< Frees resources.

 public:

  /*!
   * \brief Constructs the instance members.
   *
   * The instance is not usable until this method has been called.
   * This method must be called before initialize().
   *
   * \warning This method must only be called once.
   */
  virtual void build() =0;

  /*!
   * \brief Initializes the instance.
   *
   * The instance is not usable until this method has been called.
   * \warning This method must only be called once.
   */
  virtual void initialize() =0;

 public:

  //! Returns the main manager.
  virtual IApplication* application() const =0;

  //! Thread manager.
  virtual IThreadMng* threadMng() const =0;
	
  //! Returns true if the execution is parallel
  virtual bool isParallel() const =0;

  //! Returns the process number (between 0 and nbProcess()-1)
  virtual Int32 commRank() const =0;

  //! Returns the total number of processes used
  virtual Int32 commSize() const =0;

  //! Rank of this instance for traces.
  virtual Int32 traceRank() const =0;

  /*!
   * \brief Address of the MPI communicator associated with this manager.
   *
   * The communicator is only valid if MPI is used. Otherwise, the returned address
   * is 0. The returned value is of type (MPI_Comm*).
   */

  virtual void* getMPICommunicator() =0;

  /*!
   * \brief MPI communicator associated with this manager
   *
   * \sa IParallelMng::communicator()
   */
  virtual Parallel::Communicator communicator() const =0;

  /*!
   * \internal
   * \brief Creates a parallelism manager for all allocated cores.
   *
   * This operation is collective.
   *
   * This method must only be called once during initialization.
   *
   * \a local_rank is the local rank of the caller in the list of ranks.
   * In pure MPI mode, this rank is always 0 because there is only one
   * thread. In Thread or Thread/MPI mode, it is the rank of the thread used
   * during creation.
   *
   * The returned manager remains the property of this instance and 
   * must not be destroyed.
   *
   * For internal use only.
   */
  virtual Ref<IParallelMng> internalCreateWorldParallelMng(Int32 local_rank) =0;

  /*!
   * \brief Number of subdomains to create locally.
   * - 1 if sequential.
   * - 1 if pure MPI
   * - n if THREAD or THREAD/MPI
   */
  virtual Int32 nbLocalSubDomain() =0;

  /*!
   * \brief Attempts to abort.
   *
   * This method is called when an exception has been generated and the
   * current execution case must stop. It allows performing cleanup operations
   * on the manager if necessary.
   */
  virtual void tryAbort() =0;

  //! Returns true if the instance is a master I/O manager.
  virtual bool isMasterIO() const =0;

  /*!
    \brief Rank of the instance managing input/output (for which isMasterIO() is true)
    *
    * In the current implementation, this is always the rank 0 processor.
    */
  virtual Int32 masterIORank() const =0;

  /*!
   * \brief Parallelism manager for all allocated resources.
   */
  //virtual IParallelMng* worldParallelMng() const =0;

  //! Performs a barrier
  virtual void barrier() =0;

 public:

  //! @name broadcast operations
  //@{
  /*!
   * \brief Sends an array of values to all processes
   * This operation synchronizes the value array send_buf across all
   * processes. The array used is that of the process whose
   * identifier (processId()) is process_id.
   * All processes must call this method with
   * the same parameter process_id and have a send_buf array
   * containing the same number of elements.
   */
  virtual void broadcast(ByteArrayView send_buf,Integer process_id) =0;
  virtual void broadcast(Int32ArrayView send_buf,Integer process_id) =0;
  virtual void broadcast(Int64ArrayView send_buf,Integer process_id) =0;
  virtual void broadcast(RealArrayView send_buf,Integer process_id) =0;
  //@}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
