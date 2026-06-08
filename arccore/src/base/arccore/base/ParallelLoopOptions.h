// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelLoopOptions.h                                       (C) 2000-2025 */
/*                                                                           */
/* Configuration options for parallel loops in multi-threading.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_PARALLELLOOPOPTIONS_H
#define ARCCORE_BASE_PARALLELLOOPOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Concurrency
 * \brief Execution options for a parallel loop in multi-threading.
 *
 * This class allows specifying execution parameters for a parallel loop.
 */
class ARCCORE_BASE_EXPORT ParallelLoopOptions
{
 private:

  //! Flag to indicate which fields have been set.
  enum SetFlags
  {
    SF_MaxThread = 1,
    SF_GrainSize = 2,
    SF_Partitioner = 4
  };

 public:

  //! Partitioner type
  enum class Partitioner
  {
    //! Leaves the partitioner to manage partitioning and scheduling (default)
    Auto = 0,
    /*!
     * \brief Uses static partitioning.
     *
     * In this mode, grainSize() is not used, and partitioning depends only on the number of threads and the iteration interval.
     *
     * Note that the scheduling remains dynamic, so it is not necessarily the same thread that will execute
     * the same iteration block.
     */
    Static = 1,
    /*!
     * \brief Uses static partitioning and scheduling.
     *
     * This mode is similar to Partitioner::Static, but the scheduling
     * is deterministic for task assignment: the value
     * returned by TaskFactory::currentTaskIndex() is deterministic.
     *
     * \note Currently, this partitioning mode is only available
     * for 1D loop parallelization.
     */
    Deterministic = 2
  };

 public:

  ParallelLoopOptions()
  : m_grain_size(0)
  , m_max_thread(-1)
  , m_partitioner(Partitioner::Auto)
  , m_flags(0)
  {}

 public:

  //! Maximum number of allowed threads.
  Int32 maxThread() const { return m_max_thread; }
  /*!
   * \brief Sets the maximum number of allowed threads.
   *
   * If \a v is 0 or 1, the execution will be sequential.
   * If \a v is greater than ConcurrencyBase::maxAllowedThread(), the latter value will be used.
   */
  void setMaxThread(Integer v)
  {
    m_max_thread = v;
    m_flags |= SF_MaxThread;
  }
  //! Indicates if maxThread() is set
  bool hasMaxThread() const { return m_flags & SF_MaxThread; }

  //! Size of an iteration interval.
  Integer grainSize() const { return m_grain_size; }
  //! Sets the size (approximate) of an iteration interval
  void setGrainSize(Integer v)
  {
    m_grain_size = v;
    m_flags |= SF_GrainSize;
  }
  //! Indicates if grainSize() is set
  bool hasGrainSize() const { return m_flags & SF_GrainSize; }

  //! Partitioner type
  Partitioner partitioner() const { return m_partitioner; }
  //! Sets the partitioner type
  void setPartitioner(Partitioner v)
  {
    m_partitioner = v;
    m_flags |= SF_Partitioner;
  }
  //! Indicates if grainSize() is set
  bool hasPartitioner() const { return m_flags & SF_Partitioner; }

 public:

  //! Merges the unmodified values of the instance with those of \a po.
  void mergeUnsetValues(const ParallelLoopOptions& po)
  {
    if (!hasMaxThread())
      setMaxThread(po.maxThread());
    if (!hasGrainSize())
      setGrainSize(po.grainSize());
    if (!hasPartitioner())
      setPartitioner(po.partitioner());
  }

 private:

  //! Size of a loop block
  Int32 m_grain_size = 0;
  //!< Maximum number of threads for the loop
  Int32 m_max_thread = -1;
  //!< Partitioner type.
  Partitioner m_partitioner = Partitioner::Auto;

  unsigned int m_flags = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
