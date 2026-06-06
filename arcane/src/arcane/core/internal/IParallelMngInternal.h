// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelMngInternal.h                                      (C) 2000-2026 */
/*                                                                           */
/* Internal part of IParallelMng in Arcane.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IPARALLELMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_IPARALLELMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MessagePassing
{
  class IContigMachineShMemWinBaseInternal;
  class IMachineShMemWinBaseInternal;
} // namespace MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal part of IParallelMng.
 */
class ARCANE_CORE_EXPORT IParallelMngInternal
{
 public:

  virtual ~IParallelMngInternal() = default;

 public:

  //! Default runner. Can be null.
  virtual Runner runner() const = 0;

  //! Default queue for messages. Can be null.
  virtual RunQueue queue() const = 0;

  /*!
   * \brief Indicates if the implementation handles accelerators.
   *
   * If so, the accelerator memory can be used directly
   * in MPI calls, which avoids potential re-copies.
   */
  virtual bool isAcceleratorAware() const = 0;

  //! Creates a sub IParallelMng similar to MPI_Comm_split.
  virtual Ref<IParallelMng> createSubParallelMngRef(Int32 color, Int32 key) = 0;

  virtual void setDefaultRunner(const Runner& runner) = 0;

  //! Gives the writer in the case where parallel writing is possible (with
  //! MPI-IO for example).
  virtual Int32 masterParallelIORank() const = 0;

  //! Gives the number of procs that will send data to
  //! masterParallelIORank().
  virtual Int32 nbSendersToMasterParallelIO() const = 0;

  /*!
   * \brief Method allowing the initialization of the windowCreator specific to
   * the implementation.
   *
   * Collective call.
   */
  virtual void initializeWindowCreator() = 0;

  /*!
   * \brief Method allowing to know if shared memory mode is supported.
   *
   * Collective call.
   */
  virtual bool isMachineShMemWinAvailable() = 0;

  /*!
   * \brief Method allowing the creation of a memory window on the node.
   *
   * Collective call.
   *
   * \param sizeof_segment The size of our segment (in bytes).
   * \param sizeof_type The size of a segment element (in bytes).
   * \return A reference to the new window.
   */
  virtual Ref<MessagePassing::IContigMachineShMemWinBaseInternal> createContigMachineShMemWinBase(Int64 sizeof_segment, Int32 sizeof_type) = 0;

  /*!
   * \brief Method allowing the creation of a dynamic memory window on the node.
   *
   * Collective call.
   *
   * \param sizeof_segment The initial size of our segment (in bytes).
   * \param sizeof_type The size of a segment element (in bytes).
   * \return A reference to the new window.
   */
  virtual Ref<MessagePassing::IMachineShMemWinBaseInternal> createMachineShMemWinBase(Int64 sizeof_segment, Int32 sizeof_type) = 0;

  /*!
   * \brief Method allowing retrieval of a shared memory allocator.
   */
  virtual MemoryAllocationOptions machineShMemWinMemoryAllocator() = 0;

  /*!
   * \brief Method allowing retrieval of the ranks of the sub-domains of the
   * computing node.
   *
   * Non-collective call.
   */
  virtual ConstArrayView<Int32> machineRanks() = 0;

  /*!
   * \brief Method allowing a barrier for the sub-domains of the
   * computing node.
   */
  virtual void machineBarrier() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
