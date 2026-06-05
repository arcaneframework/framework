// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelTopology.h                                          (C) 2000-2011 */
/*                                                                           */
/* Information on the topology for allocating computing cores.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_PARALLELTOPOLOGY_H
#define ARCANE_IMPL_PARALLELTOPOLOGY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IParallelTopology.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information on the topology for allocating computing cores.
 *
 * Before use, initialize() must be called, which is a collective operation.
 */
class ARCANE_IMPL_EXPORT ParallelTopology
: public IParallelTopology
{
 public:

  ParallelTopology(IParallelMng* pm);
  virtual ~ParallelTopology() {} //!< Frees resources.

 public:

  //! Initializes the instance. This operation is collective
  virtual void initialize();

 public:

  virtual IParallelMng* parallelMng() const;
  virtual bool isMasterMachine() const;
  virtual Int32ConstArrayView machineRanks() const;
  virtual Int32 machineRank() const;
  virtual Int32ConstArrayView masterMachineRanks() const;
  virtual bool isMasterProcess() const;
  virtual Int32ConstArrayView processRanks() const;
  virtual Int32 processRank() const;
  virtual Int32ConstArrayView masterProcessRanks() const;

 private:

  IParallelMng* m_parallel_mng;
  Int32UniqueArray m_machine_ranks;
  Int32UniqueArray m_process_ranks;
  Int32 m_machine_rank;
  Int32 m_process_rank;
  bool m_is_machine_master;
  bool m_is_process_master;
  Int32UniqueArray m_master_machine_ranks;
  Int32UniqueArray m_master_process_ranks;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
