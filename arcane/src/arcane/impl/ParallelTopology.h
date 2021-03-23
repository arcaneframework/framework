// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelTopology.h                                          (C) 2000-2011 */
/*                                                                           */
/* Informations sur la topologie d'allocation des coeurs de calcul.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_PARALLELTOPOLOGY_H
#define ARCANE_IMPL_PARALLELTOPOLOGY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IParallelTopology.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la topologie d'allocation des coeurs de calcul.
 *
 * Avant utilisation, il faut appeler initialize() qui est une opération
 * collective.
 */
class ARCANE_IMPL_EXPORT ParallelTopology
: public IParallelTopology
{
 public:

  ParallelTopology(IParallelMng* pm);
  virtual ~ParallelTopology() {} //!< Libère les ressources.

 public:

  //! Initialise l'instance. Cette opération est collective
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

