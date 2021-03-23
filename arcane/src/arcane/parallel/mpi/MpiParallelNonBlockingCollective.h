// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelNonBlockingCollective.h                          (C) 2000-2018 */
/*                                                                           */
/* Implémentation des collectives non bloquantes avec MPI.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIPARALLELNONBLOCKINGCOLLECTIVE_H
#define ARCANE_PARALLEL_MPI_MPIPARALLELNONBLOCKINGCOLLECTIVE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/ParallelNonBlockingCollectiveDispatcher.h"

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire du parallélisme utilisant MPI.
 */
class ARCANE_MPI_EXPORT MpiParallelNonBlockingCollective
: public ParallelNonBlockingCollectiveDispatcher
{
 public:

  MpiParallelNonBlockingCollective(ITraceMng* tm,IParallelMng* pm,MpiAdapter* adapter);
  virtual ~MpiParallelNonBlockingCollective();

 public:

  virtual void build();
  virtual Request barrier();
  virtual bool hasValidReduceForDerivedType() const;

 private:

  ITraceMng* m_trace_mng;
  IParallelMng* m_parallel_mng;
  MpiAdapter* m_adapter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

