// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephKernel.h                                               (C) 2000-2022 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_KERNEL_H
#define ARCANE_ALEPH_KERNEL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"

#include "arcane/aleph/AlephGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephKernelResults
{
 public:
  Integer m_nb_iteration;
  Real m_residual_norm[4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ALEPH_EXPORT AlephKernelArguments
: public TraceAccessor
{
 public:
  AlephKernelArguments(ITraceMng* tm,
                       AlephVector* x_vector,
                       AlephVector* b_vector,
                       AlephVector* tmp_vector,
                       IAlephTopology* topology)
  : TraceAccessor(tm)
  , m_x_vector(x_vector)
  , m_b_vector(b_vector)
  , m_tmp_vector(tmp_vector)
  , m_topology_implementation(topology)
  , m_params(NULL)
  {} // m_params sera initialisé via le postSolver
  ~AlephKernelArguments()
  {
    debug() << "\33[1;5;31m[~AlephKernelArguments]"
            << "\33[0m";
  };

 public:
  AlephVector* m_x_vector;
  AlephVector* m_b_vector;
  AlephVector* m_tmp_vector;
  IAlephTopology* m_topology_implementation;
  AlephParams* m_params;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ALEPH_EXPORT AlephKernel
: public TraceAccessor
{
 public:
  AlephKernel(IParallelMng*, Integer, IAlephFactory*, Integer = 0, Integer = 0, bool = false);
  AlephKernel(ITraceMng*, ISubDomain*, IAlephFactory*, Integer = 0, Integer = 0, bool = false);
  AlephKernel(ISubDomain*, Integer alephUnderlyingSolver = 0, Integer alephNumberOfCores = 0);
  ~AlephKernel(void);
  void setup(void);
  void initialize(Integer, Integer);
  void break_and_return(void);
  AlephVector* createSolverVector(void);
  AlephMatrix* createSolverMatrix(void);
  void postSolver(AlephParams*, AlephMatrix*, AlephVector*, AlephVector*);
  void workSolver(void);
  AlephVector* syncSolver(Integer, Integer&, Real*);

 public:
  IAlephFactory* factory(void) { return m_factory; }
  AlephTopology* topology(void) { return m_topology; }
  AlephOrdering* ordering(void) { return m_ordering; }
  AlephIndexing* indexing(void) { return m_indexing; }
  Integer rank(void) { return m_rank; }
  Integer size(void) { return m_size; }
  ISubDomain* subDomain(void)
  {
    if (!m_sub_domain && !m_i_am_an_other)
      throw FatalErrorException("[AlephKernel::subDomain]", "No sub-domain to work on!");
    return m_sub_domain;
  }
  bool isParallel(void) { return m_isParallel; }
  bool isInitialized(void) { return m_has_been_initialized; }
  bool thereIsOthers(void) { return m_there_are_idles; }
  bool isAnOther(void) { return m_i_am_an_other; }
  IParallelMng* parallel(void) { return m_parallel; }
  IParallelMng* world(void) { return m_world_parallel; }
  Integer underlyingSolver(void) { return m_underlying_solver; }
  bool isCellOrdering(void) { return m_reorder; }
  Integer index(void) { return m_solver_index; }
  bool configured(void) { return m_configured; }
  void mapranks(Array<Integer>&);
  bool hitranks(Integer, ArrayView<Integer>);
  Integer nbRanksPerSolver(void) { return m_solver_size; }
  ArrayView<Integer> solverRanks(Integer i) { return m_solver_ranks.at(i).view(); }
  IParallelMng* subParallelMng(Integer i) { return m_sub_parallel_mng_queue.at(i).get(); }
  IAlephTopology* getTopologyImplementation(Integer i)
  {
    return m_arguments_queue.at(i)->m_topology_implementation;
  }

 private:

  Ref<IParallelMng> _createUnderlyingParallelMng(Integer);

 private:

  ISubDomain* m_sub_domain;
  bool m_isParallel;
  Integer m_rank;
  Integer m_size;
  Integer m_world_size;
  bool m_there_are_idles;
  bool m_i_am_an_other;
  IParallelMng* m_parallel;
  IParallelMng* m_world_parallel;

 private:

  bool m_configured = false;
  IAlephFactory* m_factory = nullptr;
  AlephTopology* m_topology = nullptr;
  AlephOrdering* m_ordering = nullptr;
  AlephIndexing* m_indexing = nullptr;
  Integer m_aleph_vector_idx;
  const Integer m_underlying_solver;
  const bool m_reorder;
  Integer m_solver_index;
  Integer m_solver_size;
  bool m_solved;
  bool m_has_been_initialized;

 private:

  UniqueArray<SharedArray<Integer>> m_solver_ranks;
  UniqueArray<Ref<IParallelMng>> m_sub_parallel_mng_queue;
  UniqueArray<AlephMatrix*> m_matrix_queue;
  UniqueArray<AlephKernelArguments*> m_arguments_queue;
  UniqueArray<AlephKernelResults*> m_results_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
