// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphGather.h                                          (C) 2000-2024 */
/*                                                                           */
/* Gathering of 'Parmetis' graphs.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_METISGRAPHGATHER
#define ARCANE_STD_INTERNAL_METISGRAPHGATHER
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"

#include "arcane/std/internal/MetisGraph.h"

#include <parmetis.h>
#include <mpi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MetisGraphGather
: public TraceAccessor
{
 public:

  explicit MetisGraphGather(IParallelMng* pm);

 public:

  /*!
   * \brief Performs a gathering of the ParMetis graph "my_graph" on processor
   * rank 0 in the communicator "comm". The resulting graph is "graph".
   */
  void gatherGraph(const bool need_part, ConstArrayView<idx_t> vtxdist, const int ncon, MetisGraphView my_graph,
                   MetisGraph& graph);

  /*!
   * \brief Distributes the partitioning "part" from processor rank 0 in the
   * communicator "comm" to all processors in this communicator. The result
   * is "my_part", which must already be sized before calling.
   */
  void scatterPart(ConstArrayView<idx_t> vtxdist, ConstArrayView<idx_t> part, ArrayView<idx_t> my_part);

 private:

  IParallelMng* m_parallel_mng = nullptr;
  Int32 m_my_rank = A_NULL_RANK;
  Int32 m_nb_rank = A_NULL_RANK;

 private:

  template <class SourceType, class TargetType>
  void _convertVector(const int size, ConstArrayView<SourceType> src, ArrayView<TargetType> dest);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
