// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphGather.h                                          (C) 2000-2024 */
/*                                                                           */
/* Regroupement de graphes de 'Parmetis'.                                    */
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
   * \brief Effectue un regroupement du graphe ParMetis "my_graph" sur le processeur de
   * rang 0 dans le communicateur "comm". Le graph résultat est "graph".
   */
  void gatherGraph(const bool need_part, ConstArrayView<idx_t> vtxdist, const int ncon, MetisGraphView my_graph,
                   MetisGraph& graph);

  /*!
   * \brief Distribue le partitionnement "part" depuis le processeur de rang 0 dans le
   * communicateur "comm" sur tous les processeurs de ce communicateur. Le resultat
   * est "my_part", qui doit deja etre dimensionne avant appel.
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
