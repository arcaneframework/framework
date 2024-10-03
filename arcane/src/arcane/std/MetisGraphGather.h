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
#ifndef ARCANE_STD_METISGRAPHGATHER
#define ARCANE_STD_METISGRAPHGATHER
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/std/MetisGraph.h"

#include <parmetis.h>
#include <mpi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MetisGraphGather
{
 public:

  /*!
   * \brief Effectue un regroupement du graphe ParMetis "my_graph" sur le processeur de
   * rang 0 dans le communicateur "comm". Le graph résultat est "graph".
   */
  void gatherGraph(const bool need_part, const String& comm_name, MPI_Comm comm,
                   ConstArrayView<idx_t> vtxdist, const int ncon, MetisGraphView my_graph,
                   MetisGraph& graph);

  /*!
   * \brief Distribue le partitionnement "part" depuis le processeur de rang 0 dans le
   * communicateur "comm" sur tous les processeurs de ce communicateur. Le resultat
   * est "my_part", qui doit deja etre dimensionne avant appel.
   */
  void scatterPart(MPI_Comm comm, ConstArrayView<idx_t> vtxdist, ConstArrayView<idx_t> part,
                   ArrayView<idx_t> my_part);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
