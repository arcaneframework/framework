// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraph.h                                                (C) 2000-2024 */
/*                                                                           */
/* Gestion d'un graphe de Metis.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_METISGRAPH
#define ARCANE_STD_METISGRAPH
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ArrayView.h"
#include <parmetis.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct MetisGraph
{
  int nb_vertices = 0;
  bool have_vsize = false;
  bool have_adjwgt = false;
  UniqueArray<idx_t> xadj;
  UniqueArray<idx_t> adjncy;
  UniqueArray<idx_t> vwgt;
  UniqueArray<idx_t> vsize;
  UniqueArray<idx_t> adjwgt;
  UniqueArray<idx_t> part;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MetisGraphView
{
 public:
  MetisGraphView()  = default;
  
  MetisGraphView(MetisGraph& graph)
  : nb_vertices(graph.nb_vertices)
  , have_vsize(graph.have_vsize)
  , have_adjwgt(graph.have_adjwgt)
  , xadj(graph.xadj)
  , adjncy(graph.adjncy)
  , vwgt(graph.vwgt)
  , vsize(graph.vsize)
  , adjwgt(graph.adjwgt)
  , part(graph.part)
  {}
 public:
  int nb_vertices = 0;
  bool have_vsize = false;
  bool have_adjwgt = false;
  ArrayView<idx_t> xadj;
  ArrayView<idx_t> adjncy;
  ArrayView<idx_t> vwgt;
  ArrayView<idx_t> vsize;
  ArrayView<idx_t> adjwgt;
  ArrayView<idx_t> part;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
