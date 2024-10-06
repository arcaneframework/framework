// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisWrapper.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Wrapper autour des appels de Parmetis.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/MetisWrapper.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/std/internal/MetisGraph.h"
#include "arcane/std/internal/MetisGraphDigest.h"
#include "arcane/std/internal/MetisGraphGather.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MetisWrapper::
MetisWrapper(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle Metis en regroupant le graph sur 2 processeurs.
 */
int MetisWrapper::
_callMetisWith2Processors(const Int32 ncon, const bool need_part,
                          ConstArrayView<idx_t> vtxdist, MetisGraphView my_graph,
                          MetisCall& metis_call)
{
  Int32 nb_rank = m_parallel_mng->commSize();
  Int32 my_rank = m_parallel_mng->commRank();

  UniqueArray<idx_t> half_vtxdist(vtxdist.size());
  
  int comm_0_size = nb_rank / 2 + nb_rank % 2;
  int comm_1_size = nb_rank / 2;
  int comm_0_io_rank = 0;
  int comm_1_io_rank = comm_0_size;

  for (int i = 0; i < nb_rank + 1; ++i) {
    half_vtxdist[i] = vtxdist[i];
  }
  
  int color = 1;
  int key = my_rank;
  
  if (my_rank >= comm_0_size) {
    color = 2;
    for (int i = 0; i < comm_1_size + 1; ++i) {
      half_vtxdist[i] = vtxdist[i + comm_0_size] - vtxdist[comm_0_size];
    }
  }
  
  MetisGraph metis_graph;
  Ref<IParallelMng> half_pm = ParallelMngUtils::createSubParallelMngRef(m_parallel_mng, color, key);
  MetisGraphGather metis_gather(half_pm.get());

  metis_gather.gatherGraph(need_part, half_vtxdist, ncon, my_graph, metis_graph);

  color = -1;
  if (my_rank == comm_0_io_rank || my_rank == comm_1_io_rank) {
    color = 1;
  }

  Ref<IParallelMng> metis_pm = ParallelMngUtils::createSubParallelMngRef(m_parallel_mng, color, key);

  UniqueArray<idx_t> metis_vtxdist(3);
  metis_vtxdist[0] = 0;
  metis_vtxdist[1] = vtxdist[comm_0_size];
  metis_vtxdist[2] = vtxdist[vtxdist.size() - 1];
  
  int ierr = 0;

  if (metis_pm.get()) {
    MetisGraphView metis_graph_view(metis_graph);
    ierr = metis_call(metis_pm.get(), metis_graph_view, metis_vtxdist);
  }

  // S'assure que tout le monde a la même valeur de l'erreur.
  half_pm->broadcast(ArrayView<int>(1, &ierr), 0);

  metis_gather.scatterPart(half_vtxdist, metis_graph.part, my_graph.part);

  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MetisWrapper::
callPartKway(const bool print_digest, const bool gather,
             idx_t *vtxdist, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, 
             idx_t *adjwgt, idx_t *wgtflag, idx_t *numflag, idx_t *ncon, idx_t *nparts, 
             real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut, idx_t *part)
{
  int ierr = 0;
  Int32 nb_rank = m_parallel_mng->commSize();
  Int32 my_rank = m_parallel_mng->commRank();

  MetisCall partkway = [&](IParallelMng* pm, MetisGraphView graph,
                           ArrayView<idx_t> graph_vtxdist)
  {
    MPI_Comm graph_comm = static_cast<MPI_Comm>(pm->communicator());
    // NOTE GG: il peut arriver que ces deux pointeurs soient nuls s'il n'y a pas
    // de voisins. Si tout le reste est cohérent cela ne pose pas de problèmes mais ParMetis
    // n'aime pas les tableaux vides donc si c'est le cas on met un 0.
    // TODO: il faudrait regarder en amont s'il ne vaudrait pas mieux mettre des valeurs
    // dans ces deux tableaux au cas où.
    idx_t null_idx = 0;
    idx_t* adjncy_data = graph.adjncy.data();
    idx_t* adjwgt_data = graph.adjwgt.data();
    if (!adjncy_data)
      adjncy_data = &null_idx;
    if (!adjwgt_data)
      adjwgt_data = &null_idx;
    return ParMETIS_V3_PartKway(graph_vtxdist.data(), graph.xadj.data(),
                                adjncy_data, graph.vwgt.data(),
                                adjwgt_data, wgtflag, numflag, ncon, nparts, tpwgts,
                                ubvec, options, edgecut, graph.part.data(), &graph_comm);
  };

  // Version séquentielle utilisant directement Metis
  // NOTE: Ce bout de code est le même que dans callAdaptativeRepart.
  // Il faudrait mutualiser les deux.
  MetisCall partkway_seq = [&](IParallelMng*, MetisGraphView graph,
                               ArrayView<idx_t> graph_vtxdist)
  {
    idx_t options2[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options2);
    options2[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options2[METIS_OPTION_UFACTOR] = 30; // TODO: a récupérer depuis les options de parmetis
    options2[METIS_OPTION_NUMBERING] = 0; // 0 pour indiquer que les tableaux sont en C (indices commencent à 0)
    options2[METIS_OPTION_MINCONN] = 0;
    options2[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options2[METIS_OPTION_SEED] = 25; // TODO: pouvoir changer la graine
    pwarning() << "MetisWrapper: using user 'imbalance_factor' is not yet implemented. Using defaut value 30";

    // Le nombre de sommets du graph est dans le premier indice de graph_vtxdist
    idx_t nvtxs = graph_vtxdist[1];
    return METIS_PartGraphKway(&nvtxs /*graph_vtxdist.data()*/, ncon, graph.xadj.data(),
                               graph.adjncy.data(), graph.vwgt.data(), graph.vsize.data(),
                               graph.adjwgt.data(), nparts, tpwgts,
                               ubvec, options2, edgecut, graph.part.data());
  };

  MetisGraphView my_graph;
  
  ArrayView<idx_t> offset(nb_rank + 1, vtxdist);
  my_graph.nb_vertices = CheckedConvert::toInt32(offset[my_rank + 1] - offset[my_rank]);
  my_graph.xadj = ArrayView<idx_t>(my_graph.nb_vertices + 1, xadj);
  const Int32 adjacency_size = CheckedConvert::toInt32(my_graph.xadj[my_graph.nb_vertices]);
  const Int32 nb_con = CheckedConvert::toInt32(*ncon);
  my_graph.adjncy = ArrayView<idx_t>(adjacency_size, adjncy);
  my_graph.vwgt = ArrayView<idx_t>(CheckedConvert::multiply(my_graph.nb_vertices, nb_con), vwgt);
  my_graph.adjwgt = ArrayView<idx_t>(adjacency_size, adjwgt);
  my_graph.part = ArrayView<idx_t>(my_graph.nb_vertices, part);
  my_graph.have_vsize = false;
  my_graph.have_adjwgt = true;
  
  if (print_digest){
    MetisGraphDigest d(m_parallel_mng);
    String digest = d.computeInputDigest(false, 3, my_graph, vtxdist, wgtflag, numflag,
                                         ncon, nparts, tpwgts, ubvec, nullptr, options);
    if (my_rank == 0) {
      info() << "signature des entrees Metis = " << digest;
    }
  }
  
  if (gather && nb_rank > 2) {
    // Normalement c'est plus rapide ...
    info() << "Partitioning metis : re-grouping " << nb_rank << " -> 2 rank";
    ierr = _callMetisWith2Processors(nb_con, false, offset, my_graph, partkway);
  }
  else {
    info() << "Partitioning metis : nb rank = " << nb_rank;
    MetisCall& metis_call = (nb_rank == 1) ? partkway_seq : partkway;
    ierr = metis_call(m_parallel_mng, my_graph, offset);
  }

  info() << "End Partitioning metis";
  if (print_digest){
    MetisGraphDigest d(m_parallel_mng);
    String digest = d.computeOutputDigest(my_graph, edgecut);
    if (my_rank == 0) {
      info() << "hash for Metis output = " << digest;
    }
  }
  
  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MetisWrapper::
callAdaptiveRepart(const bool print_digest, const bool gather,
                   idx_t *vtxdist, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, 
                   idx_t *vsize, idx_t *adjwgt, idx_t *wgtflag, idx_t *numflag, idx_t *ncon, 
                   idx_t *nparts, real_t *tpwgts, real_t *ubvec, real_t *ipc2redist, 
                   idx_t *options, idx_t *edgecut, idx_t *part)
{
  int ierr = 0;
  Int32 nb_rank = m_parallel_mng->commSize();
  Int32 my_rank = m_parallel_mng->commRank();

  MetisCall repart_func = [&](IParallelMng* pm, MetisGraphView graph,
                              ArrayView<idx_t> graph_vtxdist)
  {
    MPI_Comm graph_comm = static_cast<MPI_Comm>(pm->communicator());
    return ParMETIS_V3_AdaptiveRepart(graph_vtxdist.data(), graph.xadj.data(),
                                      graph.adjncy.data(), graph.vwgt.data(), 
                                      graph.vsize.data(), graph.adjwgt.data(),
                                      wgtflag, numflag, ncon, nparts, tpwgts, ubvec,
                                      ipc2redist, options, edgecut,
                                      graph.part.data(), &graph_comm);
  };

  // Version séquentielle utilisant directement Metis
  // NOTE: Ce bout de code est le même que dans callPartKWay
  // Il faudrait mutualiser les deux.
  MetisCall repart_seq_func = [&](IParallelMng*, MetisGraphView graph,
                                  ArrayView<idx_t> graph_vtxdist)
  {
    idx_t options2[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options2);
    options2[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options2[METIS_OPTION_UFACTOR] = 30; // TODO: a récupérer depuis les options de parmetis
    options2[METIS_OPTION_NUMBERING] = 0; // 0 pour indiquer que les tableaux sont en C (indices commencent à 0)
    options2[METIS_OPTION_MINCONN] = 0;
    options2[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options2[METIS_OPTION_SEED] = 25; // TODO: pouvoir changer la graine
    pwarning() << "MetisWrapper: using user 'imbalance_factor' is not yet implemented. Using defaut value 30";
    // Le nombre de sommets du graph est dans le premier indice de graph_vtxdist
    idx_t nvtxs = graph_vtxdist[1];
    return METIS_PartGraphKway(&nvtxs /*graph_vtxdist.data()*/, ncon, graph.xadj.data(),
                               graph.adjncy.data(), graph.vwgt.data(), graph.vsize.data(),
                               graph.adjwgt.data(), nparts, tpwgts,
                               ubvec, options2, edgecut, graph.part.data());
  };

  MetisGraphView my_graph;
  

  ArrayView<idx_t> offset(nb_rank + 1, vtxdist);
  my_graph.nb_vertices = CheckedConvert::toInt32(offset[my_rank + 1] - offset[my_rank]);
  my_graph.xadj = ArrayView<idx_t>(my_graph.nb_vertices + 1, xadj);
  const Int32 adjacency_size = CheckedConvert::toInt32(my_graph.xadj[my_graph.nb_vertices]);
  const Int32 nb_con = CheckedConvert::toInt32(*ncon);
  my_graph.adjncy = ArrayView<idx_t>(adjacency_size, adjncy);
  my_graph.vwgt = ArrayView<idx_t>(CheckedConvert::multiply(my_graph.nb_vertices, nb_con), vwgt);
  my_graph.vsize = ArrayView<idx_t>(my_graph.nb_vertices, vsize);
  my_graph.adjwgt = ArrayView<idx_t>(adjacency_size, adjwgt);
  my_graph.part = ArrayView<idx_t>(my_graph.nb_vertices, part);
  my_graph.have_vsize = true;
  my_graph.have_adjwgt = true;

  
  if (print_digest){
    MetisGraphDigest d(m_parallel_mng);
    String digest = d.computeInputDigest(true, 4, my_graph, vtxdist, wgtflag, numflag,
                                         ncon, nparts, tpwgts, ubvec, nullptr, options);
    if (my_rank == 0) {
      info() << "signature des entrees Metis = " << digest;
    }
  }

  if (gather && nb_rank > 2) {
    info() << "Partionnement metis : regroupement " << nb_rank << " -> 2 processeurs";
    ierr = _callMetisWith2Processors(nb_con, true, offset, my_graph, repart_func);
  }
  else {
    info() << "Partionnement metis : nb processeurs = " << nb_rank;
    MetisCall& metis_call = (nb_rank == 1) ? repart_seq_func : repart_func;
    ierr = metis_call(m_parallel_mng, my_graph, offset);
  }

  if (print_digest) {
    MetisGraphDigest d(m_parallel_mng);
    String digest = d.computeOutputDigest(my_graph, edgecut);
    if (my_rank == 0) {
      info() << "signature des sorties Metis = " << digest;
    }
  }
  
  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
