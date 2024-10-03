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
/* Calcule une somme de contrôle globale des entrées/sorties Metis.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IParallelMng.h"

#include "arcane/std/MetisGraph.h"
#include "arcane/std/MetisGraphDigest.h"
#include "arcane/std/MetisGraphGather.h"
#include "arcane/std/MetisWrapper.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using MetisCall = std::function<int(MPI_Comm& comm, MetisGraphView graph,
                                    ArrayView<idx_t> vtxdist)>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle Metis sans regroupement de graph.
 */
int
_callMetis(MPI_Comm comm, ArrayView<idx_t> vtxdist, MetisGraphView my_graph,
           MetisCall& metis)
{
  return metis(comm, my_graph, vtxdist);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle Metis en regroupant le graph sur 2 processeurs.
 */
int
_callMetisWith2Processors(const idx_t ncon, const bool need_part, MPI_Comm comm,
                          ConstArrayView<idx_t> vtxdist, MetisGraphView my_graph,
                          MetisCall& metis)
{
  int my_rank = -1;
  int nb_rank = -1;
  
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nb_rank);
  
  String half_comm_name = "first";
  UniqueArray<idx_t> half_vtxdist(vtxdist.size());
  
  int comm_0_size = nb_rank / 2 + nb_rank % 2;
  int comm_1_size = nb_rank / 2;
  int comm_0_io_rank = 0;
  int comm_1_io_rank = comm_0_size;
  
  MPI_Comm half_comm;
  
  for (int i = 0; i < nb_rank + 1; ++i) {
    half_vtxdist[i] = vtxdist[i];
  }
  
  int color = 1;
  int key = my_rank;
  
  if (my_rank >= comm_0_size) {
    color = 2;
    half_comm_name = "second";
    for (int i = 0; i < comm_1_size + 1; ++i) {
      half_vtxdist[i] = vtxdist[i + comm_0_size] - vtxdist[comm_0_size];
    }
  }
  
  MetisGraph metis_graph;
  MetisGraphGather metis_gather;

  MPI_Comm_split(comm, color, key, &half_comm);

  metis_gather.gatherGraph(need_part, half_comm_name, half_comm, half_vtxdist,
                           ncon, my_graph, metis_graph);
  
  color = MPI_UNDEFINED;
  if (my_rank == comm_0_io_rank || my_rank == comm_1_io_rank) {
    color = 1;
  }
  
  MPI_Comm metis_comm;
  MPI_Comm_split(comm, color, key, &metis_comm);
  
  UniqueArray<idx_t> metis_vtxdist(3);
  metis_vtxdist[0] = 0;
  metis_vtxdist[1] = vtxdist[comm_0_size];
  metis_vtxdist[2] = vtxdist[vtxdist.size() - 1];
  
  int ierr = 0;
  
  if (metis_comm != MPI_COMM_NULL) {
    MetisGraphView metis_graph_view(metis_graph);
    ierr = metis(metis_comm, metis_graph_view, metis_vtxdist);
    MPI_Comm_free(&metis_comm);
  }
  
  MPI_Bcast(&ierr, 1, MPI_INT, 0, half_comm);
  
  metis_gather.scatterPart(half_comm, half_vtxdist, metis_graph.part, my_graph.part);

  MPI_Comm_free(&half_comm); 
  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle Metis en regroupant le graph sur 1 seul processeur.
 *
 * \warning Cette méthode n'est pas compatible avec la routine AdaptiveRepart de ParMetis qui
 * est buggee lorsqu'il n'y a qu'un seul processeur.
 */
int
_callMetisWith1Processor(const idx_t ncon, const bool need_part, MPI_Comm comm,
                         ConstArrayView<idx_t> vtxdist, MetisGraphView my_graph,
                         MetisCall& metis)
{
  int my_rank = -1;
  int nb_rank = -1;
  
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nb_rank);
  
  MetisGraph metis_graph;
  MetisGraphGather metis_gather;

  metis_gather.gatherGraph(need_part, "maincomm", comm, vtxdist, ncon,
                           my_graph, metis_graph);
  
  MPI_Comm metis_comm = MPI_COMM_SELF;
  
  UniqueArray<idx_t> metis_vtxdist(2);
  metis_vtxdist[0] = 0;
  metis_vtxdist[1] = vtxdist[vtxdist.size() - 1];
  
  int ierr = 0;
  
  if (my_rank == 0) {
    MetisGraphView metis_graph_view(metis_graph);
    ierr = metis(metis_comm, metis_graph_view, metis_vtxdist);
  }
  
  MPI_Bcast(&ierr, 1, MPI_INT, 0, comm);
  
  metis_gather.scatterPart(comm, vtxdist, metis_graph.part, my_graph.part);

  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MetisWrapper::
callPartKway(IParallelMng* pm, const bool print_digest, const bool gather,
             idx_t *vtxdist, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, 
             idx_t *adjwgt, idx_t *wgtflag, idx_t *numflag, idx_t *ncon, idx_t *nparts, 
             real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut, idx_t *part, 
             MPI_Comm *comm)
{
  ITraceMng* tm = pm->traceMng();
  int ierr = 0;
  int nb_rank = -1;
  int my_rank = -1;
  
  MPI_Comm_size(*comm, &nb_rank);
  MPI_Comm_rank(*comm, &my_rank);
  
  MetisCall partkway = [&](MPI_Comm& graph_comm, MetisGraphView graph,
                           ArrayView<idx_t> graph_vtxdist)
  {
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
  MetisCall partkway_seq = [&](MPI_Comm& graph_comm, MetisGraphView graph,
                               ArrayView<idx_t> graph_vtxdist)
  {
    ARCANE_UNUSED(graph_comm);
    idx_t options2[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options2);
    options2[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options2[METIS_OPTION_UFACTOR] = 30; // TODO: a récupérer depuis les options de parmetis
    options2[METIS_OPTION_NUMBERING] = 0; // 0 pour indiquer que les tableaux sont en C (indices commencent à 0)
    options2[METIS_OPTION_MINCONN] = 0;
    options2[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options2[METIS_OPTION_SEED] = 25; // TODO: pouvoir changer la graine
    tm->pwarning() << "MetisWrapper: using user 'imbalance_factor' is not yet implemented. Using defaut value 30";

    // Le nombre de sommets du graph est dans le premier indice de graph_vtxdist
    idx_t nvtxs = graph_vtxdist[1];
    return METIS_PartGraphKway(&nvtxs /*graph_vtxdist.data()*/, ncon, graph.xadj.data(),
                               graph.adjncy.data(), graph.vwgt.data(), graph.vsize.data(),
                               graph.adjwgt.data(), nparts, tpwgts,
                               ubvec, options2, edgecut, graph.part.data());
  };

  MetisGraphView my_graph;
  
  ArrayView<idx_t> offset(nb_rank + 1, vtxdist);
  my_graph.nb_vertices = offset[my_rank+1] - offset[my_rank];
  my_graph.xadj = ArrayView<idx_t>(my_graph.nb_vertices + 1, xadj);
  idx_t adjncy_size = my_graph.xadj[my_graph.nb_vertices];
  my_graph.adjncy = ArrayView<idx_t>(adjncy_size, adjncy);
  my_graph.vwgt = ArrayView<idx_t>(my_graph.nb_vertices * (*ncon), vwgt);
  my_graph.adjwgt = ArrayView<idx_t>(adjncy_size, adjwgt);
  my_graph.part = ArrayView<idx_t>(my_graph.nb_vertices, part);
  my_graph.have_vsize = false;
  my_graph.have_adjwgt = true;
  
  if (print_digest){
    MetisGraphDigest d(pm);
    String digest = d.computeInputDigest(false, 3, my_graph, vtxdist, wgtflag, numflag,
                                         ncon, nparts, tpwgts, ubvec, nullptr, options);
    if (my_rank == 0) {
      tm->info() << "signature des entrees Metis = " << digest;
    }
  }
  
  if (gather && nb_rank > 2) {
    //     tm->info() << "Partionnement metis avec regroupement sur 1 processeur";
    //     ierr = callMetisWith1Processor(*ncon, false, *comm, offset, my_graph, partkway);
    
    // Normalement c'est plus rapide ...
    tm->info() << "Partionnement metis : regroupement " << nb_rank << " -> 2 processeurs";
    ierr = _callMetisWith2Processors(*ncon, false, *comm, offset, my_graph, partkway);
  }
  else {
    tm->info() << "Partionnement metis : nb processeurs = " << nb_rank;
    ierr = _callMetis(*comm, offset, my_graph, (nb_rank==1) ? partkway_seq : partkway);
  }
  
  tm->info() << "End Partionnement metis";
  if (print_digest){
    MetisGraphDigest d(pm);
    String digest = d.computeOutputDigest(my_graph, edgecut);
    if (my_rank == 0) {
      tm->info() << "signature des sorties Metis = " << digest;
    }
  }
  
  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MetisWrapper::
callAdaptiveRepart(IParallelMng* pm, const bool print_digest, const bool gather,
                   idx_t *vtxdist, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, 
                   idx_t *vsize, idx_t *adjwgt, idx_t *wgtflag, idx_t *numflag, idx_t *ncon, 
                   idx_t *nparts, real_t *tpwgts, real_t *ubvec, real_t *ipc2redist, 
                   idx_t *options, idx_t *edgecut, idx_t *part, MPI_Comm *comm)
{
  ITraceMng* tm = pm->traceMng();
  int ierr = 0;
  int nb_rank = -1;
  int my_rank = -1;
  
  MPI_Comm_size(*comm, &nb_rank);
  MPI_Comm_rank(*comm, &my_rank);
  
  MetisCall repart_func = [&](MPI_Comm& graph_comm, MetisGraphView graph,
                              ArrayView<idx_t> graph_vtxdist)
  {
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
  MetisCall repart_seq_func = [&](MPI_Comm& graph_comm, MetisGraphView graph,
                                  ArrayView<idx_t> graph_vtxdist)
  {
    ARCANE_UNUSED(graph_comm);
    idx_t options2[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options2);
    options2[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options2[METIS_OPTION_UFACTOR] = 30; // TODO: a récupérer depuis les options de parmetis
    options2[METIS_OPTION_NUMBERING] = 0; // 0 pour indiquer que les tableaux sont en C (indices commencent à 0)
    options2[METIS_OPTION_MINCONN] = 0;
    options2[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options2[METIS_OPTION_SEED] = 25; // TODO: pouvoir changer la graine
    tm->pwarning() << "MetisWrapper: using user 'imbalance_factor' is not yet implemented. Using defaut value 30";
    // Le nombre de sommets du graph est dans le premier indice de graph_vtxdist
    idx_t nvtxs = graph_vtxdist[1];
    return METIS_PartGraphKway(&nvtxs /*graph_vtxdist.data()*/, ncon, graph.xadj.data(),
                               graph.adjncy.data(), graph.vwgt.data(), graph.vsize.data(),
                               graph.adjwgt.data(), nparts, tpwgts,
                               ubvec, options2, edgecut, graph.part.data());
  };

  MetisGraphView my_graph;
  

  ArrayView<idx_t> offset(nb_rank + 1, vtxdist);
  my_graph.nb_vertices = offset[my_rank+1] - offset[my_rank];
  my_graph.xadj = ArrayView<idx_t>(my_graph.nb_vertices + 1, xadj);
  idx_t adjncy_size = my_graph.xadj[my_graph.nb_vertices];
  my_graph.adjncy = ArrayView<idx_t>(adjncy_size, adjncy);
  my_graph.vwgt = ArrayView<idx_t>(my_graph.nb_vertices * (*ncon), vwgt);
  my_graph.vsize = ArrayView<idx_t>(my_graph.nb_vertices, vsize);
  my_graph.adjwgt = ArrayView<idx_t>(adjncy_size, adjwgt);
  my_graph.part = ArrayView<idx_t>(my_graph.nb_vertices, part);
  my_graph.have_vsize = true;
  my_graph.have_adjwgt = true;

  
  if (print_digest){
    MetisGraphDigest d(pm);
    String digest = d.computeInputDigest(true, 4, my_graph, vtxdist, wgtflag, numflag,
                                         ncon, nparts, tpwgts, ubvec, nullptr, options);
    if (my_rank == 0) {
      tm->info() << "signature des entrees Metis = " << digest;
    }
  }

  if (gather && nb_rank > 2) {
    tm->info() << "Partionnement metis : regroupement " << nb_rank << " -> 2 processeurs";
    ierr = _callMetisWith2Processors(*ncon, true, *comm, offset, my_graph, repart_func);
  }
  else {
    tm->info() << "Partionnement metis : nb processeurs = " << nb_rank;
    ierr = _callMetis(*comm, offset, my_graph, (nb_rank==1) ? repart_seq_func : repart_func);
  }

  if (print_digest) {
    MetisGraphDigest d(pm);
    String digest = d.computeOutputDigest(my_graph, edgecut);
    if (my_rank == 0) {
      tm->info() << "signature des sorties Metis = " << digest;
    }
  }
  
  return ierr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
