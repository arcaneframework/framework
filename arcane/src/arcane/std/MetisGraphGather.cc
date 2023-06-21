// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphGather.h                                          (C) 2000-2023 */
/*                                                                           */
/* Regroupement de graphes de 'Parmetis'.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/MetisGraphGather.h"
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/ 

namespace
{
template <class src_type, class dest_type>
void convertVector(const int size, ConstArrayView<src_type> src, ArrayView<dest_type> dest)
{
  for (int i = 0; i < size; ++i) {
    dest[i] = src[i];
  }
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void MetisGraphGather::
gatherGraph(const bool need_part, const String&, MPI_Comm comm,
            ConstArrayView<idx_t> vtxdist, const idx_t ncon, MetisGraphView my_graph,
            MetisGraph& graph)
{
  int my_rank = -1;
  int nb_rank = -1;
  int io_rank = 0;
  
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nb_rank);
  
  // buffers
  
  UniqueArray<int> my_buffer; // pour les entrees des routines MPI
  UniqueArray<int> buffer; // pour les sorties des routines MPI
  UniqueArray<int> offset; // pour les gather
  
  // nombre de sommets du graph complet
  
  if (my_rank == io_rank) {
    graph.nb_vertices = vtxdist[nb_rank];
    graph.have_vsize = my_graph.have_vsize; // on suppose que tous les processeurs ont la meme valeur
    graph.have_adjwgt = my_graph.have_adjwgt; // on suppose que tous les processeurs ont la meme valeur
  } else {
    graph.nb_vertices = 0;
    graph.have_vsize = false;
    graph.have_adjwgt = false;
  }
  
  // recupere les dimensions caracterisant la repartition du graph sur les processeurs

  my_buffer.resize(2);
  if (my_rank == io_rank) {
    offset.resize(nb_rank);
    buffer.resize(2 * nb_rank);
  }

  my_buffer[0] = my_graph.nb_vertices; // nb de sommets
  my_buffer[1] = my_graph.adjncy.size(); // taille de la liste d'adjacences

  MPI_Gather((void*)my_buffer.data(), 2, MPI_INT, (void*)buffer.data(), 2, MPI_INT, io_rank, comm);

  int adjcny_size = 0;
  UniqueArray<int> nb_vertices_per_rank(nb_rank);
  UniqueArray<int> adjncy_size_per_rank(nb_rank);
  UniqueArray<int> nb_vertice_cons_per_rank(nb_rank);
  if (my_rank == io_rank) {
    for (int rank = 0; rank < nb_rank; ++rank) {
      nb_vertices_per_rank[rank] = buffer[2 * rank];
      adjncy_size_per_rank[rank] = buffer[2 * rank + 1];
      nb_vertice_cons_per_rank[rank] = nb_vertices_per_rank[rank] * ncon;
      adjcny_size += adjncy_size_per_rank[rank];
    }
  }

  // retaille les buffers maintenant que les dimensions sont connues (le plus gros tableau est adjcny)
  
  int max_buffer_size = my_graph.nb_vertices * ncon;
  max_buffer_size = std::max(my_graph.adjncy.size(), max_buffer_size);
  my_buffer.resize(max_buffer_size);
  if (my_rank == io_rank) {
    max_buffer_size = graph.nb_vertices * ncon;
    max_buffer_size = std::max(adjcny_size, max_buffer_size);
    buffer.resize(max_buffer_size);
  }

  // Recupere la liste d'adjacences.

  if (my_rank == io_rank) {
    offset[0] = 0;
    for (int rank = 1; rank < nb_rank; ++rank) {
      offset[rank] = offset[rank - 1] + adjncy_size_per_rank[rank - 1];
    }
    graph.adjncy.resize(adjcny_size);
  }

  convertVector(my_graph.adjncy.size(), my_graph.adjncy.constView(), my_buffer.view());
  
  MPI_Gatherv(my_buffer.data(), my_graph.adjncy.size(), MPI_INT,
              buffer.data(), adjncy_size_per_rank.data(),
              offset.data(), MPI_INT, io_rank, comm);
              
  convertVector(adjcny_size, buffer.constView(), graph.adjncy.view());

  // Recupere la liste des poids aux arretes du graph.
  
  if (my_graph.have_adjwgt) {
    if (my_rank == io_rank) {
      graph.adjwgt.resize(adjcny_size);
    }
    
    convertVector(my_graph.adjwgt.size(), my_graph.adjwgt.constView(), my_buffer.view());
    
    MPI_Gatherv(my_buffer.data(), my_graph.adjwgt.size(), MPI_INT,
                buffer.data(), adjncy_size_per_rank.data(),
                offset.data(), MPI_INT, io_rank, comm);
                
    convertVector(adjcny_size, buffer.constView(), graph.adjwgt.view());            
  }
  
  // Recupere la liste des indexes dans la liste d'adjacences.
  // A cette etape, la liste recuperee n'est pas correcte, sauf le dernier indice.
  // En effet les indexes par processeur ne sont pas les meme que les indexes 
  // dans le liste d'adjacences complete.
  
  if (my_rank == io_rank) {
    graph.xadj.resize(graph.nb_vertices + 1);
    graph.xadj[graph.nb_vertices] = graph.adjncy.size();
    offset[0] = 0;
    for (int rank = 1; rank < nb_rank; ++rank) {
      offset[rank] = offset[rank - 1] + nb_vertices_per_rank[rank - 1];
    }
  }

  convertVector(my_graph.nb_vertices, my_graph.xadj.constView(), my_buffer.view());

  MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices, MPI_INT,
              buffer.data(), nb_vertices_per_rank.data(),
              offset.data(), MPI_INT, io_rank, comm);

  convertVector(graph.nb_vertices, buffer.constView(), graph.xadj.view());
           
  // Correction des indexes
  
  if (my_rank == io_rank) {
    int start_adjncy_index = 0;
    for (int rank = 1; rank < nb_rank; ++rank) {
      start_adjncy_index += adjncy_size_per_rank[rank-1];
      //std::cerr << "rank " << rank << " offset " << start_adjncy_index << "  vtxdist[rank] " <<  vtxdist[rank] << " vtxdist[rank+1] " << vtxdist[rank+1] << std::endl;
      for (int ixadj = vtxdist[rank]; ixadj < vtxdist[rank+1]; ++ixadj) {
        graph.xadj[ixadj] +=  start_adjncy_index;
      }
    }
  }

  // Recupere la liste des poids "memoire" aux sommets du graph.

  if (my_graph.have_vsize) {
    if (my_rank == io_rank) {
      graph.vsize.resize(graph.nb_vertices);
    }

    convertVector(my_graph.nb_vertices, my_graph.vsize.constView(), my_buffer.view());
    
    MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices, MPI_INT,
                buffer.data(), nb_vertices_per_rank.data(),
                offset.data(), MPI_INT, io_rank, comm);
                
    convertVector(graph.nb_vertices, buffer.constView(), graph.vsize.view());
  }

  // Recupere la liste des numeros de sous-domaine d'un precedent partitionnement
  
  if (my_rank == io_rank) {
    graph.part.resize(graph.nb_vertices);
  }
  
  if (need_part) {

    convertVector(my_graph.nb_vertices, my_graph.part.constView(), my_buffer.view());
    
    MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices, MPI_INT,
                buffer.data(), nb_vertices_per_rank.data(),
                offset.data(), MPI_INT, io_rank, comm);
                
    convertVector(graph.nb_vertices, buffer.constView(), graph.part.view());
  }

  // Recupere la liste des poids aux sommets du graph. Il peut y en avoir plusieurs (ncon).

  if (my_rank == io_rank) {
    graph.vwgt.resize(graph.nb_vertices * ncon);
    for (auto& x : offset) {
      x *= ncon;
    }
  }
  
  convertVector(my_graph.nb_vertices * ncon, my_graph.vwgt.constView(), my_buffer.view());
  
  MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices * ncon, MPI_INT,
              buffer.data(), nb_vertice_cons_per_rank.data(),
              offset.data(), MPI_INT, io_rank, comm);
              
  convertVector(graph.nb_vertices * ncon, buffer.constView(), graph.vwgt.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphGather::
scatterPart(MPI_Comm comm, ConstArrayView<idx_t> vtxdist, ConstArrayView<idx_t> part,
            ArrayView<idx_t> my_part)
{
  int my_rank = -1;
  int nb_rank = -1;
  int io_rank = 0;
  
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nb_rank);
  
  UniqueArray<int> nb_vertices(nb_rank);
  UniqueArray<int> displ(nb_rank);
  
  for (int rank = 0; rank < nb_rank; ++rank) {
    displ[rank] = vtxdist[rank];
    nb_vertices[rank] = vtxdist[rank+1] - vtxdist[rank];
  }
  
  UniqueArray<int> send_buffer;
  UniqueArray<int> recv_buffer(my_part.size());

  if (my_rank == io_rank) {
    send_buffer.resize(part.size());
  }
  
  convertVector(send_buffer.size(), part, send_buffer.view());
  
  MPI_Scatterv(send_buffer.data(), nb_vertices.data(),
               displ.data(), MPI_INT, recv_buffer.data(),
               my_part.size(), MPI_INT, io_rank, comm);
  
  convertVector(recv_buffer.size(), recv_buffer.constView(), my_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
