// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphGather.cc                                         (C) 2000-2024 */
/*                                                                           */
/* Graph gathering for 'Parmetis'.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/MetisGraphGather.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
MPI_Comm _getMPICommunicator(Arcane::IParallelMng* pm)
{
  return static_cast<MPI_Comm>(pm->communicator());
}
} // namespace

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MetisGraphGather::
MetisGraphGather(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_my_rank(pm->commRank())
, m_nb_rank(pm->commSize())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class SourceType, class TargetType> void MetisGraphGather::
_convertVector(const int size, ConstArrayView<SourceType> src, ArrayView<TargetType> dest)
{
  if (size > src.size())
    ARCANE_FATAL("Source size is too small size={0} src_size={1}", size, src.size());
  if (size > dest.size())
    ARCANE_FATAL("Target size is too small size={0} target_size={1}", size, dest.size());
  for (int i = 0; i < size; ++i) {
    dest[i] = static_cast<TargetType>(src[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphGather::
gatherGraph(const bool need_part, ConstArrayView<idx_t> vtxdist, const int ncon, MetisGraphView my_graph,
            MetisGraph& graph)
{
  MPI_Comm comm = _getMPICommunicator(m_parallel_mng);
  info() << "Metis: gather graph";

  const Int32 io_rank = m_parallel_mng->masterIORank();
  const Int32 nb_rank = m_nb_rank;
  const bool is_master_io = m_parallel_mng->isMasterIO();
  // buffers
  
  UniqueArray<int> my_buffer; // for the inputs of the MPI routines
  UniqueArray<int> buffer; // for the outputs of the MPI routines
  UniqueArray<int> offset; // for the gather
  
  // number of vertices in the complete graph

  if (m_my_rank == io_rank) {
    graph.nb_vertices = CheckedConvert::toInt32(vtxdist[m_nb_rank]);
    graph.have_vsize = my_graph.have_vsize; // it is assumed that all processors have the same value
    graph.have_adjwgt = my_graph.have_adjwgt; // it is assumed that all processors have the same value
  } else {
    graph.nb_vertices = 0;
    graph.have_vsize = false;
    graph.have_adjwgt = false;
  }

  // retrieves the dimensions characterizing the graph distribution across processors

  my_buffer.resize(2);
  if (m_my_rank == io_rank) {
    offset.resize(nb_rank);
    buffer.resize(2 * nb_rank);
  }

  my_buffer[0] = my_graph.nb_vertices; // number of vertices
  my_buffer[1] = my_graph.adjncy.size(); // size of the adjacency list

  MPI_Gather((void*)my_buffer.data(), 2, MPI_INT, (void*)buffer.data(), 2, MPI_INT, io_rank, comm);

  int adjcny_size = 0;
  UniqueArray<int> nb_vertices_per_rank(nb_rank);
  UniqueArray<int> adjncy_size_per_rank(nb_rank);
  UniqueArray<int> nb_vertice_cons_per_rank(nb_rank);
  if (is_master_io) {
    for (int rank = 0; rank < nb_rank; ++rank) {
      nb_vertices_per_rank[rank] = buffer[2 * rank];
      adjncy_size_per_rank[rank] = buffer[2 * rank + 1];
      nb_vertice_cons_per_rank[rank] = nb_vertices_per_rank[rank] * ncon;
      adjcny_size += adjncy_size_per_rank[rank];
    }
  }

  // resize the buffers now that the dimensions are known (the largest array is adjcny)
  
  int max_buffer_size = my_graph.nb_vertices * ncon;
  max_buffer_size = std::max(my_graph.adjncy.size(), max_buffer_size);
  my_buffer.resize(max_buffer_size);
  if (is_master_io) {
    max_buffer_size = graph.nb_vertices * ncon;
    max_buffer_size = std::max(adjcny_size, max_buffer_size);
    buffer.resize(max_buffer_size);
  }

  // Retrieves the adjacency list.

  if (is_master_io) {
    offset[0] = 0;
    for (int rank = 1; rank < nb_rank; ++rank) {
      offset[rank] = offset[rank - 1] + adjncy_size_per_rank[rank - 1];
    }
    graph.adjncy.resize(adjcny_size);
  }

  _convertVector(my_graph.adjncy.size(), my_graph.adjncy.constView(), my_buffer.view());

  MPI_Gatherv(my_buffer.data(), my_graph.adjncy.size(), MPI_INT,
              buffer.data(), adjncy_size_per_rank.data(),
              offset.data(), MPI_INT, io_rank, comm);

  _convertVector(adjcny_size, buffer.constView(), graph.adjncy.view());

  // Retrieves the list of edge weights.
  
  if (my_graph.have_adjwgt) {
    if (is_master_io) {
      graph.adjwgt.resize(adjcny_size);
    }

    _convertVector(my_graph.adjwgt.size(), my_graph.adjwgt.constView(), my_buffer.view());

    MPI_Gatherv(my_buffer.data(), my_graph.adjwgt.size(), MPI_INT,
                buffer.data(), adjncy_size_per_rank.data(),
                offset.data(), MPI_INT, io_rank, comm);

    _convertVector(adjcny_size, buffer.constView(), graph.adjwgt.view());
  }

  // Retrieves the list of indices in the adjacency list.
  // At this stage, the retrieved list is not correct, except for the last index.
  // In fact, the indices per processor are not the same as the indices
  // in the complete adjacency list.

  if (is_master_io) {
    graph.xadj.resize(graph.nb_vertices + 1);
    graph.xadj[graph.nb_vertices] = graph.adjncy.size();
    offset[0] = 0;
    for (int rank = 1; rank < nb_rank; ++rank) {
      offset[rank] = offset[rank - 1] + nb_vertices_per_rank[rank - 1];
    }
  }

  _convertVector(my_graph.nb_vertices, my_graph.xadj.constView(), my_buffer.view());

  MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices, MPI_INT,
              buffer.data(), nb_vertices_per_rank.data(),
              offset.data(), MPI_INT, io_rank, comm);

  _convertVector(graph.nb_vertices, buffer.constView(), graph.xadj.view());

  // Index correction

  if (is_master_io) {
    int start_adjncy_index = 0;
    for (int rank = 1; rank < nb_rank; ++rank) {
      start_adjncy_index += adjncy_size_per_rank[rank-1];
      //std::cerr << "rank " << rank << " offset " << start_adjncy_index << "  vtxdist[rank] " <<  vtxdist[rank] << " vtxdist[rank+1] " << vtxdist[rank+1] << std::endl;
      Int32 vtxdist_rank = CheckedConvert::toInt32(vtxdist[rank]);
      Int32 vtxdist_rank_plus_one = CheckedConvert::toInt32(vtxdist[rank + 1]);
      for (Int32 ixadj = vtxdist_rank; ixadj < vtxdist_rank_plus_one; ++ixadj) {
        graph.xadj[ixadj] +=  start_adjncy_index;
      }
    }
  }

  // Retrieves the "memory" weights for the graph vertices.

  if (my_graph.have_vsize) {
    if (is_master_io) {
      graph.vsize.resize(graph.nb_vertices);
    }

    _convertVector(my_graph.nb_vertices, my_graph.vsize.constView(), my_buffer.view());

    MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices, MPI_INT,
                buffer.data(), nb_vertices_per_rank.data(),
                offset.data(), MPI_INT, io_rank, comm);

    _convertVector(graph.nb_vertices, buffer.constView(), graph.vsize.view());
  }

  // Retrieves the list of subdomain numbers from a previous partitioning

  if (is_master_io) {
    graph.part.resize(graph.nb_vertices);
  }
  
  if (need_part) {

    _convertVector(my_graph.nb_vertices, my_graph.part.constView(), my_buffer.view());

    MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices, MPI_INT,
                buffer.data(), nb_vertices_per_rank.data(),
                offset.data(), MPI_INT, io_rank, comm);

    _convertVector(graph.nb_vertices, buffer.constView(), graph.part.view());
  }

  // Retrieves the list of weights for the graph vertices. There may be several (ncon).

  if (is_master_io) {
    graph.vwgt.resize(graph.nb_vertices * ncon);
    for (auto& x : offset) {
      x *= ncon;
    }
  }

  _convertVector(my_graph.nb_vertices * ncon, my_graph.vwgt.constView(), my_buffer.view());

  MPI_Gatherv(my_buffer.data(), my_graph.nb_vertices * ncon, MPI_INT,
              buffer.data(), nb_vertice_cons_per_rank.data(),
              offset.data(), MPI_INT, io_rank, comm);

  _convertVector(graph.nb_vertices * ncon, buffer.constView(), graph.vwgt.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphGather::
scatterPart(ConstArrayView<idx_t> vtxdist, ConstArrayView<idx_t> part, ArrayView<idx_t> my_part)
{
  MPI_Comm comm = _getMPICommunicator(m_parallel_mng);

  const Int32 nb_rank = m_nb_rank;
  const bool is_master_io = m_parallel_mng->isMasterIO();
  int io_rank = 0;
  
  UniqueArray<Int32> nb_vertices(nb_rank);
  UniqueArray<Int32> displ(nb_rank);

  for (int rank = 0; rank < nb_rank; ++rank) {
    displ[rank] = CheckedConvert::toInt32(vtxdist[rank]);
    nb_vertices[rank] = CheckedConvert::toInt32(vtxdist[rank + 1] - vtxdist[rank]);
  }
  
  UniqueArray<int> send_buffer;
  UniqueArray<int> recv_buffer(my_part.size());

  if (is_master_io) {
    send_buffer.resize(part.size());
  }

  _convertVector(send_buffer.size(), part, send_buffer.view());

  MPI_Scatterv(send_buffer.data(), nb_vertices.data(),
               displ.data(), MPI_INT, recv_buffer.data(),
               my_part.size(), MPI_INT, io_rank, comm);

  _convertVector(recv_buffer.size(), recv_buffer.constView(), my_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
