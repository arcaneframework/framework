// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephOrdering.cc                                            (C) 2000-2024 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephArcane.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// *****************************************************************************
// * Minimal AlephOrdering for AlephIndexing
// *****************************************************************************
AlephOrdering::
AlephOrdering(AlephKernel* kernel)
: TraceAccessor(kernel->parallel()->traceMng())
, m_do_swap(false)
, m_kernel(kernel)
, m_swap(0)
{}

/******************************************************************************
 *****************************************************************************/
AlephOrdering::
AlephOrdering(AlephKernel* kernel,
              Integer global_nb_row,
              Integer local_nb_row,
              bool do_swap)
: TraceAccessor(kernel->parallel()->traceMng())
, m_do_swap(do_swap)
, m_kernel(kernel)
, m_swap(0)
{
  ItacFunction(AlephOrdering);
  if (!do_swap) {
    debug() << "\t[AlephOrdering::AlephOrdering] No ordering!";
    return;
  }
  debug() << "\t[AlephOrdering::AlephOrdering] Ordering!";

  if (m_kernel->nbRanksPerSolver() != 1)
    throw FatalErrorException("AlephOrdering", "Ordering not allowed in parallel");

  Integer local_nb_cell = m_kernel->subDomain()->defaultMesh()->ownCells().size();
  Integer total_nb_cell = m_kernel->subDomain()->parallelMng()->reduce(Parallel::ReduceSum, local_nb_cell);

  if ((local_nb_cell == local_nb_row) && (total_nb_cell == global_nb_row)) {
    debug() << "\t[AlephOrdering::AlephOrdering] Now cell ordering";
    this->initCellOrder();
    return;
  }

  if ((2 * local_nb_cell == local_nb_row) && (2 * total_nb_cell == global_nb_row)) {
    debug() << "\t[AlephOrdering::AlephOrdering] Now 2*cell ordering";
    this->initTwiceCellOrder();
    return;
  }

  Integer local_nb_face = m_kernel->subDomain()->defaultMesh()->ownFaces().size();
  Integer total_nb_face = m_kernel->subDomain()->parallelMng()->reduce(Parallel::ReduceSum, local_nb_face);
  if (((local_nb_cell + local_nb_face) == local_nb_row) && ((total_nb_cell + total_nb_face) == global_nb_row)) {
    debug() << "\t[AlephOrdering::AlephOrdering] Now cell+face ordering";
    this->initCellFaceOrder();
    return;
  }

  if ((local_nb_face == local_nb_row) && (total_nb_face == global_nb_row)) {
    debug() << "\t[AlephOrdering::AlephOrdering] Now face ordering";
    this->initFaceOrder();
    return;
  }

  Integer local_nb_node = m_kernel->subDomain()->defaultMesh()->ownNodes().size();
  Integer total_nb_node = m_kernel->subDomain()->parallelMng()->reduce(Parallel::ReduceSum, local_nb_node);

  if ((((local_nb_cell + local_nb_node)) == local_nb_row) && (((total_nb_cell + total_nb_node)) == global_nb_row)) {
    debug() << "\t[AlephOrdering::AlephOrdering] Now (cell+node) ordering";
    this->initCellNodeOrder();
    return;
  }

  if (((2 * (local_nb_cell + local_nb_node)) == local_nb_row) && ((2 * (total_nb_cell + total_nb_node)) == global_nb_row)) {
    debug() << "\t[AlephOrdering::AlephOrdering] Now 2*(cell+node) ordering";
    this->initTwiceCellNodeOrder();
    return;
  }

  throw FatalErrorException("AlephOrdering", "Could not guess cell||face||cell+face");
}

/******************************************************************************
 *****************************************************************************/
AlephOrdering::
~AlephOrdering()
{
  debug() << "\33[5m\t[~AlephOrdering]\33[0m";
}

/******************************************************************************
 * initCellOrder
 *****************************************************************************/
void AlephOrdering::
initCellOrder(void)
{
  ItacFunction(AlephOrdering);
  debug() << "\t[AlephOrdering::InitializeCellOrder] " << m_kernel->topology()->gathered_nb_row(m_kernel->size());
  m_swap.resize(m_kernel->topology()->gathered_nb_row(m_kernel->size()));
  UniqueArray<Int64> all;
  Integer added = 0;
  ENUMERATE_CELL (cell, m_kernel->subDomain()->defaultMesh()->ownCells()) {
    all.add(cell->uniqueId().asInt64());
    added += 1;
  }
  debug() << "\t[AlephOrdering::InitializeCellOrder] added=" << added;
  m_kernel->parallel()->allGatherVariable(all, m_swap);
}

/******************************************************************************
 * initTwiceCellOrder
 * Pour le mode Complexe
 *****************************************************************************/
void AlephOrdering::
initTwiceCellOrder(void)
{
  ItacFunction(AlephOrdering);
  debug() << "\t[AlephOrdering::InitializeTwiceCellOrder] " << m_kernel->topology()->gathered_nb_row(m_kernel->size());
  m_swap.resize(m_kernel->topology()->gathered_nb_row(m_kernel->size()));
  UniqueArray<Int64> all;
  Integer added = 0;
  ENUMERATE_CELL (cell, m_kernel->subDomain()->defaultMesh()->ownCells()) {
    all.add(2 * cell->uniqueId().asInt64());
    added += 1;
    all.add(2 * cell->uniqueId().asInt64() + 1);
    added += 1;
  }
  debug() << "\t[AlephOrdering::InitializeTwiceCellOrder] added=" << added;
  m_kernel->parallel()->allGatherVariable(all, m_swap);
}

/******************************************************************************
 * initFaceOrder
 *****************************************************************************/
void AlephOrdering::
initFaceOrder(void)
{
  ItacFunction(AlephOrdering);
  debug() << "\t[AlephOrdering::InitializeFaceOrder] " << m_kernel->topology()->gathered_nb_row(m_kernel->size());
  m_swap.resize(m_kernel->topology()->gathered_nb_row(m_kernel->size()));
  UniqueArray<Int64> all;
  Integer added = 0;
  ENUMERATE_FACE (face, m_kernel->subDomain()->defaultMesh()->ownFaces()) {
    all.add(face->uniqueId().asInt64());
    added += 1;
  }
  debug() << "\t[AlephOrdering::InitializeFaceOrder] added=" << added;
  m_kernel->parallel()->allGatherVariable(all, m_swap);
}

/******************************************************************************
 * initCellFaceOrder
 *****************************************************************************/
void AlephOrdering::
initCellFaceOrder(void)
{
  ItacFunction(AlephOrdering);
  debug() << "\t[AlephOrdering::InitializeCellFaceOrder] " << m_kernel->topology()->gathered_nb_row(m_kernel->size());

  UniqueArray<Integer> all_cells;
  UniqueArray<Integer> all_faces;
  UniqueArray<Integer> gathered_nb_cells(m_kernel->size());
  UniqueArray<Integer> gathered_nb_faces(m_kernel->size());
  all_cells.add(m_kernel->subDomain()->defaultMesh()->ownCells().size());
  all_faces.add(m_kernel->subDomain()->defaultMesh()->ownFaces().size());
  m_kernel->parallel()->allGather(all_cells, gathered_nb_cells);
  m_kernel->parallel()->allGather(all_faces, gathered_nb_faces);
  /*  for(Integer i=0,N=gathered_nb_cells.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] gathered_nb_cells["<<i<<"]="<<gathered_nb_cells.at(i);
  for(Integer i=0,N=gathered_nb_faces.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] gathered_nb_faces["<<i<<"]="<<gathered_nb_faces.at(i);
*/

  UniqueArray<Int64> all;
  UniqueArray<Int64> m_swap_cell;
  ENUMERATE_CELL (cell, m_kernel->subDomain()->defaultMesh()->ownCells()) {
    all.add(cell->uniqueId().asInt64());
  }
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] added for cells="<<all.size();
  m_kernel->parallel()->allGatherVariable(all, m_swap_cell);
  /*  for(Integer i=0,N=m_swap_cell.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap_cell["<<i<<"]="<<m_swap_cell.at(i);
*/
  all.clear();
  UniqueArray<Int64> m_swap_face;
  ENUMERATE_FACE (face, m_kernel->subDomain()->defaultMesh()->ownFaces()) {
    all.add(face->uniqueId().asInt64());
  }
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] added for faces="<<all.size();
  m_kernel->parallel()->allGatherVariable(all, m_swap_face);
  /*  for(Integer i=0,N=m_swap_face.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap_face["<<i<<"]="<<m_swap_face.at(i);
*/

  Int64 cell_offset = m_swap_cell.size();
  //  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Now combining cells+faces of size="<<m_swap_cell.size()+m_swap_face.size();
  /*  m_swap.resize(m_swap_cell.size()+m_swap_face.size());
  m_swap.copy(m_swap_cell.constView());
  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Shifting faces of "<<m_swap_cell.size();
  for(Integer i=0,N=m_swap_face.size();i<N;++i)
    m_swap_face[i]+=cell_offset;
  m_swap.addRange(m_swap_face.constView());
*/

  m_swap.resize(m_swap_cell.size() + m_swap_face.size());
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Now combining cells";
  Integer iCell = 0;
  for (Integer i = 0; i < m_kernel->size(); ++i) {
    Integer offset = m_kernel->topology()->gathered_nb_row(i);
    for (Integer j = 0; j < gathered_nb_cells.at(i); ++j) {
      //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap["<<offset+j<<"]="<<m_swap_cell.at(iCell);
      m_swap[offset + j] = m_swap_cell.at(iCell);
      iCell += 1;
    }
  }
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Now combining faces";
  Integer iFace = 0;
  for (Integer i = 0; i < m_kernel->size(); ++i) {
    Integer offset = 0;
    if (i > 0)
      offset = m_kernel->topology()->gathered_nb_row(i);
    offset += gathered_nb_cells.at(i);
    for (Integer j = 0; j < gathered_nb_faces.at(i); ++j) {
      //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap["<<offset+j<<"]="<<m_swap_face.at(iFace);
      m_swap[offset + j] = cell_offset + m_swap_face.at(iFace);
      iFace += 1;
    }
  }

  /*  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Like it?";
  for(Integer i=0,N=m_swap.size();i<N;++i){
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap["<<i<<"]="<<m_swap.at(i);
    }*/
}

/******************************************************************************
 * initCellNodeOrder
 *****************************************************************************/
void AlephOrdering::
initCellNodeOrder(void)
{
  ItacFunction(AlephOrdering);
  debug() << "\t[AlephOrdering::InitializeCellNodeOrder] " << m_kernel->topology()->gathered_nb_row(m_kernel->size());

  UniqueArray<Integer> all_cells;
  UniqueArray<Integer> all_nodes;
  UniqueArray<Integer> gathered_nb_cells(m_kernel->size());
  UniqueArray<Integer> gathered_nb_nodes(m_kernel->size());
  all_cells.add(m_kernel->subDomain()->defaultMesh()->ownCells().size());
  all_nodes.add(m_kernel->subDomain()->defaultMesh()->ownNodes().size());
  m_kernel->parallel()->allGather(all_cells, gathered_nb_cells);
  m_kernel->parallel()->allGather(all_nodes, gathered_nb_nodes);

  UniqueArray<Int64> all;
  UniqueArray<Int64> m_swap_cell;
  ENUMERATE_CELL (cell, m_kernel->subDomain()->defaultMesh()->ownCells()) {
    all.add(cell->uniqueId().asInt64());
  }
  m_kernel->parallel()->allGatherVariable(all, m_swap_cell);
  all.clear();
  UniqueArray<Int64> m_swap_node;
  ENUMERATE_NODE (node, m_kernel->subDomain()->defaultMesh()->ownNodes()) {
    all.add(node->uniqueId().asInt64());
  }
  m_kernel->parallel()->allGatherVariable(all, m_swap_node);

  Int64 cell_offset = m_swap_cell.size();

  m_swap.resize(m_swap_cell.size() + m_swap_node.size());
  Integer iCell = 0;
  for (Integer i = 0; i < m_kernel->size(); ++i) {
    AlephInt offset = m_kernel->topology()->gathered_nb_row(i);
    for (Integer j = 0; j < gathered_nb_cells.at(i); ++j) {
      m_swap[offset + j] = m_swap_cell.at(iCell);
      iCell += 1;
    }
  }
  Integer iNode = 0;
  for (Integer i = 0; i < m_kernel->size(); ++i) {
    AlephInt offset = 0;
    if (i > 0)
      offset = m_kernel->topology()->gathered_nb_row(i);
    offset += gathered_nb_cells.at(i);
    for (Integer j = 0; j < gathered_nb_nodes.at(i); ++j) {
      m_swap[offset + j] = cell_offset + m_swap_node.at(iNode);
      iNode += 1;
    }
  }
}

void AlephOrdering::
initTwiceCellNodeOrder(void)
{
  ItacFunction(AlephOrdering);
  debug() << "\t[AlephOrdering::initTwiceCellNodeOrder] " << m_kernel->topology()->gathered_nb_row(m_kernel->size());

  UniqueArray<Integer> all_cells;
  UniqueArray<Integer> all_nodes;
  UniqueArray<Integer> gathered_nb_cells(m_kernel->size());
  UniqueArray<Integer> gathered_nb_nodes(m_kernel->size());
  all_cells.add(m_kernel->subDomain()->defaultMesh()->ownCells().size());
  all_nodes.add(m_kernel->subDomain()->defaultMesh()->ownNodes().size());
  m_kernel->parallel()->allGather(all_cells, gathered_nb_cells);
  m_kernel->parallel()->allGather(all_nodes, gathered_nb_nodes);

  UniqueArray<Int64> all;
  UniqueArray<Int64> m_swap_cell;
  ENUMERATE_CELL (cell, m_kernel->subDomain()->defaultMesh()->ownCells()) {
    all.add(2 * cell->uniqueId().asInt64());
    all.add(2 * cell->uniqueId().asInt64() + 1);
  }
  m_kernel->parallel()->allGatherVariable(all, m_swap_cell);
  all.clear();
  UniqueArray<Int64> m_swap_node;
  ENUMERATE_NODE (node, m_kernel->subDomain()->defaultMesh()->ownNodes()) {
    all.add(2 * node->uniqueId().asInt64());
    all.add(2 * node->uniqueId().asInt64() + 1);
  }
  m_kernel->parallel()->allGatherVariable(all, m_swap_node);

  Int64 cell_offset = m_swap_cell.size();

  m_swap.resize(m_swap_cell.size() + m_swap_node.size());
  Integer iCell = 0;
  for (Integer i = 0; i < m_kernel->size(); ++i) {
    Integer offset = m_kernel->topology()->gathered_nb_row(i);
    for (Integer j = 0; j < gathered_nb_cells.at(i); ++j) {
      m_swap[offset + j] = m_swap_cell.at(iCell);
      iCell += 1;
    }
  }
  Integer iNode = 0;
  for (Integer i = 0; i < m_kernel->size(); ++i) {
    Integer offset = 0;
    if (i > 0)
      offset = m_kernel->topology()->gathered_nb_row(i);
    offset += gathered_nb_cells.at(i);
    for (Integer j = 0; j < gathered_nb_nodes.at(i); ++j) {
      m_swap[offset + j] = cell_offset + m_swap_node.at(iNode);
      iNode += 1;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
