// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerBuilder.h                                         (C) 2000-2024 */
/*                                                                           */
/* Construction des couches fantômes.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_GHOSTLAYERBUILDER_H
#define ARCANE_MESH_GHOSTLAYERBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class SerializeBuffer;
}

namespace Arcane::mesh
{
class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des couches fantômes.
 */
class GhostLayerBuilder
: public TraceAccessor
{
 public:

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;

  typedef HashTableMapT<Int32,SharedArray<Int64> > BoundaryInfosMap;
  typedef HashTableMapEnumeratorT<Int32,SharedArray<Int64> > BoundaryInfosMapEnumerator;
  
 public:

  //! Construit une instance pour le maillage \a mesh
  explicit GhostLayerBuilder(DynamicMeshIncrementalBuilder* mesh_builder);
  virtual ~GhostLayerBuilder();

 public:

  void addGhostLayers(bool is_allocate);

  //! AMR
  void addGhostChildFromParent();
  void addGhostChildFromParent2(Array<Int64>& ghost_cell_to_refine);

 private:

  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;

 private:
  
  void _addOneGhostLayerV2();
  void _exchangeData(IParallelExchanger* exchanger,BoundaryInfosMap& boundary_infos_to_send);
  void _printItem(ItemInternal* ii,std::ostream& o);
  void _exchangeCells(HashTableMapT<Int32,SharedArray<Int32>>& cells_to_send,bool with_flags);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//TOTO A definir ailleurs
class NodeUidToSubDomain
{
 public:
  NodeUidToSubDomain(Int64 max_uid,Int32 nb_rank);

  Int32 uidToRank(Int64 uid)
  {
    Int32 rank = (Int32)(uid / m_nb_by_rank);
    if (rank>=m_nb_rank)
      --rank;
    Int32 nrank = rank % m_modulo;
    return nrank;
  }
  Int32 modulo() const { return m_modulo; }

 private:

  Int32 m_nb_rank;
  Int32 m_modulo;
  Int64 m_nb_by_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
