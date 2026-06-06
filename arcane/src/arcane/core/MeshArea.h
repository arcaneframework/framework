// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshArea.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Mesh area.                                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHAREA_H
#define ARCANE_CORE_MESHAREA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshArea.h"
#include "arcane/core/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh area.
 */
class ARCANE_CORE_EXPORT MeshArea
: public IMeshArea
{
 public:

  explicit MeshArea(IMesh* mesh);
  ~MeshArea() override;

 public:

  //! Associated sub-domain
  ISubDomain* subDomain() override;

  //! Associated trace manager
  ITraceMng* traceMng() override;

  //! Mesh to which the area belongs
  IMesh* mesh() override;

 public:

  //! Number of mesh nodes
  Integer nbNode() override;

  //! Number of mesh edges
  virtual Integer nbEdge();

  //! Number of mesh faces
  virtual Integer nbFace();

  //! Number of mesh cells
  Integer nbCell() override;

  //! Number of elements of type \a ik
  virtual Integer nbItem(eItemKind ik);

 public:

  //! Group of all nodes
  NodeGroup allNodes() override;

  //! Group of all edges
  virtual EdgeGroup allEdges();

  //! Group of all faces
  virtual FaceGroup allFaces();

  //! Group of all cells
  CellGroup allCells() override;

  //! Group of all entities of type \a item_kind
  virtual ItemGroup allItems(eItemKind item_kind);

  //! Group of all nodes belonging to the domain
  NodeGroup ownNodes() override;

  //! Group of all edges belonging to the domain
  virtual EdgeGroup ownEdges();

  //! Group of all faces belonging to the domain
  virtual FaceGroup ownFaces();

  //! Group of all cells belonging to the domain
  virtual CellGroup ownCells() override;

  //! Group of all entities belonging to the sub-domain of type \a item_kind
  virtual ItemGroup ownItems(eItemKind item_kind);

 public:

  void setArea(const NodeGroup& nodes, const CellGroup& cells);

 protected:

  IMesh* m_mesh = nullptr;
  ISubDomain* m_sub_domain = nullptr;
  ITraceMng* m_trace_mng = nullptr;
  NodeGroup m_all_nodes;
  EdgeGroup m_all_edges;
  FaceGroup m_all_faces;
  CellGroup m_all_cells;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
