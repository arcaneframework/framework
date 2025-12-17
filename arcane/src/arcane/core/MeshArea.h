// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshArea.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Zone du maillage.                                                         */
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
 * \brief Zone du maillage.
 */
class ARCANE_CORE_EXPORT MeshArea
: public IMeshArea
{
 public:

  explicit MeshArea(IMesh* mesh);
  ~MeshArea() override;

 public:

  //! Sous-domaine associé
  ISubDomain* subDomain() override;

  //! Gestionnaire de trace associé
  ITraceMng* traceMng() override;

  //! Maillage auquel appartient la zone
  IMesh* mesh() override;

 public:

  //! Nombre de noeuds du maillage
  Integer nbNode() override;

  //! Nombre d'arêtes du maillage
  virtual Integer nbEdge();

  //! Nombre de faces du maillage
  virtual Integer nbFace();

  //! Nombre de mailles du maillage
  Integer nbCell() override;

  //! Nombre d'éléments du genre \a ik
  virtual Integer nbItem(eItemKind ik);

 public:

  //! Groupe de tous les noeuds
  NodeGroup allNodes() override;

  //! Groupe de tous les arêtes
  virtual EdgeGroup allEdges();

  //! Groupe de toutes les faces
  virtual FaceGroup allFaces();

  //! Groupe de toutes les mailles
  CellGroup allCells() override;

  //! Groupe de toutes les entités du genre \a item_kind
  virtual ItemGroup allItems(eItemKind item_kind);

  //! Groupe de tous les noeuds propres au domaine
  NodeGroup ownNodes() override;

  //! Groupe de tous les arêtes propres au domaine
  virtual EdgeGroup ownEdges();

  //! Groupe de toutes les faces propres au domaine
  virtual FaceGroup ownFaces();

  //! Groupe de toutes les mailles propres au domaine
  virtual CellGroup ownCells() override;

  //! Groupe de toutes les entités propres au sous-domaine du genre \a item_kind
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

