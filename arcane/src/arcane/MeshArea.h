// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshArea.h                                                  (C) 2000-2005 */
/*                                                                           */
/* Zone du maillage.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHAREA_H
#define ARCANE_MESHAREA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/IMeshArea.h"
#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Zone du maillage.
 */
class ARCANE_CORE_EXPORT MeshArea
: public IMeshArea
{
 public:

  MeshArea(IMesh* mesh);
  virtual ~MeshArea();

 public:

  //! Sous-domaine associé
  virtual ISubDomain* subDomain();

  //! Gestionnaire de trace associé
  virtual ITraceMng* traceMng();

  //! Maillage auquel appartient la zone
  virtual IMesh* mesh();

 public:

  //! Nombre de noeuds du maillage
  virtual Integer nbNode();

  //! Nombre d'arêtes du maillage
  virtual Integer nbEdge();

  //! Nombre de faces du maillage
  virtual Integer nbFace();

  //! Nombre de mailles du maillage
  virtual Integer nbCell();

  //! Nombre d'éléments du genre \a ik
  virtual Integer nbItem(eItemKind ik);

 public:

  //! Groupe de tous les noeuds
  virtual NodeGroup allNodes();

  //! Groupe de tous les arêtes
  virtual EdgeGroup allEdges();

  //! Groupe de toutes les faces
  virtual FaceGroup allFaces();

  //! Groupe de toutes les mailles
  virtual CellGroup allCells();

  //! Groupe de toutes les entités du genre \a item_kind
  virtual ItemGroup allItems(eItemKind item_kind);

  //! Groupe de tous les noeuds propres au domaine
  virtual NodeGroup ownNodes();

  //! Groupe de tous les arêtes propres au domaine
  virtual EdgeGroup ownEdges();

  //! Groupe de toutes les faces propres au domaine
  virtual FaceGroup ownFaces();

  //! Groupe de toutes les mailles propres au domaine
  virtual CellGroup ownCells();

  //! Groupe de toutes les entités propres au sous-domaine du genre \a item_kind
  virtual ItemGroup ownItems(eItemKind item_kind);

 public:

  //void setArea(const NodeGroup& nodes,const EdgeGroup& edges,
  //const FaceGroup& faces,const CellGroup& cells);

  void setArea(const NodeGroup& nodes,const CellGroup& cells);

 protected:

  IMesh* m_mesh;
  ISubDomain* m_sub_domain;
  ITraceMng* m_trace_mng;
  NodeGroup m_all_nodes;
  EdgeGroup m_all_edges;
  FaceGroup m_all_faces;
  CellGroup m_all_cells;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

