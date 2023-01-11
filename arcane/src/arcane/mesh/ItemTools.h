// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTools.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Utilitaires aidant à retrouver des items à partir d'autres                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMTOOLS_H
#define ARCANE_MESH_ITEMTOOLS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Utilitaires aidant à retrouver des items à partir d'autres
 */
class ItemTools
{
 public:

  /*!
   * Vérifie si la liste des noeuds d'une face correspond à une liste fournie
   * On compare les uids des noeuds. L'ordre doit être le même.
   *
   * @param face : la face à tester
   * @param face_nodes_uid : une liste de uids de noeuds
   *
   */
  static bool isSameFace(Face face, Int64ConstArrayView face_nodes_uid);

  /*!
   * Recherche une face connectée au noeud \a node correspondant à la liste de
   * noeuds \a  face_nodes_uid.
   *
   * @param node : noeud à tester
   * @param face_type_id : type de la face recherchée
   * @param face_nodes_uid : une liste de uids de noeuds
   *
   */
  static Face findFaceInNode2(Node node,
                              Integer face_type_id,
                              Int64ConstArrayView face_nodes_uid);

  /*!
   * Recherche une arête connectée à un noeud \a node et connectant les noeuds 
   * d'uids \a begin_node et \a end_node
   *
   * @param node : noeud à tester
   * @param begin_node : uid du premier noeud de l'arête recherchée
   * @param end_node : uid du second noeud de l'arête recherchée
   *
   */
  static Edge findEdgeInNode2(Node node, Int64 begin_node, Int64 end_node);

 private:

  /*!
   * Recherche une face connectée au noeud \a node correspondant à la liste de
   * noeuds \a  face_nodes_uid.
   *
   * @param node : noeud à tester
   * @param face_type_id : type de la face recherchée
   * @param face_nodes_uid : une liste de uids de noeuds
   *
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use findFaceInNode2() instead")
  static ItemInternal* findFaceInNode(Node node,
                                      Integer face_type_id,
                                      Int64ConstArrayView face_nodes_uid);

  /*!
   * Recherche une arête connectée à un noeud \a node et connectant les noeuds
   * d'uids \a begin_node et \a end_node
   *
   * @param node : noeud à tester
   * @param begin_node : uid du premier noeud de l'arête recherchée
   * @param end_node : uid du second noeud de l'arête recherchée
   *
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use findEdgeInNode2() instead")
  static ItemInternal* findEdgeInNode(Node node, Int64 begin_node, Int64 end_node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_ITEMTOOLS_H */
