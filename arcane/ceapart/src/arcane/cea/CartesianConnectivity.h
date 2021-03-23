// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianConnectivity.h                                     (C) 2000-2014 */
/*                                                                           */
/* Informations de connectivité d'un maillage cartésien.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CEA_CARTESIANCONNECTIVITY_H
#define ARCANE_CEA_CARTESIANCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/Item.h"
#include "arcane/VariableTypedef.h"
#include "arcane/cea/CeaGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellDirectionMng;
class FaceDirectionMng;
class NodeDirectionMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Informations de connectivité d'un maillage cartésien.
 *
 * Pour l'instant, cela ne fonctionne que pour les maillages 2D.
 *
 * Comme tous les objets liés au maillage cartésien, ces instances ne
 * sont valides que tant que la topologie du maillage n'évolue pas.
 *
 * Cette classe sert à la fois pour les connectivités 2D et les connectivités
 * 3D. Les méthodes qui commencent par topZ ne sont valides que en 3D.
 *
 * Le nom des méthodes suit la nomenclature suivante:
 * - topZ/.: pour la direction Z
 * - upper/lower: pour la direction Y
 * - left/right: pour la direction X
 *
 * Pour la connectivité des noeuds autour d'une maille de coordonnées (X0,Y0,Z0),
 * le noeud de coordonnées (X,Y,Z) se récupère comme suit:
 * - En 3D, topZ si Z>Z0, sinon pas de préfixe. en 2D, jamais de préfixe.
 * - upper si Y>Y0, lower sinon,
 * - right si X>X0, left sinon,
 *
 * Donc par exemple, si Z>Z0, Y<Y0 et X>X0, le nom de la méthode est topZLowerRight().
 * Si Z<Z0, Y>Y0 et X>X0, le nom est upperRight().
 *
 * Le fonctionnement est le même pour les connectivités des mailles autour d'un noeud.
 */
class ARCANE_CEA_EXPORT CartesianConnectivity
{
  /*!
   * \brief Type énuméré indiquant la position.
   * \warning Les valeurs exactes ne doivent pas être utilisées car elles sont
   * susceptibles de changer.
   */
  enum ePosition
  {
    P_UpperLeft = 0,
    P_UpperRight = 1,
    P_LowerRight = 2,
    P_LowerLeft = 3,

    P_TopZUpperLeft = 4,
    P_TopZUpperRight = 5,
    P_TopZLowerRight = 6,
    P_TopZLowerLeft = 7
  };

 public:
  
  struct Index
  {
   public:
    void fill(ItemInternal* i){ v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = v[6] = v[7] = i; }
    ItemInternal* v[8];
  };

 public:

  //! Maille en haut à gauche du noeud \a n
  Cell upperLeft(Node n) const { return m_nodes_to_cell[n.localId()].v[P_UpperLeft]; }
  //! Maille en haut à droite du noeud \a n
  Cell upperRight(Node n) const { return m_nodes_to_cell[n.localId()].v[P_UpperRight]; }
  //! Maille en bas à droite du noeud \a n
  Cell lowerRight(Node n) const { return m_nodes_to_cell[n.localId()].v[P_LowerRight]; }
  //! Maille en bas à gauche du noeud \a n
  Cell lowerLeft(Node n) const { return m_nodes_to_cell[n.localId()].v[P_LowerLeft]; }

  //! En 3D, maille en haut à gauche du noeud \a n
  Cell topZUpperLeft(Node n) const { return m_nodes_to_cell[n.localId()].v[P_TopZUpperLeft]; }
  //! En 3D, maille en haut à droite du noeud \a n
  Cell topZUpperRight(Node n) const { return m_nodes_to_cell[n.localId()].v[P_TopZUpperRight]; }
  //! En 3D, maille en bas à droite du noeud \a n
  Cell topZLowerRight(Node n) const { return m_nodes_to_cell[n.localId()].v[P_TopZLowerRight]; }
  //! En 3D, maille en bas à gauche du noeud \a n
  Cell topZLowerLeft(Node n) const { return m_nodes_to_cell[n.localId()].v[P_TopZLowerLeft]; }

  //! Noeud en haut à gauche de la maille \a c
  Node upperLeft(Cell c) const { return m_cells_to_node[c.localId()].v[P_UpperLeft]; }
  //! Noeud en haut à droite de la maille \a c
  Node upperRight(Cell c) const { return m_cells_to_node[c.localId()].v[P_UpperRight]; }
  //! Noeud en bas à droite de la maille \a c
  Node lowerRight(Cell c) const { return m_cells_to_node[c.localId()].v[P_LowerRight]; }
  //! Noeud en bad à gauche de la maille \a c
  Node lowerLeft(Cell c) const { return m_cells_to_node[c.localId()].v[P_LowerLeft]; }

  //! En 3D, noeud au dessus en haut à gauche de la maille \a c
  Node topZUpperLeft(Cell c) const { return m_cells_to_node[c.localId()].v[P_TopZUpperLeft]; }
  //! En 3D, noeud au dessus en haut à droite de la maille \a c
  Node topZUpperRight(Cell c) const { return m_cells_to_node[c.localId()].v[P_TopZUpperRight]; }
  //! En 3D, noeud au dessus en bas à droite de la maille \a c
  Node topZLowerRight(Cell c) const { return m_cells_to_node[c.localId()].v[P_TopZLowerRight]; }
  //! En 3D, noeud au dessus en bas à gauche de la maille \a c
  Node topZLowerLeft(Cell c) const { return m_cells_to_node[c.localId()].v[P_TopZLowerLeft]; }

 public:

  /*!
  ** \name Fonctions internes réservées à Arcane.
  */
  //@{
  //! Calcule les infos de connectivité.
  void computeInfos(IMesh* mesh,VariableNodeReal3& nodes_coord,VariableCellReal3& cells_coord);
  //! Positionne les tableaux contenant les infos de connectivité
  void setStorage(ArrayView<Index> nodes_to_cell,ArrayView<Index> cells_to_node);
  //@}

 private:
  
  Index& _index(Node n) { return m_nodes_to_cell[n.localId()]; }
  Index& _index(Cell c) { return m_cells_to_node[c.localId()]; }

 private:

  ArrayView<Index> m_nodes_to_cell;
  ArrayView<Index> m_cells_to_node;

  void _computeInfos2D(IMesh* mesh,VariableNodeReal3& nodes_coord,VariableCellReal3& cells_coord);
  void _computeInfos3D(IMesh* mesh,VariableNodeReal3& nodes_coord,VariableCellReal3& cells_coord);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

