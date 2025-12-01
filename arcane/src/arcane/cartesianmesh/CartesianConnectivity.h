// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianConnectivity.h                                     (C) 2000-2023 */
/*                                                                           */
/* Informations de connectivité d'un maillage cartésien.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANCONNECTIVITY_H
#define ARCANE_CARTESIANMESH_CARTESIANCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/core/Item.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Informations de connectivité d'un maillage cartésien.
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
class ARCANE_CARTESIANMESH_EXPORT CartesianConnectivity
{
  // NOTE: Pour l'instant, on doit conserver par entité les entités connectées par
  // direction, car la numérotation ne permet pas de les retrouver simplement.
  // À terme, on pourra le déduire directement.

  friend class CartesianConnectivityLocalId;
  friend class CartesianMeshImpl;

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

 private:

  //! Liste des 8 entités autout d'une autre entité
  struct Index
  {
    friend class CartesianConnectivity;

   private:

    void fill(Int32 i) { v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = v[6] = v[7] = i; }
    Int32 v[8];
  };

  //! Permutation dans Index pour chaque direction
  struct Permutation
  {
    friend class CartesianConnectivity;

   public:

    void compute();

   private:

    ePosition permutation[3][8];
  };

 public:

  //! Maille en haut à gauche du noeud \a n
  Cell upperLeft(Node n) const { return _nodeToCell(n, P_UpperLeft); }
  //! Maille en haut à droite du noeud \a n
  Cell upperRight(Node n) const { return _nodeToCell(n, P_UpperRight); }
  //! Maille en bas à droite du noeud \a n
  Cell lowerRight(Node n) const { return _nodeToCell(n, P_LowerRight); }
  //! Maille en bas à gauche du noeud \a n
  Cell lowerLeft(Node n) const { return _nodeToCell(n, P_LowerLeft); }

  //! Maille en haut à gauche du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId upperLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_UpperLeft); }
  //! Maille en haut à droite du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId upperRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_UpperRight); }
  //! Maille en bas à droite du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId lowerRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_LowerRight); }
  //! Maille en bas à gauche du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId lowerLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_LowerLeft); }

  //! Maille en haut à gauche du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId upperLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_UpperLeft); }
  //! Maille en haut à droite du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId upperRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_UpperRight); }
  //! Maille en bas à droite du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId lowerRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_LowerRight); }
  //! Maille en bas à gauche du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId lowerLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_LowerLeft); }

  //! En 3D, maille en haut à gauche du noeud \a n
  Cell topZUpperLeft(Node n) const { return _nodeToCell(n, P_TopZUpperLeft); }
  //! En 3D, maille en haut à droite du noeud \a n
  Cell topZUpperRight(Node n) const { return _nodeToCell(n, P_TopZUpperRight); }
  //! En 3D, maille en bas à droite du noeud \a n
  Cell topZLowerRight(Node n) const { return _nodeToCell(n, P_TopZLowerRight); }
  //! En 3D, maille en bas à gauche du noeud \a n
  Cell topZLowerLeft(Node n) const { return _nodeToCell(n, P_TopZLowerLeft); }

  //! En 3D, maille en haut à gauche du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId topZUpperLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZUpperLeft); }
  //! En 3D, maille en haut à droite du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId topZUpperRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZUpperRight); }
  //! En 3D, maille en bas à droite du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId topZLowerRightId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZLowerRight); }
  //! En 3D, maille en bas à gauche du noeud \a n
  ARCCORE_HOST_DEVICE CellLocalId topZLowerLeftId(NodeLocalId n) const { return _nodeToCellLocalId(n, P_TopZLowerLeft); }

  //! En 3D, maille en haut à gauche du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZUpperLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZUpperLeft); }
  //! En 3D, maille en haut à droite du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZUpperRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZUpperRight); }
  //! En 3D, maille en bas à droite du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZLowerRightId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZLowerRight); }
  //! En 3D, maille en bas à gauche du noeud \a n pour la direction \a dir
  ARCCORE_HOST_DEVICE CellLocalId topZLowerLeftId(NodeLocalId n, Int32 dir) const { return _nodeToCellLocalId(n, dir, P_TopZLowerLeft); }

  //! Noeud en haut à gauche de la maille \a c
  Node upperLeft(Cell c) const { return _cellToNode(c, P_UpperLeft); }
  //! Noeud en haut à droite de la maille \a c
  Node upperRight(Cell c) const { return _cellToNode(c, P_UpperRight); }
  //! Noeud en bas à droite de la maille \a c
  Node lowerRight(Cell c) const { return _cellToNode(c, P_LowerRight); }
  //! Noeud en bad à gauche de la maille \a c
  Node lowerLeft(Cell c) const { return _cellToNode(c, P_LowerLeft); }

  //! Noeud en haut à gauche de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId upperLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_UpperLeft); }
  //! Noeud en haut à droite de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId upperRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_UpperRight); }
  //! Noeud en bas à droite de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId lowerRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_LowerRight); }
  //! Noeud en bad à gauche de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId lowerLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_LowerLeft); }

  //! Noeud en haut à gauche de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId upperLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_UpperLeft); }
  //! Noeud en haut à droite de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId upperRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_UpperRight); }
  //! Noeud en bas à droite de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId lowerRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_LowerRight); }
  //! Noeud en bad à gauche de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId lowerLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_LowerLeft); }

  //! En 3D, noeud au dessus en haut à gauche de la maille \a c
  Node topZUpperLeft(Cell c) const { return _cellToNode(c, P_TopZUpperLeft); }
  //! En 3D, noeud au dessus en haut à droite de la maille \a c
  Node topZUpperRight(Cell c) const { return _cellToNode(c, P_TopZUpperRight); }
  //! En 3D, noeud au dessus en bas à droite de la maille \a c
  Node topZLowerRight(Cell c) const { return _cellToNode(c, P_TopZLowerRight); }
  //! En 3D, noeud au dessus en bas à gauche de la maille \a c
  Node topZLowerLeft(Cell c) const { return _cellToNode(c, P_TopZLowerLeft); }

  //! En 3D, noeud au dessus en haut à gauche de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZUpperLeft); }
  //! En 3D, noeud au dessus en haut à droite de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZUpperRight); }
  //! En 3D, noeud au dessus en bas à droite de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerRightId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZLowerRight); }
  //! En 3D, noeud au dessus en bas à gauche de la maille \a c
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerLeftId(CellLocalId c) const { return _cellToNodeLocalId(c, P_TopZLowerLeft); }

  //! En 3D, noeud au dessus en haut à gauche de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZUpperLeft); }
  //! En 3D, noeud au dessus en haut à droite de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZUpperRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZUpperRight); }
  //! En 3D, noeud au dessus en bas à droite de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerRightId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZLowerRight); }
  //! En 3D, noeud au dessus en bas à gauche de la maille \a c pour la direction \a dir
  ARCCORE_HOST_DEVICE NodeLocalId topZLowerLeftId(CellLocalId c, Int32 dir) const { return _cellToNodeLocalId(c, dir, P_TopZLowerLeft); }

 private:

  //! Calcule les infos de connectivité.
  void _computeInfos(IMesh* mesh, VariableNodeReal3& nodes_coord, VariableCellReal3& cells_coord);
  void _computeInfos(ICartesianMesh* cmesh);
  //! Positionne les tableaux contenant les infos de connectivité
  void _setStorage(ArrayView<Index> nodes_to_cell, ArrayView<Index> cells_to_node,
                   const Permutation* permutation);

 private:

  ARCCORE_HOST_DEVICE CellLocalId _nodeToCellLocalId(NodeLocalId n, ePosition p) const
  {
    return CellLocalId(m_nodes_to_cell[n].v[p]);
  }
  ARCCORE_HOST_DEVICE NodeLocalId _cellToNodeLocalId(CellLocalId c, ePosition p) const
  {
    return NodeLocalId(m_cells_to_node[c].v[p]);
  }
  ARCCORE_HOST_DEVICE CellLocalId _nodeToCellLocalId(NodeLocalId n, Int32 dir, ePosition p) const
  {
    ARCCORE_CHECK_AT(dir, 3);
    return _nodeToCellLocalId(n, m_permutation->permutation[dir][p]);
  }
  ARCCORE_HOST_DEVICE NodeLocalId _cellToNodeLocalId(CellLocalId c, Int32 dir, ePosition p) const
  {
    ARCCORE_CHECK_AT(dir, 3);
    return _cellToNodeLocalId(c, m_permutation->permutation[dir][p]);
  }
  Cell _nodeToCell(Node n, ePosition p) const { return m_cells[m_nodes_to_cell[n.localId()].v[p]]; }
  Node _cellToNode(Cell c, ePosition p) const { return m_nodes[m_cells_to_node[c.localId()].v[p]]; }

 private:

  // Ces deux méthodes sont pour les tests
  Index& _index(Node n) { return m_nodes_to_cell[n.localId()]; }
  Index& _index(Cell c) { return m_cells_to_node[c.localId()]; }

 private:

  ArrayView<Index> m_nodes_to_cell;
  ArrayView<Index> m_cells_to_node;
  CellInfoListView m_cells;
  NodeInfoListView m_nodes;
  const Permutation* m_permutation = nullptr;

 private:

  void _computeInfos2D(IMesh* mesh, VariableNodeReal3& nodes_coord, VariableCellReal3& cells_coord);
  void _computeInfos2D(ICartesianMesh* cmesh);
  void _computeInfos3D(IMesh* mesh, VariableNodeReal3& nodes_coord, VariableCellReal3& cells_coord);
  void _computeInfos3D(ICartesianMesh* cmesh);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe d'accès aux connectivités cartésiennes.
 *
 * \sa CartesianConnectivity.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianConnectivityLocalId
: public CartesianConnectivity
{
 private:

  using Index = CartesianConnectivity::Index;

 public:

  CartesianConnectivityLocalId(const CartesianConnectivity& c)
  : CartesianConnectivity(c)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

