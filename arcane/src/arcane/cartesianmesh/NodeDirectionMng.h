// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeDirectionMng.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Infos sur les noeuds d'une direction X Y ou Z d'un maillage structuré.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_NODEDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_NODEDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/VariableTypedef.h"
#include "arcane/IndexedItemConnectivityView.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Noeud avant et après un noeud suivant une direction.
 *
 * Les instances de cette classe sont temporaires et construites via
 * NodeDirectionMng::node().
 */
class ARCANE_CARTESIANMESH_EXPORT DirNode
{
  friend NodeDirectionMng;
  friend class DirNodeLocalId;

 private:

  typedef signed char IndexType;
  static constexpr IndexType NULL_CELL = -1;

  struct DirNodeCellIndex
  {
   public:

    ARCCORE_HOST_DEVICE IndexType operator[](Int32 i) const
    {
      ARCANE_CHECK_AT(i, 8);
      return m_indexes[i];
    }

   public:

    IndexType m_indexes[8];
  };

 private:

  // Seul NodeDirectionMng à le droit de construire un DirNode.
  DirNode(Node current, Node next, Node prev, DirNodeCellIndex idx)
  : m_current(current)
  , m_previous(prev)
  , m_next(next)
  , m_cell_index(idx)
  {}

 public:

  //! Maille avant
  Node previous() const { return m_previous; }
  //! Maille avant
  NodeLocalId previousId() const { return m_previous.itemLocalId(); }
  //! Maille après
  Node next() const { return m_next; }
  //! Maille après
  NodeLocalId nextId() const { return m_next.itemLocalId(); }
  /*!
   * \brief Indice dans la liste des mailles de ce noeud d'une
   * maille en fonction de sa position.
   *
   * Les valeurs possibles pour \a position sont données par l'énumération
   * eCellNodePosition.
   */
  Int32 cellIndex(Int32 position) const { return m_cell_index[position]; }
  /*!
   * \brief Indice local d'une maille en fonction de sa position par rapport à ce noeud.
   *
   * Les valeurs possibles pour \a position sont données par l'énumération
   * eCellNodePosition.
   */
  CellLocalId cellId(Int32 position) const
  {
    Int32 x = cellIndex(position);
    return (x == NULL_CELL) ? CellLocalId(NULL_ITEM_LOCAL_ID) : CellLocalId(m_current.cellId(x));
  }
  /*!
   * \brief Maille en fonction de sa position par rapport à ce noeud.
   *
   * Les valeurs possibles pour \a position sont données par l'énumération
   * eCellNodePosition.
   */
  Cell cell(Int32 position) const
  {
    Int32 x = cellIndex(position);
    return (x == NULL_CELL) ? Cell() : Cell(m_current.cell(x));
  }

  //! Noeud devant à gauche dans la direction
  Cell nextLeftCell() const { return cell(CNP_NextLeft); }
  //! Noeud devant à droite dans la direction
  Cell nextRightCell() const { return cell(CNP_NextRight); }
  //! Noeud derrière à droite dans la direction
  Cell previousRightCell() const { return cell(CNP_PreviousRight); }
  //! Noeud derrière à gauche dans la direction
  Cell previousLeftCell() const { return cell(CNP_PreviousLeft); }

  //! Noeud devant à gauche dans la direction
  CellLocalId nextLeftCellId() const { return cellId(CNP_NextLeft); }
  //! Noeud devant à droite dans la direction
  CellLocalId nextRightCellId() const { return cellId(CNP_NextRight); }
  //! Noeud derrière à droite dans la direction
  CellLocalId previousRightCellId() const { return cellId(CNP_PreviousRight); }
  //! Noeud derrière à gauche dans la direction
  CellLocalId previousLeftCellId() const { return cellId(CNP_PreviousLeft); }

  //! Noeud devant à gauche dans la direction
  Cell topNextLeftCell() const { return cell(CNP_TopNextLeft); }
  //! Noeud devant à droite dans la direction
  Cell topNextRightCell() const { return cell(CNP_TopNextRight); }
  //! Noeud derrière à droite dans la direction
  Cell topPreviousRightCell() const { return cell(CNP_TopPreviousRight); }
  //! Noeud derrière à gauche dans la direction
  Cell topPreviousLeftCell() const { return cell(CNP_TopPreviousLeft); }

  //! Noeud devant à gauche dans la direction
  CellLocalId topNextLeftCellId() const { return cellId(CNP_TopNextLeft); }
  //! Noeud devant à droite dans la direction
  CellLocalId topNextRightCellId() const { return cellId(CNP_TopNextRight); }
  //! Noeud derrière à droite dans la direction
  CellLocalId topPreviousRightCellId() const { return cellId(CNP_TopPreviousRight); }
  //! Noeud derrière à gauche dans la direction
  CellLocalId topPreviousLeftCellId() const { return cellId(CNP_TopPreviousLeft); }

 private:

  Node m_current;
  Node m_previous;
  Node m_next;
  DirNodeCellIndex m_cell_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Noeud avant et après un noeud suivant une direction.
 *
 * Les instances de cette classe sont temporaires et construites via
 * NodeDirectionMng::dirNodeId().
 */
class ARCANE_CARTESIANMESH_EXPORT DirNodeLocalId
{
  friend NodeDirectionMng;

 private:

  typedef signed char IndexType;
  static constexpr IndexType NULL_CELL = -1;

 private:

  // Seul NodeDirectionMng à le droit de construire un DirNode.
  ARCCORE_HOST_DEVICE DirNodeLocalId(NodeLocalId current, NodeLocalId next, NodeLocalId prev,
                                     DirNode::DirNodeCellIndex idx,
                                     IndexedNodeCellConnectivityView view)
  : m_current(current)
  , m_previous(prev)
  , m_next(next)
  , m_cell_index(idx)
  , m_view(view)
  {}

 public:

  //! Maille avant
  ARCCORE_HOST_DEVICE NodeLocalId previous() const { return m_previous; }
  //! Maille avant
  ARCCORE_HOST_DEVICE NodeLocalId previousId() const { return m_previous; }
  //! Maille après
  ARCCORE_HOST_DEVICE NodeLocalId next() const { return m_next; }
  //! Maille après
  ARCCORE_HOST_DEVICE NodeLocalId nextId() const { return m_next; }
  /*!
   * \brief Indice dans la liste des mailles de ce noeud d'une
   * maille en fonction de sa position.
   *
   * Les valeurs possibles pour \a position sont données par l'énumération
   * eCellNodePosition.
   */
  ARCCORE_HOST_DEVICE Int32 cellIndex(Int32 position) const { return m_cell_index[position]; }
  /*!
   * \brief Indice local d'une maille en fonction de sa position par rapport à ce noeud.
   *
   * Les valeurs possibles pour \a position sont données par l'énumération
   * eCellNodePosition.
   */
  ARCCORE_HOST_DEVICE CellLocalId cellId(Int32 position) const
  {
    Int32 x = cellIndex(position);
    return (x == NULL_CELL) ? CellLocalId(NULL_ITEM_LOCAL_ID) : m_view.cellId(m_current, x);
  }

  //! Noeud devant à gauche dans la direction
  ARCCORE_HOST_DEVICE CellLocalId nextLeftCellId() const { return cellId(CNP_NextLeft); }
  //! Noeud devant à droite dans la direction
  ARCCORE_HOST_DEVICE CellLocalId nextRightCellId() const { return cellId(CNP_NextRight); }
  //! Noeud derrière à droite dans la direction
  ARCCORE_HOST_DEVICE CellLocalId previousRightCellId() const { return cellId(CNP_PreviousRight); }
  //! Noeud derrière à gauche dans la direction
  ARCCORE_HOST_DEVICE CellLocalId previousLeftCellId() const { return cellId(CNP_PreviousLeft); }

  //! Noeud devant à gauche dans la direction
  ARCCORE_HOST_DEVICE CellLocalId topNextLeftCellId() const { return cellId(CNP_TopNextLeft); }
  //! Noeud devant à droite dans la direction
  ARCCORE_HOST_DEVICE CellLocalId topNextRightCellId() const { return cellId(CNP_TopNextRight); }
  //! Noeud derrière à droite dans la direction
  ARCCORE_HOST_DEVICE CellLocalId topPreviousRightCellId() const { return cellId(CNP_TopPreviousRight); }
  //! Noeud derrière à gauche dans la direction
  ARCCORE_HOST_DEVICE CellLocalId topPreviousLeftCellId() const { return cellId(CNP_TopPreviousLeft); }

 private:

  NodeLocalId m_current;
  NodeLocalId m_previous;
  NodeLocalId m_next;
  DirNode::DirNodeCellIndex m_cell_index;
  IndexedNodeCellConnectivityView m_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Infos sur les noeuds d'une direction spécifique X,Y ou Z
 * d'un maillage structuré.
 */
class ARCANE_CARTESIANMESH_EXPORT NodeDirectionMng
{
  friend CartesianMeshImpl;
  friend CartesianMeshPatch;
  class Impl;
  using IndexType = DirNode::IndexType;
  using DirNodeCellIndex = DirNode::DirNodeCellIndex;

 private:

  struct ItemDirectionInfo
  {
   public:

    /*!
     * \brief Constructeur par défaut.
     * \warning Les valeurs m_next_item et m_previous_item sont initialisées
     * à nullptr.
     */
    ItemDirectionInfo() = default;
    ItemDirectionInfo(Int32 next_lid, Int32 prev_lid)
    : m_next_lid(next_lid)
    , m_previous_lid(prev_lid)
    {}

   public:

    //! entité après l'entité courante dans la direction
    NodeLocalId m_next_lid;
    //! entité avant l'entité courante dans la direction
    NodeLocalId m_previous_lid;

   public:

    void setCellIndexes(IndexType idx[8])
    {
      for (int i = 0; i < 8; ++i)
        m_cell_index.m_indexes[i] = idx[i];
    }
    DirNodeCellIndex m_cell_index;
  };

 public:

  /*!
   * \brief Créé une instance vide.
   *
   * L'instance n'est pas valide tant que _internalInit() n'a pas été appelé.
   */
  NodeDirectionMng();

 public:

  //! Noeud direction correspondant au noeud \a n
  DirNode node(Node n) const
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant au noeud \a n
  DirNode node(NodeLocalId n) const
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant au noeud \a n
  DirNode dirNode(NodeLocalId n) const
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant au noeud \a n
  ARCCORE_HOST_DEVICE DirNodeLocalId dirNodeId(NodeLocalId n) const
  {
    return _dirNodeId(n);
  }

  //! Groupe de tous les noeuds dans la direction.
  NodeGroup allNodes() const;

  /*!
   * \brief Groupe de tous les noeuds de recouvrement dans la direction.
   *
   * Un noeud de recouvrement est un noeud qui possède uniquement des mailles
   * de recouvrement autour de lui.
   *
   *   0   1  2  3  4
   * ┌───┬──┬──┬──┬──┐
   * │   │  │  │  │  │
   * │   ├──┼──┼──┼──┤
   * │   │  │  │  │  │
   * └───┴──┴──┴──┴──┘
   *
   * 0 : level -1
   * 1 et 2 : Mailles de recouvrements (overallCells)
   * 3 : Mailles externes (outerCells)
   * 4 : Mailles internes (innerCells)
   *
   * La couche de mailles de recouvrements désigne la couche de mailles de même
   * niveau autour du patch. Ces mailles peuvent appartenir à un ou plusieurs
   * patchs.
   */
  NodeGroup overallNodes() const;

  /*!
   * \brief Groupe de tous les noeuds du patch dans la direction.
   *
   * Les noeuds du patch sont les noeuds n'ayant pas toutes ces mailles qui
   * soient de recouvrement. TODO reformuler
   * (`innerNodes() + outerNodes()` ou simplement `!overallNodes()`)
   *
   * \warning Les noeuds au bord du domaine (donc ayant que des mailles
   * "outer") sont inclus dans ce groupe.
   */
  NodeGroup inPatchNodes() const;

  /*!
   * \brief Groupe de tous les noeuds internes dans la direction.
   *
   * Un noeud est considéré comme interne si son noeud
   * avant ou après n'est pas nul.
   */
  NodeGroup innerNodes() const;

  /*!
   * \brief Groupe de tous les noeuds externes dans la direction.
   *
   * Un noeud est considéré comme externe si son noeud
   * avant ou après est de recouvrement ou est nul (si l'on est au bord du
   * domaine ou s'il n'y a pas de couches de mailles de recouvrements).
   *
   * \note Les noeuds entre patchs ne sont pas dupliquées. Donc certains noeuds
   * de ce groupe peuvent être aussi dans un outerNodes() d'un autre patch.
   */
  NodeGroup outerNodes() const;

  //! Noeud direction correspondant au noeud \a n.
  DirNode operator[](Node n)
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant au noeud \a n.
  DirNode operator[](NodeLocalId n) const
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant à l'itérateur du noeud \a inode.
  DirNode operator[](NodeEnumerator inode) const
  {
    return _node(inode.itemLocalId());
  }

  //! Valeur de la direction
  eMeshDirection direction() const
  {
    return m_direction;
  }

 protected:

  /*!
   * \internal
   * \brief Calcule les informations sur les noeuds associées aux mailles de
   * la direction \a cell_dm. \a all_nodes est le groupe de tous les noeuds des mailles
   * gérées par \a cell_dm.
   * Suppose que init() a été appelé.
   */
  void _internalComputeInfos(const CellDirectionMng& cell_dm, const NodeGroup& all_nodes,
                             const VariableCellReal3& cells_center);

  void _internalComputeInfos(const CellDirectionMng& cell_dm, const NodeGroup& all_nodes);

  /*!
   * \internal
   * Initialise l'instance.
   */
  void _internalInit(ICartesianMesh* cm, eMeshDirection dir, Integer patch_index);

  /*!
   * \internal
   * Détruit les ressources associées à l'instance.
   */
  void _internalDestroy();

  /*!
   * \brief Redimensionne le conteneur contenant les \a ItemDirectionInfo.
   *
   * Cela invalide les instances courantes de NodeDirectionMng.
   */
  void _internalResizeInfos(Int32 new_size);

 private:

  SmallSpan<ItemDirectionInfo> m_infos_view;
  NodeInfoListView m_nodes;
  eMeshDirection m_direction;
  IndexedNodeCellConnectivityView m_node_cell_view;
  Impl* m_p;

 private:

  //! Noeud direction correspondant au noeud de numéro local \a local_id
  DirNode _node(Int32 local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirNode(m_nodes[local_id], m_nodes[d.m_next_lid], m_nodes[d.m_previous_lid], d.m_cell_index);
  }

  //! Noeud direction correspondant au noeud de numéro local \a local_id
  ARCCORE_HOST_DEVICE DirNodeLocalId _dirNodeId(NodeLocalId local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id.localId()];
    return DirNodeLocalId(local_id, d.m_next_lid, d.m_previous_lid, d.m_cell_index, m_node_cell_view);
  }

  void _computeNodeCellInfos(const CellDirectionMng& cell_dm,
                             const VariableCellReal3& cells_center);
  void _computeNodeCellInfos() const;
  void _filterNodes();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

