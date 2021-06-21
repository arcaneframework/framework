// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeDirectionMng.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Infos sur les noeuds d'une direction X Y ou Z d'un maillage structuré.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CEA_NODEDIRECTIONMNG_H
#define ARCANE_CEA_NODEDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/cea/CeaGlobal.h"

#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/VariableTypedef.h"

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
class ARCANE_CEA_EXPORT DirNode
{
  friend NodeDirectionMng;
 private:
  typedef signed char IndexType;
  static constexpr IndexType NULL_CELL = -1;
  struct DirNodeCellIndex
  {
   public:
    IndexType operator[](Int32 i) const
    {
      ARCANE_CHECK_AT(i,8);
      return m_indexes[i];
    }
    IndexType& operator[](Int32 i)
    {
      ARCANE_CHECK_AT(i,8);
      return m_indexes[i];
    }
   public:
    IndexType m_indexes[8];
  };
 private:
  // Seul NodeDirectionMng à le droit de construire un DirNode.
  DirNode(Node n,Node p,DirNodeCellIndex idx)
  : m_previous(p), m_next(n), m_cell_index(idx) {}
 public:
  //! Maille avant
  Node previous() const { return m_previous; }
  //! Maille avant
  NodeLocalId previousId() const { return NodeLocalId(m_previous.localId()); }
  //! Maille après
  Node next() const { return m_next; }
  //! Maille après
  NodeLocalId nextId() const { return NodeLocalId(m_next.localId()); }
  Int32 cellIndex(Int32 index) const { return m_cell_index[index]; }
 private:
  Node m_previous;
  Node m_next;
  DirNodeCellIndex m_cell_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Infos sur les noeuds d'une direction spécifique X,Y ou Z
 * d'un maillage structuré.
 */
class ARCANE_CEA_EXPORT NodeDirectionMng
{
  friend CartesianMesh;
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
    ItemDirectionInfo()
    : m_next_item(nullptr), m_previous_item(nullptr){}
    ItemDirectionInfo(ItemInternal* next,ItemInternal* prev)
    : m_next_item(next), m_previous_item(prev){}
   public:
    //! entité après l'entité courante dans la direction
    ItemInternal* m_next_item;
    //! entité avant l'entité courante dans la direction
    ItemInternal* m_previous_item;
   public:
    void setCellIndexes(IndexType idx[8])
    {
      for( int i=0; i<8; ++i )
        m_cell_index[i] = idx[i];
    }
    DirNodeCellIndex m_cell_index;
  };
 public:
  
  //! Créé une instance vide. L'instance n'est pas valide tant que init() n'a pas été appelé.
  NodeDirectionMng();
  NodeDirectionMng(const NodeDirectionMng& rhs);
  ~NodeDirectionMng();

  //! Noeud direction correspondant au noeud \a n
  DirNode node(Node n)
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant au noeud \a n
  DirNode node(NodeLocalId n)
  {
    return _node(n.localId());
  }

  //! Groupe de tous les noeuds dans la direction.
  NodeGroup allNodes() const;

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
   * avant ou après est nul.
   */
  NodeGroup outerNodes() const;

  //! Noeud direction correspondant au noeud \a n.
  DirNode operator[](Node n)
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant au noeud \a n.
  DirNode operator[](NodeLocalId n)
  {
    return _node(n.localId());
  }

  //! Noeud direction correspondant à l'itérateur du noeud \a inode.
  DirNode operator[](NodeEnumerator inode)
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
  void _internalComputeInfos(const CellDirectionMng& cell_dm,const NodeGroup& all_nodes,
                             const VariableCellReal3& cells_center);

  /*!
   * \internal
   * Initialise l'instance.
   */
  void _internalInit(ICartesianMesh* cm,eMeshDirection dir,Integer patch_index);

  /*!
   * \internal
   * Détruit les ressources associées à l'instance.
   */
  void _internalDestroy();

 private:

  SharedArray<ItemDirectionInfo> m_infos;
  eMeshDirection m_direction;
  Impl* m_p;

 private:
  
  //! Noeud direction correspondant au noeud de numéro local \a local_id
  DirNode _node(Int32 local_id)
  {
    Node next = Node(m_infos[local_id].m_next_item);
    Node prev = Node(m_infos[local_id].m_previous_item);
    return DirNode(next,prev,m_infos[local_id].m_cell_index);
  }

  void _computeNodeCellInfos(const CellDirectionMng& cell_dm,
                             const VariableCellReal3& cells_center);
  void _filterNodes();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

