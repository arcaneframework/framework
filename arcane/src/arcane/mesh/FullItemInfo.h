// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FullItemInfo.h                                              (C) 2000-2016 */
/*                                                                           */
/* Information de sérialisation d'une maille.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_FULLCELLINFO_H
#define ARCANE_MESH_FULLCELLINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"

#include "arcane/ItemTypeMng.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;
class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour créer une maille connaissant les uniqueId() et owner()
 * de tous ces noeuds et faces.
 *
 */
class FullCellInfo
{
 public:  
  enum
    { // parent info
      PI_Node = 1 << 1,
      PI_Edge = 1 << 2,
      PI_Face = 1 << 3,
      PI_Cell = 1 << 4
    };
  
 public:
  /*!
   * \brief Référence les infos d'une maille.
   *
   * Le tableau cells_infos doit rester valide tant que l'instance existe.
   */
  FullCellInfo(Int64ConstArrayView cells_infos, Integer cell_index,
               ItemTypeMng* itm, Integer parent_info, 
               bool has_edge, bool has_amr, bool with_flags=false);
  
 public:
  
  ItemTypeInfo* typeInfo() const { return m_type; }
  Integer typeId() const { return CheckedConvert::toInteger(m_infos[0]); }
  Int64 uniqueId() const { return m_infos[1]; }
  Integer owner() const { return CheckedConvert::toInteger(m_infos[2]); }
  Integer nbNode() const { return m_nb_node; }
  Integer nbEdge() const { return m_nb_edge; }
  Integer nbFace() const { return m_nb_face; }
  Int64 faceUniqueId(Integer index) const { return m_infos[m_first_face + (index*2)]; }
  Int32 faceOwner(Integer index) const { return CheckedConvert::toInt32(m_infos[m_first_face + (index*2) + 1]); }
  Int64 edgeUniqueId(Integer index) const { return m_infos[m_first_edge + (index*2)]; }
  Int32 edgeOwner(Integer index) const { return CheckedConvert::toInt32(m_infos[m_first_edge + (index*2) + 1]); }
  Int64 nodeUniqueId(Integer index) const { return m_infos[3 + (index*2)]; }
  Int32 nodeOwner(Integer index) const { return CheckedConvert::toInt32(m_infos[3 + (index*2) + 1]); }
  Integer memoryUsed() const { return m_memory_used; }
  bool hasParentNode() const { return (m_parent_info & PI_Node); }
  Int64 parentNodeUniqueId(Integer index) const { return m_infos[m_first_parent_node + index]; }
  bool hasParentEdge() const { return (m_parent_info & PI_Edge); }
  Int64 parentEdgeUniqueId(Integer index) const { return m_infos[m_first_parent_edge + (index*2)]; }
  Int64 parentEdgeTypeId(Integer index) const { return m_infos[m_first_parent_edge + (index*2) + 1]; }
  bool hasParentFace() const { return (m_parent_info & PI_Face); }
  Int64 parentFaceUniqueId(Integer index) const { return m_infos[m_first_parent_face + (index*2)]; }
  Int64 parentFaceTypeId(Integer index) const { return m_infos[m_first_parent_face + (index*2) + 1]; }
  bool hasParentCell() const { return (m_parent_info & PI_Cell); }
  Int64 parentCellUniqueId() const { return m_infos[m_first_parent_cell]; }
  Int64 parentCellTypeId() const { return m_infos[m_first_parent_cell + 1]; }
  //! AMR
  Integer level() const { return CheckedConvert::toInteger(m_infos[m_first_hParent_cell]); }
  Int64 hParentCellUniqueId() const { return m_infos[m_first_hParent_cell + 1]; }
  Integer whichChildAmI () const { return CheckedConvert::toInteger(m_infos[m_first_hParent_cell + 2]); }
  Int32 flags() const { return CheckedConvert::toInt32(m_with_flags?m_infos[m_first_hParent_cell + 3]:0) ; }

 public:
  
  void print(ostream& o) const;

 public:

  //! Taille memoire en Int64 pour représenter une cellule de type it
  /*! \a parent_info décrit quelle relation de parenté doit y être comptée */
  ARCCORE_DEPRECATED_2020("Use dump() overload with buffer")
  static Integer memoryUsed(ItemTypeInfo* it, Integer parent_info, bool has_edge, bool has_amr,bool with_flags);
  ARCCORE_DEPRECATED_2020("Use dump() overload with buffer")
  static void dump(ItemInternal* cell, ISerializer* buf, Integer parent_info, bool has_edge, bool has_amr,bool with_flags);
  static void dump(ItemInternal* cell, Array<Int64>& buf, Integer parent_info, bool has_edge, bool has_amr,bool with_flags);
  static Integer parentInfo(IMesh* mesh);

 protected:

  void _setInternalInfos();

 public:

  Int64ConstArrayView m_infos;
 private:
  Integer m_nb_node;
  Integer m_nb_edge;
  Integer m_nb_face;
 public:
  Integer m_first_edge;
  Integer m_first_face;
  Integer m_memory_used;
  ItemTypeInfo* m_type;
  Integer m_first_parent_node;
  Integer m_first_parent_edge;
  Integer m_first_parent_face;
  Integer m_first_parent_cell;
  Integer m_parent_info;
  bool m_has_edge;
  //! AMR
  Integer m_first_hParent_cell;
  bool m_has_amr;
  bool m_with_flags;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ostream&
operator<<(ostream& o,const FullCellInfo& i)
{
  i.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
 
/*!
 * \brief Infos pour créer une liaison connaissant les uniqueId() et owner()
 * de tous ces noeuds duaux
 *
 */
class FullLinkInfo
{
public:
  /*! \brief Référence les infos d'une liaison.
   * Le tableau links_infos doit rester valide tant que l'instance existe.
   */
  FullLinkInfo(Int64ConstArrayView links_infos,Integer link_index,ItemTypeMng* itm)
    : m_type_id(-1)
    , m_nb_dual_node(0)
    , m_memory_used(0)
    , m_type(NULL)
  {
    m_type_id      = (Integer)links_infos[link_index    ];
    m_nb_dual_node = (Integer)links_infos[link_index + 3];
    ItemTypeInfo* it = itm->typeFromId(m_type_id);
    m_type = it;
    m_memory_used = 4 + m_nb_dual_node;
    m_infos = Int64ConstArrayView(m_memory_used,&links_infos[link_index]);
  }
  
public:
  
  ItemTypeInfo* typeInfo() const { return m_type; }
  Integer typeId() const { return (Integer)m_infos[0]; }
  Int64 uniqueId() const { return m_infos[1]; }
  Integer owner() const { return (Integer)m_infos[2]; }
  Integer nbDualNode() const { return m_nb_dual_node; }
  Int64 dualNodeUniqueId(Integer index) const { return m_infos[4 + index]; }
  Integer memoryUsed() const { return m_memory_used; }

public:

  void print(ostream& o) const;
  
public:
  
  Int64ConstArrayView m_infos;
  Integer m_type_id;
  Integer m_nb_dual_node;
  Integer m_memory_used;
  ItemTypeInfo* m_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ostream&
operator<<(ostream& o,const FullLinkInfo& i)
{
  i.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Infos pour créer un noeud dual connaissant les uniqueId() et owner()
 * de son dual.
 *
 */
class FullDualNodeInfo
{
public:
  /*! \brief Référence les infos d'un noeud dual.
   * Le tableau dual_nodes_infos doit rester valide tant que l'instance existe.
   */
  FullDualNodeInfo(Int64ConstArrayView dual_nodes_infos,
                   Integer dual_node_index,
                   ItemTypeMng* itm)
    : m_memory_used(4)
    , m_type(0)
  {
    Integer item_type_id = (Integer)dual_nodes_infos[dual_node_index];
    ItemTypeInfo* it = itm->typeFromId(item_type_id);
    m_type = it;
    m_infos = Int64ConstArrayView(m_memory_used,&dual_nodes_infos[dual_node_index]);
  }
  
public:
  
  ItemTypeInfo* typeInfo() const { return m_type; }
  Integer typeId() const { return (Integer)m_infos[0]; }
  Int64 uniqueId() const { return m_infos[1]; }
  Int64 dualItemUniqueId() const { return m_infos[2]; }
  Integer owner() const { return (Integer)m_infos[3]; }
  Integer memoryUsed() const { return m_memory_used; }

public:

  void print(ostream& o) const {
    o<<"DualNode : uid="<<uniqueId()<<" dual item uid="<<dualItemUniqueId()<<" owner="<<owner();
  }
  
public:
  
  Int64ConstArrayView m_infos;
  Integer m_memory_used;
  ItemTypeInfo* m_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ostream&
operator<<(ostream& o,const FullDualNodeInfo& i)
{
  i.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_FULLCELLINFO_H */
