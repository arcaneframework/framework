// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractItemFamilyTopologyModifier.h                        (C) 2000-2020 */
/*                                                                           */
/* Modification de la topologie des entités d'une famille.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ABSTRACTITEMFAMILYTOPOLOGYMODIFIER_H
#define ARCANE_MESH_ABSTRACTITEMFAMILYTOPOLOGYMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/IItemFamilyTopologyModifier.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Modification de la topologie des entités d'une famille.
 */
class ARCANE_MESH_EXPORT AbstractItemFamilyTopologyModifier
: public TraceAccessor
, public IItemFamilyTopologyModifier
{
 public:

  AbstractItemFamilyTopologyModifier(IItemFamily* afamily);
  virtual ~AbstractItemFamilyTopologyModifier() {} //<! Libère les ressources

 public:

  //! Famille associée
  IItemFamily* family() const override;

 public:

  void replaceNode(ItemLocalId item_lid,Integer index,ItemLocalId new_node_lid) override;
  void replaceEdge(ItemLocalId item_lid,Integer index,ItemLocalId new_edge_lid) override;
  void replaceFace(ItemLocalId item_lid,Integer index,ItemLocalId new_face_lid) override;
  void replaceCell(ItemLocalId item_lid,Integer index,ItemLocalId new_cell_lid) override;
  void replaceHParent(ItemLocalId item_lid,Integer index,ItemLocalId new_hparent_lid) override;
  void replaceHChild(ItemLocalId item_lid,Integer index,ItemLocalId new_hchild_lid) override;

  void findAndReplaceNode(ItemLocalId item_lid,ItemLocalId old_node_lid,
                          ItemLocalId new_node_lid) override;
  void findAndReplaceEdge(ItemLocalId item_lid,ItemLocalId old_edge_lid,
                          ItemLocalId new_edge_lid) override;
  void findAndReplaceFace(ItemLocalId item_lid,ItemLocalId old_face_lid,
                          ItemLocalId new_face_lid) override;
  void findAndReplaceCell(ItemLocalId item_lid,ItemLocalId old_cell_lid,
                          ItemLocalId new_cell_lid) override;

 private:

  IItemFamily* m_family;

  void _throwNotSupported();
  inline Int32 _getItemIndex(const Int32* items,Integer nb_item,Int32 local_id);
  inline Int32 _getItemIndex(ItemVectorView items,Int32 local_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
