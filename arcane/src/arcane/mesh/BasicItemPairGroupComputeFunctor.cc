// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicItemPairGroupComputeFunctor.cc                         (C) 2000-2023 */
/*                                                                           */
/* Fonctions basiques de calcul des valeurs des ItemPairGroup.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/IMesh.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/ItemGroup.h"
#include "arcane/IItemFamily.h"

#include "arcane/mesh/BasicItemPairGroupComputeFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * TODO: Regarder pourquoi les méthodes de calcul suivantes:
 * _computeCellCellFaceAdjacency(ItemPairGroupImpl* array)
 * _computeFaceCellNodeAdjacency(ItemPairGroupImpl* array)
 * _computeCellFaceFaceAdjacency(ItemPairGroupImpl* array)
 * sont spéciales et les fusionner avec les autres.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicItemPairGroupComputeFunctor::
BasicItemPairGroupComputeFunctor(ITraceMng* tm)
: TraceAccessor(tm)
{
  _addComputeAdjacency(IK_Cell, IK_Cell, IK_Node,
                       &BasicItemPairGroupComputeFunctor::_computeCellCellNodeAdjacency);
  _addComputeAdjacency(IK_Cell, IK_Cell, IK_Face,
                       &BasicItemPairGroupComputeFunctor::_computeCellCellFaceAdjacency);
  _addComputeAdjacency(IK_Node, IK_Node, IK_Cell,
                       &BasicItemPairGroupComputeFunctor::_computeNodeNodeCellAdjacency);
  _addComputeAdjacency(IK_Face, IK_Cell, IK_Node,
                       &BasicItemPairGroupComputeFunctor::_computeFaceCellNodeAdjacency);
  _addComputeAdjacency(IK_Face, IK_Face, IK_Node,
                       &BasicItemPairGroupComputeFunctor::_computeFaceFaceNodeAdjacency);
  _addComputeAdjacency(IK_Cell, IK_Face, IK_Face,
                       &BasicItemPairGroupComputeFunctor::_computeCellFaceFaceAdjacency);
  _addComputeAdjacency(IK_Node, IK_Node, IK_Face,
                       &BasicItemPairGroupComputeFunctor::_computeNodeNodeFaceAdjacency);
  _addComputeAdjacency(IK_Node, IK_Node, IK_Edge,
                       &BasicItemPairGroupComputeFunctor::_computeNodeNodeEdgeAdjacency);
  _addComputeAdjacency(IK_Face, IK_Face, IK_Edge,
                       &BasicItemPairGroupComputeFunctor::_computeFaceFaceEdgeAdjacency);
  _addComputeAdjacency(IK_Face, IK_Face, IK_Cell,
                       &BasicItemPairGroupComputeFunctor::_computeFaceFaceCellAdjacency);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
computeAdjacency(ItemPairGroup adjency_array, eItemKind link_kind,
                 Integer nb_layer)
{
  if (nb_layer!=1)
    throw ArgumentException(A_FUNCINFO,"nb_layer should be 1");
  eItemKind item_kind = adjency_array.itemKind();
  eItemKind sub_item_kind = adjency_array.subItemKind();
  AdjacencyType atype(item_kind, sub_item_kind, link_kind);
  auto i = m_compute_adjacency_functions.find(atype);
  if (i == m_compute_adjacency_functions.end()) {
    String s = String::format("Invalid adjency computation item_kind={0} sub_item_kind={1} link_item_kind={2}",
                            item_kind,sub_item_kind,link_kind);
    throw NotImplementedException(A_FUNCINFO,s);
  }
  auto acf = new AdjacencyComputeFunctor(this, adjency_array.internal(), i->second);
  adjency_array.internal()->setComputeFunctor(acf);
  adjency_array.invalidate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_addComputeAdjacency(eItemKind ik, eItemKind sik, eItemKind lik, ComputeFunctor f)
{
  m_compute_adjacency_functions.insert(std::make_pair(AdjacencyType(ik, sik, lik), f));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeAdjacency(ItemPairGroupImpl* array,
                  GetItemVectorViewFunctor get_link_item_enumerator,
                  GetItemVectorViewFunctor get_sub_item_enumerator)
{
  const ItemGroup& group = array->itemGroup();
  const ItemGroup& sub_group = array->subItemGroup();
  auto& indexes = array->unguardedIndexes();
  indexes.clear();
  auto& sub_items_local_id = array->unguardedLocalIds();
  sub_items_local_id.clear();

  IItemFamily* sub_family = sub_group.itemFamily();
  const Integer max_sub_id = sub_family->maxLocalId();
  Int32UniqueArray items_list(max_sub_id); // indéxé par des sub-items

  IItemFamily* family = group.itemFamily();
  const Integer max_id = family->maxLocalId(); // index maxi des items

  const Integer forbidden_value = NULL_ITEM_ID;
  const Integer undef_value = max_id+1;
  items_list.fill(forbidden_value);
  
  ENUMERATE_ITEM(iitem,sub_group) {
    items_list[iitem.itemLocalId()] = undef_value;
  }

  indexes.add(0);
  ENUMERATE_ITEM(iitem,group){
    Item item = *iitem;
    Integer local_id = iitem.itemLocalId();
    // Pour ne pas s'ajouter à sa propre liste
    if (items_list[local_id] != forbidden_value)
      items_list[local_id] = local_id;
    for( Item linkitem : get_link_item_enumerator(item) ){
      for( Item subitem : get_sub_item_enumerator(linkitem) ){
        Int32 sub_local_id = subitem.localId();
        if (items_list[sub_local_id]==forbidden_value || items_list[sub_local_id]==local_id)
          continue;
        items_list[sub_local_id] = local_id;
        sub_items_local_id.add(sub_local_id);
      }
    }
    indexes.add(sub_items_local_id.largeSize());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeCellCellFaceAdjacency(ItemPairGroupImpl* array)
{
  const ItemGroup& group = array->itemGroup();
  const ItemGroup& sub_group = array->subItemGroup();
  auto& indexes = array->unguardedIndexes();
  indexes.clear();
  auto& sub_items_local_id = array->unguardedLocalIds();
  sub_items_local_id.clear();

  IItemFamily* sub_family = sub_group.itemFamily();
  const Integer max_sub_id = sub_family->maxLocalId();
  Int32UniqueArray items_list(max_sub_id); // indéxé par des sub-items

  IItemFamily* family = group.itemFamily();
  const Integer max_id = family->maxLocalId(); // index maxi des items

  const Integer forbidden_value = NULL_ITEM_ID;
  const Integer undef_value = max_id+1; 
  items_list.fill(forbidden_value);
  
  ENUMERATE_CELL(icell,sub_group) {
    items_list[icell.itemLocalId()] = undef_value;
  }

  indexes.add(0);
  ENUMERATE_CELL(icell,group){
    Cell cell = *icell;
    //Integer local_id = cell.localId();
    for( Face face : cell.faces() ){
      if (face.nbCell()!=2)
        continue;
      Cell back_cell = face.backCell();
      Cell front_cell = face.frontCell();
      
      // On choisit l'autre cellule du cote de la face
      const Integer sub_local_id = (back_cell==cell)?front_cell.localId():back_cell.localId();
      if (items_list[sub_local_id] == forbidden_value)
        continue;
      sub_items_local_id.add(sub_local_id);
    }
    indexes.add(sub_items_local_id.largeSize());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeFaceCellNodeAdjacency(ItemPairGroupImpl* array)
{
  const ItemGroup& group = array->itemGroup();
  const ItemGroup& sub_group = array->subItemGroup();
  auto& indexes = array->unguardedIndexes();
  indexes.clear();
  auto& sub_items_local_id = array->unguardedLocalIds();
  sub_items_local_id.clear();

  IItemFamily* sub_family = sub_group.itemFamily();
  const Integer max_sub_id = sub_family->maxLocalId();
  Int32UniqueArray items_list(max_sub_id); // indéxé par des sub-items

  IItemFamily* family = group.itemFamily();
  const Integer max_id = family->maxLocalId(); // index maxi des items

  const Integer forbidden_value = NULL_ITEM_ID;
  const Integer undef_value = max_id+1; 
  items_list.fill(forbidden_value);
  
  ENUMERATE_CELL(icell,sub_group) {
    items_list[icell.itemLocalId()] = undef_value;
  }

  indexes.add(0);
  ENUMERATE_FACE(iface,group){
    Face face = *iface;
    Int32 local_id = face.localId();
    for( Node node : face.nodes() ){
      for( Cell isubcell : node.cells() ){
        Int32 sub_local_id = isubcell.localId();
        if (items_list[sub_local_id]==forbidden_value || items_list[sub_local_id]==local_id)
          continue;
        items_list[sub_local_id] = local_id;
        sub_items_local_id.add(sub_local_id);
      }
    }
    indexes.add(sub_items_local_id.largeSize());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeCellFaceFaceAdjacency(ItemPairGroupImpl* array)
{
  const ItemGroup& group = array->itemGroup();
  const ItemGroup& sub_group = array->subItemGroup();
  auto& indexes = array->unguardedIndexes();
  indexes.clear();
  auto& sub_items_local_id = array->unguardedLocalIds();
  sub_items_local_id.clear();

  IItemFamily* sub_family = sub_group.itemFamily();
  const Integer max_sub_id = sub_family->maxLocalId();
  Int32UniqueArray items_list(max_sub_id); // indéxé par des sub-items

  IItemFamily* family = group.itemFamily();
  const Integer max_id = family->maxLocalId(); // index maxi des items

  const Integer forbidden_value = NULL_ITEM_ID;
  const Integer undef_value = max_id+1;
  items_list.fill(forbidden_value);
  
  ENUMERATE_FACE(iface,sub_group) {
    items_list[iface.itemLocalId()] = undef_value;
  }

  indexes.add(0);
  ENUMERATE_CELL(icell,group){
    Cell cell = *icell;
    for( FaceLocalId iface : cell.faceIds() ){
      Int32 sub_local_id = iface.localId();
      // Les controles sur les faces sont inutiles car on ne peut pas retomber dessus par l'énumération actuelle.
      if (items_list[sub_local_id]==forbidden_value /* or items_list[sub_local_id]==local_id */)
        continue;
      // items_list[sub_local_id] = local_id;
      sub_items_local_id.add(sub_local_id);
    }
    indexes.add(sub_items_local_id.largeSize());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeCellCellNodeAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item cell){ return cell.toCell().nodes(); };
  GetItemVectorViewFunctor y = [](Item node){ return node.toNode().cells(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeNodeNodeCellAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item node){ return node.toNode().cells(); };
  GetItemVectorViewFunctor y = [](Item cell){ return cell.toCell().nodes(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeFaceFaceNodeAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item face){ return face.toFace().nodes(); };
  GetItemVectorViewFunctor y = [](Item node){ return node.toNode().faces(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeNodeNodeFaceAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item node){ return node.toNode().faces(); };
  GetItemVectorViewFunctor y = [](Item face){ return face.toFace().nodes(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeNodeNodeEdgeAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item node){ return node.toNode().edges(); };
  GetItemVectorViewFunctor y = [](Item edge){ return edge.toEdge().nodes(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeFaceFaceEdgeAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item face){ return face.toFace().edges(); };
  GetItemVectorViewFunctor y = [](Item edge){ return edge.toEdge().faces(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicItemPairGroupComputeFunctor::
_computeFaceFaceCellAdjacency(ItemPairGroupImpl* array)
{
  GetItemVectorViewFunctor x = [](Item face){ return face.toFace().cells(); };
  GetItemVectorViewFunctor y = [](Item cell){ return cell.toCell().faces(); };

  return _computeAdjacency(array, x, y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
