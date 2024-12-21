// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivity.cc                                         (C) 2000-2024 */
/*                                                                           */
/* External connectivities. First version with DoF                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/ExtraGhostItemsManager.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivity::
compute()
{
  ItemVectorView from_items = _sourceFamily()->allItems().own().view();
  ItemVectorView to_items = _targetFamily()->allItems().own().view();
  ARCANE_ASSERT((from_items.size() == to_items.size()),
                ("Connected families must have the same number of items "))
  m_item_property.resize(_sourceFamily(),NULL_ITEM_LOCAL_ID);
  // Link from and to items
  Integer i = 0;
  ENUMERATE_ITEM(iitem,from_items){
    m_item_property[iitem] = to_items[i++].localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemArrayConnectivity::
compute()
{
  ItemVectorView from_items = _sourceFamily()->allItems().own().view();
  ItemVectorView to_items = _targetFamily()->allItems().own().view();
  ARCANE_ASSERT((from_items.size()*m_nb_dof_per_item == to_items.size()),
                ("Incorrect connected family size. Should be FromFamily.own.size * nb_element_per_item"))
  m_item_property.resize(_sourceFamily(),m_nb_dof_per_item,NULL_ITEM_LOCAL_ID);
  Integer i = 0;
  ENUMERATE_ITEM(iitem,from_items){
    for (Integer j = 0; j < m_nb_dof_per_item ; ++j)
      m_item_property[iitem][j] = to_items[i++].localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemMultiArrayConnectivity::
compute(IntegerConstArrayView nb_dof_per_item)
{
  ItemVectorView from_items = _sourceFamily()->allItems().own().view();
  DoFVectorView to_items = _targetFamily()->allItems().own().view();
  Integer total_nb_dof_per_item = 0;
  for (Integer i = 0; i < nb_dof_per_item.size(); ++i)
    total_nb_dof_per_item += nb_dof_per_item[i];

  ARCANE_ASSERT((total_nb_dof_per_item == to_items.size()),
                ("Incorrect connected family size. Should be equal to the sum of array nb_element_per_item elements"))

  m_item_property.resize(_sourceFamily(),nb_dof_per_item,NULL_ITEM_LOCAL_ID);

  Integer i = 0;
  ENUMERATE_ITEM(iitem,from_items){
    for (Integer j = 0; j < nb_dof_per_item[iitem->localId()]; ++j)
      m_item_property[iitem][j] = to_items[i++].localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivity::
updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items)
{
  ARCANE_ASSERT((from_items.size() == to_items.size()),("from_items and to_items arrays must have the same size to update connectivity"))
  // Adapt to possible evolution of from family size
  m_item_property.resize(_sourceFamily(),NULL_ITEM_LOCAL_ID);
  ItemVectorView from_items_view = _sourceFamily()->view(from_items);
  for (Integer i = 0; i < from_items_view.size(); ++i)
    m_item_property[from_items_view[i]]  = to_items[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemArrayConnectivity::
updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items)
{
  ARCANE_ASSERT((from_items.size() == to_items.size()),("from_items and to_items arrays must have the same size to update connectivity"))
  // Adapt to possible evolution of from family size
  m_item_property.resize(_sourceFamily(),m_nb_dof_per_item, NULL_ITEM_LOCAL_ID);
  IntegerSharedArray to_items_index(_sourceFamily()->maxLocalId(),0);// index in the connexion : from 0 to nb_dof_per_item -1. Fill with 0
  std::set<Int32> from_items_set;
  ItemVectorView from_items_view = _sourceFamily()->view(from_items);
  for (Integer i = 0; i < from_items.size(); ++i){
    if (! from_items_set.insert(from_items[i]).second)
      ++to_items_index[from_items[i]]; // update the index in the connexion
    m_item_property[from_items_view[i]][to_items_index[from_items[i]]] = to_items[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemMultiArrayConnectivity::
updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items)
{
  ARCANE_ASSERT((from_items.size() == to_items.size()),
                ("from_items and to_items arrays must have the same size to update connectivity"))
  // Resize item property
  IntegerSharedArray nb_connected_element_per_item(m_item_property.dim2Sizes()); // Array indexed by lids

  // Adapt to possible evolution of from family size
  nb_connected_element_per_item.resize(_sourceFamily()->maxLocalId(),0);

  // Remove history
  for (Integer i = 0; i < from_items.size(); ++i)
    nb_connected_element_per_item[from_items[i]] = 0;
  for (Integer i = 0; i < from_items.size(); ++i)
    ++nb_connected_element_per_item[from_items[i]];
  m_item_property.resize(_sourceFamily(),nb_connected_element_per_item,NULL_ITEM_LOCAL_ID);

  // Update item property
  IntegerSharedArray to_items_index(_sourceFamily()->maxLocalId(),0);// index in the connexion : from 0 to nb_dof_per_item -1. Fill with 0
  std::set<Int32> from_items_set;
  ItemVectorView from_items_view = _sourceFamily()->view(from_items);
  for (Integer i = 0; i < from_items.size(); ++i){
    if (! from_items_set.insert(from_items[i]).second)
      ++to_items_index[from_items[i]]; // update the index in the connexion
    m_item_property[from_items_view[i]][to_items_index[from_items[i]]] = to_items[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivity::
notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids)
{
  m_item_property.resize(_sourceFamily(), NULL_ITEM_LOCAL_ID);
  ENUMERATE_ITEM(item,_sourceFamily()->allItems())
  {
    if (m_item_property[item] != NULL_ITEM_LOCAL_ID)
      m_item_property[item] = old_to_new_ids[m_item_property[item]];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemArrayConnectivity::
notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids)
{
  ENUMERATE_ITEM(item,_sourceFamily()->allItems())
  {
    for (Integer i = 0; i < m_nb_dof_per_item; ++i)
      {
        if (m_item_property[item][i] != NULL_ITEM_LOCAL_ID)
          m_item_property[item][i] = old_to_new_ids[m_item_property[item][i]];
      }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemMultiArrayConnectivity::
notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids)
{
  ENUMERATE_ITEM(item,_sourceFamily()->allItems())
  {
    for (Integer i = 0; i < m_item_property.dim2Sizes()[item.localId()]; ++i)
      {
        if (m_item_property[item][i] != NULL_ITEM_LOCAL_ID)
          m_item_property[item][i] = old_to_new_ids[m_item_property[item][i]];
      }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
