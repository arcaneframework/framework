// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityContainerView.cc                            (C) 2000-2022 */
/*                                                                           */
/* Vues sur les conteneurs contenant les connectivités.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemConnectivityContainerView.h"

#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityContainerView::
checkSame(ItemConnectivityContainerView rhs) const
{
  auto current_list = m_list_data;
  auto ref_list = rhs.m_list_data;
  auto current_indexes = m_indexes;
  auto ref_indexes = rhs.m_indexes;
  auto* current_list_ptr = current_list;
  auto* ref_list_ptr = ref_list;
  Int32 current_list_size = m_list_data_size;
  Int32 ref_list_size = rhs.m_list_data_size;
  if (current_list_ptr!=ref_list_ptr)
    ARCANE_FATAL("Bad list base pointer current={0} ref={1}",current_list_ptr,ref_list_ptr);
  if (current_list_size!=ref_list_size)
    ARCANE_FATAL("Bad list size current={0} ref={1}",current_list_size,ref_list_size);
  auto* current_indexes_ptr = current_indexes;
  auto* ref_indexes_ptr = ref_indexes;
  Int32 current_indexes_size = m_nb_item;
  Int32 ref_indexes_size = rhs.m_nb_item;
  if (current_indexes_ptr!=ref_indexes_ptr)
    ARCANE_FATAL("Bad indexes base pointer current={0} ref={1}",current_indexes_ptr,ref_indexes_ptr);
  if (current_indexes_size!=ref_indexes_size)
    ARCANE_FATAL("Bad indexes size current={0} ref={1}",current_indexes_size,ref_indexes_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemConnectivityContainerView::
_checkSize(Int32 indexes_size, Int32 nb_connected_item_size)
{
  // La valeur de 'nb_connected_item_size' doit être égale à 'indexes_size'
  if (indexes_size!=nb_connected_item_size)
    ARCANE_FATAL("Bad sizes indexes_size={0} nb_connected_item_size={1}",
                 indexes_size,nb_connected_item_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
