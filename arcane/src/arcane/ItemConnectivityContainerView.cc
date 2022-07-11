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
  auto current_list = m_list;
  auto ref_list = rhs.m_list;
  auto current_indexes = m_indexes;
  auto ref_indexes = rhs.m_indexes;
  auto* current_list_ptr = current_list.data();
  auto* ref_list_ptr = ref_list.data();
  if (current_list_ptr!=ref_list_ptr)
    ARCANE_FATAL("Bad list base pointer current={0} ref={1}",current_list_ptr,ref_list_ptr);
  if (current_list.size()!=ref_list.size())
    ARCANE_FATAL("Bad list size current={0} ref={1}",current_list.size(),ref_list.size());
  auto* current_indexes_ptr = current_indexes.data();
  auto* ref_indexes_ptr = ref_indexes.data();
  if (current_indexes_ptr!=ref_indexes_ptr)
    ARCANE_FATAL("Bad indexes base pointer current={0} ref={1}",current_indexes_ptr,ref_indexes_ptr);
  if (current_indexes.size()!=ref_indexes.size())
    ARCANE_FATAL("Bad indexes size current={0} ref={1}",current_indexes.size(),ref_indexes.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
