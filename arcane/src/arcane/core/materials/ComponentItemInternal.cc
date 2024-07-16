// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternal.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Partie interne d'une maille matériau ou milieu.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemInternal.h"

#include "arcane/utils/FixedArray.h"
#include "arcane/utils/FatalErrorException.h"

#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur pour l'entité nulle.
 *
 * Cela permet d'utiliser ComponentItemSharedInfo::_nullInstance()
 */
class NullComponentItemSharedInfoContainer
{
 public:

  FixedArray<ConstituentItemIndex, 2> m_first_sub_constituent_item_id_list;
  FixedArray<Int16, 2> m_component_id_list = { { -1, -1 } };
  FixedArray<Int16, 2> m_nb_sub_constituent_item_list = {};
  FixedArray<Int32, 2> m_global_item_local_id_list = { { NULL_ITEM_LOCAL_ID } };
  FixedArray<ConstituentItemIndex, 2> m_super_component_item_local_id_list = {};
  FixedArray<MatVarIndex, 2> m_var_index_list = { { MatVarIndex(-1, -1) } };
};

namespace
{
  NullComponentItemSharedInfoContainer global_null_component_item_shared_info_container;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemSharedInfo ComponentItemSharedInfo::null_shared_info;
ComponentItemSharedInfo* ComponentItemSharedInfo::null_shared_info_pointer = &ComponentItemSharedInfo::null_shared_info;
namespace
{
  std::once_flag component_set_null_instance_once_flag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemSharedInfo::
_setNullInstance()
{
  auto init_func = []() {
    ComponentItemSharedInfo* x = null_shared_info_pointer;
    NullComponentItemSharedInfoContainer& c = global_null_component_item_shared_info_container;

    x->m_storage_size = 0;
    x->m_first_sub_constituent_item_id_data = c.m_first_sub_constituent_item_id_list.data() + 1;
    x->m_super_component_item_local_id_data = c.m_super_component_item_local_id_list.data() + 1;
    x->m_component_id_data = c.m_component_id_list.data() + 1;

    x->m_nb_sub_constituent_item_data = c.m_nb_sub_constituent_item_list.data() + 1;
    x->m_global_item_local_id_data = c.m_global_item_local_id_list.data() + 1;
    x->m_var_index_data = c.m_var_index_list.data() + 1;
  };
  // Garanti que cela ne sera appelé qu'une seule fois et protège des appels
  // concurrents.
  std::call_once(component_set_null_instance_once_flag, init_func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o, const ConstituentItemIndex& id)
{
  o << id.localId();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemLocalIdListView::
_checkCoherency() const
{
  if (!m_component_shared_info)
    ARCANE_FATAL("Null ComponentItemSharedInfo");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
