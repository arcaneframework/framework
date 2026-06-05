// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentData.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Data of a constituent (material or medium) of a mesh.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/MeshComponentData.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"
#include "arcane/materials/internal/MeshComponentPartData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentData::
MeshComponentData(IMeshComponent* component, const String& name,
                  Int16 component_id, ComponentItemSharedInfo* shared_info,
                  bool create_indexer)
: TraceAccessor(component->traceMng())
, m_component(component)
, m_component_id(component_id)
, m_name(name)
, m_constituent_local_id_list(shared_info, String("MeshComponentDataIdList") + name)
, m_recompute_part_data_functor(this, &MeshComponentData::_rebuildPartDataDirect)
{
  if (create_indexer) {
    m_variable_indexer = new MeshMaterialVariableIndexer(traceMng(), name);
    m_is_indexer_owner = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshComponentData::
~MeshComponentData()
{
  delete m_part_data;
  if (m_is_indexer_owner)
    delete m_variable_indexer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_setPartInfo()
{
  if (m_part_data)
    m_part_data->_setConstituentListView(m_constituent_local_id_list.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_resizeItemsInternal(Integer nb_item)
{
  m_constituent_local_id_list.resize(nb_item);
  _setPartInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_setVariableIndexer(MeshMaterialVariableIndexer* indexer)
{
  m_variable_indexer = indexer;
  m_is_indexer_owner = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_setItems(const ItemGroup& group)
{
  m_items = group;
  if (m_variable_indexer)
    m_variable_indexer->setCells(group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_buildPartData()
{
  m_part_data = new MeshComponentPartData(m_component, m_name);
  m_part_data->setRecomputeFunctor(&m_recompute_part_data_functor);
  _setPartInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Updates the constituent's m_items_internal after a change in local numbering.
 *
 * \warning This method must be called before updating the
 * m_variable_indexer because its local_ids are used.
 */
void MeshComponentData::
_changeLocalIdsForInternalList(Int32ConstArrayView old_to_new_ids)
{
  // TODO: check if it is possible to remove the temporary array
  // new_internals (it is almost certain that it is possible).
  ConstArrayView<ConstituentItemIndex> current_internals(m_constituent_local_id_list.localIds());
  UniqueArray<ConstituentItemIndex> new_internals;

  Int32ConstArrayView local_ids = variableIndexer()->localIds();

  for (Integer i = 0, nb = current_internals.size(); i < nb; ++i) {
    Int32 lid = local_ids[i];
    Int32 new_lid = old_to_new_ids[lid];
    if (new_lid != NULL_ITEM_LOCAL_ID) {
      new_internals.add(current_internals[i]);
      m_constituent_local_id_list.itemBase(i)._setGlobalItem(ItemLocalId(new_lid));
    }
  }

  // TODO: check to remove this copy as well.
  {
    _resizeItemsInternal(new_internals.size());
    m_constituent_local_id_list.copy(new_internals);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_rebuildPartData(RunQueue& queue)
{
  if (!m_part_data)
    _buildPartData();
  _setPartInfo();
  const bool do_lazy_evaluation = true;
  if (do_lazy_evaluation)
    m_part_data->setNeedRecompute();
  else
    m_part_data->_setFromMatVarIndexes(m_variable_indexer->matvarIndexes(), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
_rebuildPartDataDirect()
{
  RunQueue& queue = m_component->materialMng()->_internalApi()->runQueue();
  m_part_data->_setFromMatVarIndexes(m_variable_indexer->matvarIndexes(), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshComponentData::
checkValid()
{
  if (!arcaneIsCheck())
    return;

  info(4) << "Check valid component name=" << name();
  m_variable_indexer->checkValid();

  // Checks that the groups are consistent between this instance
  // and the group in m_variable_indexer
  if (m_items != m_variable_indexer->cells())
    ARCANE_FATAL("Incoherent group for component name={0} data={1} indexer={2}",
                 name(), m_items.name(), m_variable_indexer->cells().name());

  // Checks that the list of indexer->localIds() and the one
  // from the 'cells' group is the same. To do this, an array
  // is declared which is indexed by the mesh's localId(). For each element,
  // 1 is added if the mesh is in the group and 2 if it is in the
  // component's internal structures. If the final value is not 3,
  // there is an incoherence.
  {
    IItemFamily* family = m_items.itemFamily();
    UniqueArray<Int32> presence(family->maxLocalId());
    presence.fill(0);
    ENUMERATE_ITEM (iitem, m_items) {
      presence[iitem.itemLocalId()] = 1;
    }
    Int32ConstArrayView indexer_local_ids = m_variable_indexer->localIds();
    for (Integer i = 0, n = indexer_local_ids.size(); i < n; ++i)
      presence[indexer_local_ids[i]] += 2;
    ItemInfoListView items_internal = family->itemInfoListView();
    Integer nb_error = 0;
    for (Integer i = 0, n = presence.size(); i < n; ++i) {
      Int32 v = presence[i];
      if (v == 3 || v == 0)
        continue;
      Cell cell(items_internal[i]);
      ++nb_error;
      info(4) << "WARNING: Incoherence between group and internals "
              << " component=" << name() << " v=" << v
              << " cell=" << ItemPrinter(cell);
    }
    if (nb_error != 0) {
      warning() << "WARNING: Incoherence between group and internals "
                << " component=" << name() << " nb_error=" << nb_error;
    }
  }

  // Checks the consistency of MatVarIndex between the direct array
  // and the values in m_items_internal
  ConstArrayView<MatVarIndex> mat_var_indexes(m_variable_indexer->matvarIndexes());
  Integer nb_val = mat_var_indexes.size();
  info(4) << "CheckValid component_name=" << name()
          << " matvar_indexes=" << mat_var_indexes;
  info(4) << "Cells=" << m_variable_indexer->cells().view().localIds();
  for (Integer i = 0; i < nb_val; ++i) {
    MatVarIndex component_mvi = m_constituent_local_id_list.variableIndex(i);
    MatVarIndex mvi = mat_var_indexes[i];
    if (component_mvi != mvi)
      ARCANE_FATAL("Bad 'var_index' environment={3} component='{0}' direct='{1}' i={2}",
                   component_mvi, mvi, i, name());
  }

  if (m_part_data)
    m_part_data->checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
