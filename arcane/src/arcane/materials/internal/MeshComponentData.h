// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentData.h                                         (C) 2000-2025 */
/*                                                                           */
/* Data of a constituent (material or medium) of a mesh.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHCOMPONENTDATA_H
#define ARCANE_MATERIALS_MESHCOMPONENTDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Functor.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/internal/ConstituentItemLocalIdList.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class MeshEnvironment;
class MatItemVectorView;
class MeshMaterialVariableIndexer;
class MeshComponentPartData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Data of a constituent (material or medium) of a mesh.
 *
 * This class contains data common to MeshMaterial and MeshEnvironment.
 *
 * This class is internal to Arcane.
 */
class MeshComponentData
: public TraceAccessor
{
  friend class MeshEnvironment;
  friend class MeshMaterial;
  friend class AllEnvData;

 public:

  MeshComponentData(IMeshComponent* component, const String& name,
                    Int16 component_id, ComponentItemSharedInfo* shared_info,
                    bool create_indexer);
  ~MeshComponentData() override;

 public:

  const String& name() const { return m_name; }

  MeshMaterialVariableIndexer* variableIndexer() const
  {
    return m_variable_indexer;
  }

  ConstituentItemLocalIdListView constituentItemListView() const
  {
    return m_constituent_local_id_list.view();
  }

 private:

  //! Returns an instance of the \a index-th entity in the list
  matimpl::ConstituentItemBase _itemBase(Int32 index) const
  {
    return m_constituent_local_id_list.itemBase(index);
  }

  void _setConstituentItem(Int32 index, ConstituentItemIndex id)
  {
    return m_constituent_local_id_list.setConstituentItem(index, id);
  }

  void checkValid();

 public:

  const ItemGroup& items() const
  {
    return m_items;
  }

  Int16 componentId() const
  {
    return m_component_id;
  }

  void setSpecificExecutionPolicy(Accelerator::eExecutionPolicy policy)
  {
    m_specific_execution_policy = policy;
  }

  Accelerator::eExecutionPolicy specificExecutionPolicy() const
  {
    return m_specific_execution_policy;
  }

 private:

  void _resizeItemsInternal(Int32 nb_item);
  void _setVariableIndexer(MeshMaterialVariableIndexer* indexer);
  void _setItems(const ItemGroup& group);
  void _changeLocalIdsForInternalList(Int32ConstArrayView old_to_new_ids);
  void _rebuildPartData(RunQueue& queue);
  void _buildPartData();
  MeshComponentPartData* _partData() const { return m_part_data; }

 private:

  //! Constituent whose data is managed.
  IMeshComponent* m_component = nullptr;

  /*!
   * \brief Constituent index (in the list of constituents of this type).
   * \sa IMeshMaterialMng.
   */
  Int16 m_component_id = -1;

  //! Name of the constituent
  String m_name;

  //! List of entities of this constituent
  ItemGroup m_items;

  //! Indicates if we own the indexer (in this case, it will be destroyed with the instance)
  bool m_is_indexer_owner = false;

  //! Info for indexing partial variables.
  MeshMaterialVariableIndexer* m_variable_indexer = nullptr;

  //! List of ConstituentItemIndex for this constituent.
  ConstituentItemLocalIdList m_constituent_local_id_list;

  MeshComponentPartData* m_part_data = nullptr;
  FunctorT<MeshComponentData> m_recompute_part_data_functor;

  //! Specific execution policy
  Accelerator::eExecutionPolicy m_specific_execution_policy = Accelerator::eExecutionPolicy::None;

 private:

  void _setPartInfo();
  void _rebuildPartDataDirect();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
