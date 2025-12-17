// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshComponentData.h                                         (C) 2000-2025 */
/*                                                                           */
/* Données d'un constituant (matériau ou milieu) d'un maillage.              */
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
 * \brief Données d'un constituant (matériau ou milieu) d'un maillage.
 *
 * Cette classe contient les données communes à MeshMaterial et MeshEnvironnment.
 *
 * Cette classe est interne à Arcane.
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

  //! Retourne une instance vers la \a index-ème entité de la liste
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

  //! Constituant dont on gère les données.
  IMeshComponent* m_component = nullptr;

  /*!
   * \brief Indice du constituant (dans la liste des constituants de ce type).
   * \sa IMeshMaterialMng.
   */
  Int16 m_component_id = -1;

  //! Nom du constituant
  String m_name;

  //! Liste des entités de ce constituant
  ItemGroup m_items;

  //! Indique si on est propriétaire de l'indexeur (dans ce cas on le détruira avec l'instance)
  bool m_is_indexer_owner = false;

  //! Infos pour l'indexation des variables partielles.
  MeshMaterialVariableIndexer* m_variable_indexer = nullptr;

  //! Liste des ConstituentItemIndex pour ce constituant.
  ConstituentItemLocalIdList m_constituent_local_id_list;

  MeshComponentPartData* m_part_data = nullptr;
  FunctorT<MeshComponentData> m_recompute_part_data_functor;

  //! Politique d'exécution spécifique
  Accelerator::eExecutionPolicy m_specific_execution_policy = Accelerator::eExecutionPolicy::None;

 private:

  void _setPartInfo();
  void _rebuildPartDataDirect();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
