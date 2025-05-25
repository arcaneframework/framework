// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemVectorImpl.cc                                (C) 2000-2025 */
/*                                                                           */
/* Implémentation de 'IConstituentItemVectorImpl'.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ConstituentItemVectorImpl.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/Partitioner.h"
#include "arcane/accelerator/RunCommandMaterialEnumerate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorImpl::
ConstituentItemVectorImpl(IMeshComponent* component)
: m_material_mng(component->materialMng())
, m_component(component)
, m_matvar_indexes(platform::getDefaultDataAllocator())
, m_items_local_id(platform::getDefaultDataAllocator())
, m_part_data(std::make_unique<MeshComponentPartData>(component, String()))
, m_recompute_part_data_functor(this, &ConstituentItemVectorImpl::_recomputePartData)
{
  Int32 level = -1;
  if (component->isMaterial())
    level = LEVEL_MATERIAL;
  else if (component->isEnvironment())
    level = LEVEL_ENVIRONMENT;
  else
    ARCANE_FATAL("Bad internal type of component");
  m_component_shared_info = m_material_mng->_internalApi()->componentItemSharedInfo(level);
  m_constituent_list = std::make_unique<ConstituentItemLocalIdList>(m_component_shared_info, String());
  m_part_data->setRecomputeFunctor(&m_recompute_part_data_functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorImpl::
ConstituentItemVectorImpl(const ComponentItemVectorView& rhs)
: ConstituentItemVectorImpl(rhs.component())
{
  RunQueue& queue = m_material_mng->_internalApi()->runQueue();
  m_constituent_list->copy(rhs._constituentItemListView());
  m_matvar_indexes.copy(rhs._matvarIndexes());
  m_items_local_id.copy(rhs._internalLocalIds());
  m_part_data->_setFromMatVarIndexes(rhs._matvarIndexes(), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Helper pour positionner les entités du vecteur.
 */
class ConstituentItemVectorImpl::SetItemHelper
{
 public:

  explicit SetItemHelper(bool use_new_impl)
  : m_use_new_impl(use_new_impl)
  {}

 public:

  template <typename ConstituentGetterLambda>
  void setItems(ConstituentItemVectorImpl* vector_impl,
                ConstituentGetterLambda constituent_getter_lambda,
                SmallSpan<const Int32> local_ids, RunQueue& queue);

 private:

  bool m_use_new_impl = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ConstituentGetterLambda> void
ConstituentItemVectorImpl::SetItemHelper::
setItems(ConstituentItemVectorImpl* vector_impl, ConstituentGetterLambda constituent_getter_lambda,
         SmallSpan<const Int32> local_ids, RunQueue& queue)
{
  SmallSpan<ConstituentItemIndex> item_indexes = vector_impl->m_constituent_list->_mutableItemIndexList();
  SmallSpan<MatVarIndex> matvar_indexes = vector_impl->m_matvar_indexes.smallSpan();
  SmallSpan<Int32> items_local_id = vector_impl->m_items_local_id.smallSpan();
  AllEnvCellVectorView all_env_cell_view = vector_impl->m_material_mng->view(local_ids);
  const bool is_env = vector_impl->m_component->isEnvironment();
  const Int32 component_id = vector_impl->m_component->id();

  const Int32 nb_pure = vector_impl->m_nb_pure;

  Int32 pure_index = 0;
  Int32 impure_index = nb_pure;

  const Int32 nb_id = local_ids.size();

  // Pas besoin de conserver les informations pour les mailles pour lesquelles
  // le constituant est absent.
  auto setter_unselected = [=] ARCCORE_HOST_DEVICE(Int32, Int32) {
  };
  auto generic_setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index, ComponentCell cc) {
    MatVarIndex idx = cc._varIndex();
    ConstituentItemIndex cii = cc._constituentItemIndex();
    item_indexes[index] = cii;
    matvar_indexes[index] = idx;
    items_local_id[index] = cc.globalCellId();
  };

  // Implémentation utilisant l'API accélérateur
  if (m_use_new_impl) {
    // Lambda pour sélectionner les mailles pures
    auto select_pure = [=] ARCCORE_HOST_DEVICE(Int32 index) {
      ComponentCell cc = constituent_getter_lambda(index);
      if (cc.null())
        return false;
      return (cc._varIndex().arrayIndex() == 0);
    };
    // Lambda pour sélectionner les mailles impures
    auto select_impure = [=] ARCCORE_HOST_DEVICE(Int32 index) {
      ComponentCell cc = constituent_getter_lambda(index);
      if (cc.null())
        return false;
      return (cc._varIndex().arrayIndex() != 0);
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index, Int32 output_index) {
      ComponentCell cc = constituent_getter_lambda(index);
      if (!cc.null())
        generic_setter_lambda(output_index, cc);
    };
    auto setter_pure = [=] ARCCORE_HOST_DEVICE(Int32 index, Int32 output_index) {
      setter_lambda(index, output_index);
    };
    auto setter_impure = [=] ARCCORE_HOST_DEVICE(Int32 index, Int32 output_index) {
      setter_lambda(index, output_index + nb_pure);
    };
    Arcane::Accelerator::GenericPartitioner generic_partitioner(queue);
    generic_partitioner.applyWithIndex(nb_id, setter_pure, setter_impure, setter_unselected,
                                       select_pure, select_impure, A_FUNCINFO);
    //SmallSpan<const Int32> nb_parts = generic_partitioner.nbParts();
    //std::cout << "NB_PART=" << nb_parts[0] << " " << nb_parts[1] << "\n";
  }
  else {
    // Ancien mécanisme qui n'utilise pas l'API accélérateur
    if (is_env) {
      ENUMERATE_ALLENVCELL (iallenvcell, all_env_cell_view) {
        AllEnvCell all_env_cell = *iallenvcell;
        for (EnvCell ec : all_env_cell.subEnvItems()) {
          if (ec.componentId() == component_id) {
            MatVarIndex idx = ec._varIndex();
            ConstituentItemIndex cii = ec._constituentItemIndex();
            Int32& base_index = (idx.arrayIndex() == 0) ? pure_index : impure_index;
            item_indexes[base_index] = cii;
            matvar_indexes[base_index] = idx;
            items_local_id[base_index] = all_env_cell.globalCellId();
            ++base_index;
          }
        }
      }
    }
    else {
      // Filtre les matériaux correspondants aux local_ids
      ENUMERATE_ALLENVCELL (iallenvcell, all_env_cell_view) {
        AllEnvCell all_env_cell = *iallenvcell;
        for (EnvCell env_cell : all_env_cell.subEnvItems()) {
          for (MatCell mc : env_cell.subMatItems()) {
            if (mc.componentId() == component_id) {
              MatVarIndex idx = mc._varIndex();
              ConstituentItemIndex cii = mc._constituentItemIndex();
              Int32& base_index = (idx.arrayIndex() == 0) ? pure_index : impure_index;
              item_indexes[base_index] = cii;
              matvar_indexes[base_index] = idx;
              items_local_id[base_index] = all_env_cell.globalCellId();
              ++base_index;
            }
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne les entités du vecteur.
 *
 * Les entités du vecteur seront les entités de numéro locaux localId() et
 * qui appartiennent à notre matériau ou notre milieu.
 */
void ConstituentItemVectorImpl::
_setItems(SmallSpan<const Int32> local_ids)
{
  const bool do_new_impl = m_material_mng->_internalApi()->isUseAcceleratorForConstituentItemVector();

  Accelerator::eExecutionPolicy exec_policy = m_component->specificExecutionPolicy();
  RunQueue queue = m_material_mng->_internalApi()->runQueue(exec_policy);

  if (do_new_impl)
    _computeNbPureAndImpure(local_ids, queue);
  else
    _computeNbPureAndImpureLegacy(local_ids);

  const Int32 nb_pure = m_nb_pure;
  const Int32 nb_impure = m_nb_impure;
  const Int32 total_nb_pure_and_impure = nb_pure + nb_impure;

  // Tableau qui contiendra les indices des mailles pures et partielles.
  // La première partie de 0 à nb_pure contiendra la partie pure.
  // La seconde partie de nb_pure à (nb_pure+nb_impure) contiendra les mailles partielles.
  // A noter que (nb_pure + nb_impure) peut être différent de local_ids.size()
  // si certaines mailles de \a local_ids n'ont pas le constituant.
  m_constituent_list->resize(total_nb_pure_and_impure);

  // TODO: Ne pas remettre à jour systématiquement les
  // 'm_items_local_id' mais ne le faire qu'à la demande
  // car ils ne sont pas utilisés souvent.

  m_matvar_indexes.resize(total_nb_pure_and_impure);
  m_items_local_id.resize(total_nb_pure_and_impure);

  const bool is_env = m_component->isEnvironment();
  AllEnvCellVectorView all_env_cell_view = m_material_mng->view(local_ids);
  const Int32 component_id = m_component->id();

  // Lambda pour récupérer le milieu associé à la maille
  auto env_component_getter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> ComponentCell {
    AllEnvCell all_env_cell = all_env_cell_view[index];
    for (EnvCell ec : all_env_cell.subEnvItems()) {
      if (ec.componentId() == component_id)
        return ec;
    }
    return {};
  };

  // Lambda pour récupérer le matériau associé à la maille
  auto mat_component_getter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> ComponentCell {
    AllEnvCell all_env_cell = all_env_cell_view[index];
    for (EnvCell ec : all_env_cell.subEnvItems()) {
      for (MatCell mc : ec.subMatItems())
        if (mc.componentId() == component_id)
          return mc;
    }
    return {};
  };

  {
    SetItemHelper helper(do_new_impl);
    if (is_env)
      helper.setItems(this, env_component_getter_lambda, local_ids, queue);
    else
      helper.setItems(this, mat_component_getter_lambda, local_ids, queue);
  }

  // Mise à jour de MeshComponentPartData
  const bool do_lazy_evaluation = true;
  if (do_lazy_evaluation)
    m_part_data->setNeedRecompute();
  else
    _recomputePartData();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemVectorImpl::
_computeNbPureAndImpure(SmallSpan<const Int32> local_ids, RunQueue& queue)
{
  IMeshComponent* component = m_component;
  const bool is_env = component->isEnvironment();
  AllEnvCellVectorView all_env_cell_view = m_material_mng->view(local_ids);
  const Int32 component_id = m_component->id();

  auto command = makeCommand(queue);
  Accelerator::ReducerSum2<Int32> nb_pure(command);
  Accelerator::ReducerSum2<Int32> nb_impure(command);

  // Calcule le nombre de mailles pures et partielles
  if (is_env) {
    command << RUNCOMMAND_MAT_ENUMERATE(AllEnvCell, all_env_cell, all_env_cell_view, nb_pure, nb_impure)
    {
      for (EnvCell ec : all_env_cell.subEnvItems()) {
        if (ec.componentId() == component_id) {
          MatVarIndex idx = ec._varIndex();
          if (idx.arrayIndex() == 0)
            nb_pure.combine(1);
          else
            nb_impure.combine(1);
        }
      }
    };
  }
  else {
    command << RUNCOMMAND_MAT_ENUMERATE(AllEnvCell, all_env_cell, all_env_cell_view, nb_pure, nb_impure)
    {
      for (EnvCell env_cell : all_env_cell.subEnvItems()) {
        for (MatCell mc : env_cell.subMatItems()) {
          if (mc.componentId() == component_id) {
            MatVarIndex idx = mc._varIndex();
            if (idx.arrayIndex() == 0)
              nb_pure.combine(1);
            else
              nb_impure.combine(1);
          }
        }
      }
    };
  }

  m_nb_pure = nb_pure.reducedValue();
  m_nb_impure = nb_impure.reducedValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du nombre de mailles pures et impures sans API accélérateur.
 */
void ConstituentItemVectorImpl::
_computeNbPureAndImpureLegacy(SmallSpan<const Int32> local_ids)
{
  IMeshComponent* component = m_component;
  const bool is_env = component->isEnvironment();
  AllEnvCellVectorView all_env_cell_view = m_material_mng->view(local_ids);
  const Int32 component_id = m_component->id();

  Int32 nb_pure = 0;
  Int32 nb_impure = 0;

  // Calcule le nombre de mailles pures et partielles
  if (is_env) {
    ENUMERATE_ALLENVCELL (iallenvcell, all_env_cell_view) {
      AllEnvCell all_env_cell = *iallenvcell;
      for (EnvCell ec : all_env_cell.subEnvItems()) {
        if (ec.componentId() == component_id) {
          MatVarIndex idx = ec._varIndex();
          if (idx.arrayIndex() == 0)
            ++nb_pure;
          else
            ++nb_impure;
        }
      }
    }
  }
  else {
    ENUMERATE_ALLENVCELL (iallenvcell, all_env_cell_view) {
      AllEnvCell all_env_cell = *iallenvcell;
      for (EnvCell env_cell : all_env_cell.subEnvItems()) {
        for (MatCell mc : env_cell.subMatItems()) {
          if (mc.componentId() == component_id) {
            MatVarIndex idx = mc._varIndex();
            if (idx.arrayIndex() == 0)
              ++nb_pure;
            else
              ++nb_impure;
          }
        }
      }
    }
  }

  m_nb_pure = nb_pure;
  m_nb_impure = nb_impure;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemVectorImpl::
_recomputePartData()
{
  // Mise à jour de MeshComponentPartData
  auto mvi_pure_view = m_matvar_indexes.subView(0, m_nb_pure);
  auto mvi_impure_view = m_matvar_indexes.subView(m_nb_pure, m_nb_impure);
  m_part_data->_setFromMatVarIndexes(mvi_pure_view, mvi_impure_view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ConstituentItemVectorImpl::
_view() const
{
  return { m_component, m_matvar_indexes,
           m_constituent_list->view(), m_items_local_id };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
