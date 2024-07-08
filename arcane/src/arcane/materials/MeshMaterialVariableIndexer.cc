// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableIndexer.cc                              (C) 2000-2024 */
/*                                                                           */
/* Indexer pour les variables materiaux.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"
#include "arcane/materials/internal/ConstituentModifierWorkInfo.h"

#include "arcane/accelerator/Filter.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableIndexer::
MeshMaterialVariableIndexer(ITraceMng* tm, const String& name)
: TraceAccessor(tm)
, m_name(name)
, m_matvar_indexes(platform::getAcceleratorHostMemoryAllocator())
, m_local_ids(platform::getAcceleratorHostMemoryAllocator())
{
  _init();
  m_matvar_indexes.setDebugName(String("VariableIndexerMatVarIndexes")+name);
  m_local_ids.setDebugName(String("VariableIndexerLocalIdsIndexes")+name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_init()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdate(const ComponentItemListBuilderOld& builder)
{
  ConstArrayView<MatVarIndex> pure_matvar = builder.pureMatVarIndexes();
  ConstArrayView<MatVarIndex> partial_matvar = builder.partialMatVarIndexes();

  Integer nb_pure = pure_matvar.size();
  Integer nb_partial = partial_matvar.size();
  m_matvar_indexes.resize(nb_pure + nb_partial);

  m_matvar_indexes.subView(0, nb_pure).copy(pure_matvar);
  m_matvar_indexes.subView(nb_pure, nb_partial).copy(partial_matvar);

  Int32ConstArrayView local_ids_in_multiple = builder.partialLocalIds();

  {
    m_local_ids.resize(nb_pure + nb_partial);
    Integer index = 0;
    for (Integer i = 0, n = nb_pure; i < n; ++i) {
      m_local_ids[index] = pure_matvar[i].valueIndex();
      ++index;
    }
    for (Integer i = 0, n = nb_partial; i < n; ++i) {
      m_local_ids[index] = local_ids_in_multiple[i];
      ++index;
    }
  }

  // NOTE: a priori, ici on est sur que m_max_index_in_multiple_array vaut
  // nb_partial+1
  {
    Int32 max_index_in_multiple = (-1);
    for (Integer i = 0; i < nb_partial; ++i) {
      max_index_in_multiple = math::max(partial_matvar[i].valueIndex(), max_index_in_multiple);
    }
    m_max_index_in_multiple_array = max_index_in_multiple;
  }

  info(4) << "END_UPDATE max_index=" << m_max_index_in_multiple_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdateAdd(const ComponentItemListBuilder& builder, RunQueue& queue)
{
  SmallSpan<const MatVarIndex> pure_matvar = builder.pureMatVarIndexes();
  SmallSpan<const MatVarIndex> partial_matvar = builder.partialMatVarIndexes();

  Integer nb_pure_to_add = pure_matvar.size();
  Integer nb_partial_to_add = partial_matvar.size();
  Integer total_to_add = nb_pure_to_add + nb_partial_to_add;
  Integer current_nb_item = nbItem();
  const Int32 new_size = current_nb_item + total_to_add;

  MemoryUtils::checkResizeArrayWithCapacity(m_matvar_indexes, new_size, false);
  MemoryUtils::checkResizeArrayWithCapacity(m_local_ids, new_size, false);

  SmallSpan<const Int32> local_ids_in_multiple = builder.partialLocalIds();
  SmallSpan<Int32> local_ids_view = m_local_ids.subView(current_nb_item, total_to_add);
  SmallSpan<MatVarIndex> matvar_indexes = m_matvar_indexes.subView(current_nb_item, total_to_add);

  Int32 max_index_in_multiple = m_max_index_in_multiple_array;
  {
    auto command = makeCommand(queue);
    Arcane::Accelerator::ReducerMax2<Int32> max_index_reducer(command);
    Int32 max_to_add = math::max(nb_pure_to_add, nb_partial_to_add);
    command << RUNCOMMAND_LOOP1_EX(iter, max_to_add, max_index_reducer)
    {
      auto [i] = iter();
      if (i < nb_pure_to_add) {
        local_ids_view[i] = pure_matvar[i].valueIndex();
        matvar_indexes[i] = pure_matvar[i];
      }
      if (i < nb_partial_to_add) {
        local_ids_view[nb_pure_to_add + i] = local_ids_in_multiple[i];
        matvar_indexes[nb_pure_to_add + i] = partial_matvar[i];
        max_index_reducer.combine(partial_matvar[i].valueIndex());
      }
    };
    max_index_in_multiple = math::max(max_index_reducer.reducedValue(), m_max_index_in_multiple_array);
  }
  m_max_index_in_multiple_array = max_index_in_multiple;

  info(4) << "END_UPDATE_ADD max_index=" << m_max_index_in_multiple_array
          << " nb_partial_to_add=" << nb_partial_to_add;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdateRemove(ConstituentModifierWorkInfo& work_info, Integer nb_remove, RunQueue& queue)
{
  endUpdateRemoveV2(work_info, nb_remove, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
endUpdateRemoveV2(ConstituentModifierWorkInfo& work_info, Integer nb_remove, RunQueue& queue)
{
  if (nb_remove == 0)
    return;

  Integer nb_item = nbItem();
  Integer orig_nb_item = nb_item;

  info(4) << "EndUpdateRemoveV2 nb_remove=" << nb_remove << " nb_item=" << nb_item;

  if (nb_remove == nb_item) {
    m_matvar_indexes.clear();
    m_local_ids.clear();
    return;
  }

  bool is_device = isAcceleratorPolicy(queue.executionPolicy());
  auto saved_matvar_indexes_modifier = work_info.m_saved_matvar_indexes.modifier(is_device);
  auto saved_local_ids_modifier = work_info.m_saved_local_ids.modifier(is_device);

  saved_matvar_indexes_modifier.resize(nb_remove);
  saved_local_ids_modifier.resize(nb_remove);

  Accelerator::GenericFilterer filterer(&queue);
  SmallSpan<const bool> removed_cells = work_info.removedCells();
  Span<MatVarIndex> last_matvar_indexes(saved_matvar_indexes_modifier.view());
  Span<Int32> last_local_ids(saved_local_ids_modifier.view());
  Span<Int32> local_ids(m_local_ids);
  Span<MatVarIndex> matvar_indexes(m_matvar_indexes);

  // Conserve \a nb_remove valeurs en partant de la fin de la liste
  {
    Int32 last_index = nb_item - 1;
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      Int32 lid = local_ids[last_index - index];
      return !removed_cells[lid];
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      Int32 true_index = (last_index - input_index);
      if (output_index < nb_remove) {
        last_matvar_indexes[output_index] = matvar_indexes[true_index];
        last_local_ids[output_index] = local_ids[true_index];
      }
    };
    filterer.applyWithIndex(orig_nb_item, select_lambda, setter_lambda, A_FUNCINFO);
    filterer.nbOutputElement();
  }

  // Remplit les trous des mailles supprimées avec les derniers éléments de la liste
  {
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      Int32 lid = local_ids[index];
      return removed_cells[lid];
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      matvar_indexes[input_index] = last_matvar_indexes[output_index];
      local_ids[input_index] = last_local_ids[output_index];
    };
    filterer.applyWithIndex(orig_nb_item - nb_remove, select_lambda, setter_lambda, A_FUNCINFO);
    filterer.nbOutputElement();
  }
  nb_item -= nb_remove;
  m_matvar_indexes.resize(nb_item);
  m_local_ids.resize(nb_item);

  // Vérifie qu'on a bien supprimé autant d'entité que prévu.
  Integer nb_remove_computed = (orig_nb_item - nb_item);
  if (nb_remove_computed != nb_remove)
    ARCANE_FATAL("Bad number of removed material items expected={0} v={1} name={2}",
                 nb_remove, nb_remove_computed, name());
  info(4) << "END_UPDATE_REMOVE nb_removed=" << nb_remove;

  // TODO: il faut recalculer m_max_index_in_multiple_array
  // et compacter éventuellement les variables. (pas indispensable)
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  this->_changeLocalIdsV2(this, old_to_new_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
_changeLocalIdsV2(MeshMaterialVariableIndexer* var_indexer, Int32ConstArrayView old_to_new_ids)
{
  // Nouvelle version du changement des localId() qui ne modifie pas l'ordre
  // des m_matvar_indexes.

  ITraceMng* tm = var_indexer->traceMng();

  tm->info(4) << "ChangeLocalIdsV2 name=" << var_indexer->name();
  // Il faut recopier le tableau des localId() car il va être modifié.
  UniqueArray<Int32> ids_copy(var_indexer->localIds());
  UniqueArray<MatVarIndex> matvar_indexes_copy(var_indexer->matvarIndexes());

  var_indexer->m_local_ids.clear();
  var_indexer->m_matvar_indexes.clear();

  Integer nb = ids_copy.size();

  tm->info(4) << "-- -- BEGIN_PROCESSING N=" << ids_copy.size();

  for (Integer i = 0; i < nb; ++i) {
    Int32 lid = ids_copy[i];
    Int32 new_lid = old_to_new_ids[lid];
    tm->info(5) << "I=" << i << " lid=" << lid << " new_lid=" << new_lid;

    if (new_lid != NULL_ITEM_LOCAL_ID) {
      MatVarIndex mvi = matvar_indexes_copy[i];
      tm->info(5) << "I=" << i << " new_lid=" << new_lid << " mv=" << mvi;
      Int32 value_index = mvi.valueIndex();
      if (mvi.arrayIndex() == 0) {
        // TODO: Vérifier si value_index, qui contient le localId() de l'entité
        // ne dois pas être modifié.
        // Normalement, il faudra avoir:
        //    value_index = new_lid;
        // Mais cela plante actuellement car comme on ne récupère pas
        // l'évènement executeReduce() sur les groupes il est possible
        // que nous n'ayons pas les bons ids. (C'est quand même bizarre...)

        // Variable globale: met à jour le localId() dans le MatVarIndex.
        var_indexer->m_matvar_indexes.add(MatVarIndex(0, value_index));
        var_indexer->m_local_ids.add(value_index);
      }
      else {
        // Valeur partielle: rien ne change dans le MatVarIndex
        var_indexer->m_matvar_indexes.add(mvi);
        var_indexer->m_local_ids.add(new_lid);
      }
    }
  }

  // TODO: remplir la fin des tableaux avec des valeurs invalides (pour détecter les problèmes)
  tm->info(4) << "-- -- ChangeLocalIdsV2 END_PROCESSING (V4)"
              << " indexer_name=" << var_indexer->name()
              << " nb_ids=" << var_indexer->m_local_ids.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
transformCellsV2(ConstituentModifierWorkInfo& work_info, RunQueue& queue)
{
  _switchBetweenPureAndPartial(work_info, queue, work_info.isAdd());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Échange des mailles entre pures et partielles.
 */
void MeshMaterialVariableIndexer::
_switchBetweenPureAndPartial(ConstituentModifierWorkInfo& work_info,
                             RunQueue& queue,
                             bool is_pure_to_partial)
{
  bool is_device = isAcceleratorPolicy(queue.executionPolicy());

  Integer nb = nbItem();
  auto pure_local_ids_modifier = work_info.pure_local_ids.modifier(is_device);
  auto partial_indexes_modifier = work_info.partial_indexes.modifier(is_device);
  pure_local_ids_modifier.resize(nb);
  partial_indexes_modifier.resize(nb);

  SmallSpan<Int32> pure_local_ids = pure_local_ids_modifier.view();
  SmallSpan<Int32> partial_indexes = partial_indexes_modifier.view();
  SmallSpan<MatVarIndex> matvar_indexes = m_matvar_indexes.view();
  SmallSpan<Int32> local_ids = m_local_ids.view();
  SmallSpan<const bool> transformed_cells = work_info.transformedCells();

  Accelerator::GenericFilterer filterer(&queue);

  if (is_pure_to_partial) {
    // Transformation Pure -> Partial
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      MatVarIndex mvi = matvar_indexes[index];
      if (mvi.arrayIndex() != 0)
        return false;
      Int32 local_id = local_ids[index];
      bool do_transform = transformed_cells[local_id];
      return do_transform;
    };

    Int32 max_index_in_multiple_array = m_max_index_in_multiple_array + 1;
    const Int32 var_index = m_index + 1;
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      Int32 current_index = max_index_in_multiple_array + output_index;
      Int32 local_id = local_ids[input_index];
      pure_local_ids[output_index] = local_id;
      partial_indexes[output_index] = current_index;
      matvar_indexes[input_index] = MatVarIndex(var_index, current_index);
    };
    filterer.applyWithIndex(nb, select_lambda, setter_lambda, A_FUNCINFO);
  }
  else {
    // Transformation Partial -> Pure
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      MatVarIndex mvi = matvar_indexes[index];
      if (mvi.arrayIndex() == 0)
        return false;
      Int32 local_id = local_ids[index];
      bool do_transform = transformed_cells[local_id];
      return do_transform;
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      MatVarIndex mvi = matvar_indexes[input_index];
      Int32 local_id = local_ids[input_index];
      Int32 var_index = mvi.valueIndex();
      pure_local_ids[output_index] = local_id;
      partial_indexes[output_index] = var_index;
      matvar_indexes[input_index] = MatVarIndex(0, local_id);
    };
    filterer.applyWithIndex(nb, select_lambda, setter_lambda, A_FUNCINFO);
  }

  Int32 nb_out = filterer.nbOutputElement();
  pure_local_ids_modifier.resize(nb_out);
  partial_indexes_modifier.resize(nb_out);

  if (is_pure_to_partial)
    m_max_index_in_multiple_array += nb_out;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableIndexer::
checkValid()
{
  ValueChecker vc(A_FUNCINFO);

  Integer nb_item = nbItem();

  vc.areEqual(nb_item, m_matvar_indexes.size(), "Incoherent size for local ids and matvar indexes");

  // TODO: vérifier que les m_local_ids pour les parties pures correspondent
  // au m_matvar_indexes.valueIndex() correspondant.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
