// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier_Accelerator.cc                 (C) 2000-2024 */
/*                                                                           */
/* Modification incrémentale des constituants.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/IncrementalComponentModifier.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FunctorUtils.h"

#include "arcane/core/internal/ItemGroupImplInternal.h"
#include "arcane/core/materials/IMeshMaterialVariable.h"

#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/MaterialModifierOperation.h"
#include "arcane/materials/internal/ConstituentConnectivityList.h"
#include "arcane/materials/internal/AllEnvData.h"

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Filter.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/core/ProfileRegion.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
apply(MaterialModifierOperation* operation)
{
  const char* str_add = "ApplyConstituentOperationAdd";
  const char* str_remove = "ApplyConstituentOperationRemove";
  bool is_add = operation->isAdd();
  Int32 color = (is_add) ? 0x00FFFF : 0x007FFF;
  Accelerator::ProfileRegion ps(m_queue, is_add ? str_add : str_remove, color);

  IMeshMaterial* mat = operation->material();
  SmallSpan<const Int32> orig_ids = operation->ids();
  SmallSpan<const Int32> ids = orig_ids;

  auto* true_mat = ARCANE_CHECK_POINTER(dynamic_cast<MeshMaterial*>(mat));

  const IMeshEnvironment* env = mat->environment();
  MeshEnvironment* true_env = true_mat->trueEnvironment();
  const Integer nb_mat = env->nbMaterial();

  info(4) << "-- ** -- Using optimisation updateMaterialDirect is_add=" << is_add
          << " mat=" << mat->name() << " nb_item=" << orig_ids.size()
          << " mat_id=" << mat->id() << " env_id=" << env->id();

  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  const bool check_if_present = !m_queue.isAcceleratorPolicy();

  const bool is_device = m_queue.isAcceleratorPolicy();

  // Remplit les tableaux indiquants si un constituant est concerné par
  // la modification en cours. Si ce n'est pas le cas, on pourra éviter de le tester
  // dans la boucle des constituants.
  {
    m_work_info.m_is_materials_modified.fillHost(false);
    m_work_info.m_is_environments_modified.fillHost(false);
    m_work_info.m_is_materials_modified.sync(is_device);
    m_work_info.m_is_environments_modified.sync(is_device);

    {
      auto mat_modifier = m_work_info.m_is_materials_modified.modifier(is_device);
      auto env_modifier = m_work_info.m_is_environments_modified.modifier(is_device);
      connectivity->fillModifiedConstituents(orig_ids, mat_modifier.view(), env_modifier.view(), mat->id(), is_add, m_queue);
      if (m_is_debug)
        connectivity->printConstituents(orig_ids);
    }
    {
      auto is_mat_modified = m_work_info.m_is_materials_modified.view(false);
      auto is_env_modified = m_work_info.m_is_environments_modified.view(false);
      info(4) << "ModifiedInfosAfter: mats=" << is_mat_modified << " envs=" << is_env_modified;
    }
  }

  if (nb_mat != 1) {

    // S'il est possible d'avoir plusieurs matériaux par milieu, il faut gérer
    // pour chaque maille si le milieu évolue suite à l'ajout/suppression de matériau.
    // Les deux cas sont :
    // - en cas d'ajout, le milieu évolue pour une maille s'il n'y avait pas
    //   de matériau avant. Dans ce cas le milieu est ajouté à la maille.
    // - en cas de suppression, le milieu évolue dans la maille s'il y avait
    //   1 seul matériau avant. Dans ce cas le milieu est supprimé de la maille.

    UniqueArray<Int32>& cells_changed_in_env = m_work_info.cells_changed_in_env;
    UniqueArray<Int32>& cells_unchanged_in_env = m_work_info.cells_unchanged_in_env;
    UniqueArray<Int16>& cells_current_nb_material = m_work_info.m_cells_current_nb_material;
    const Int32 nb_id = ids.size();
    cells_unchanged_in_env.resize(nb_id);
    cells_changed_in_env.resize(nb_id);
    cells_current_nb_material.resize(nb_id);
    const Int32 ref_nb_mat = is_add ? 0 : 1;
    const Int16 env_id = true_env->componentId();
    info(4) << "Using optimisation updateMaterialDirect is_add?=" << is_add;

    connectivity->fillCellsNbMaterial(ids, env_id, cells_current_nb_material.view(), m_queue);

    {
      Accelerator::GenericFilterer filterer(m_queue);
      SmallSpan<Int32> cells_unchanged_in_env_view = cells_unchanged_in_env.view();
      SmallSpan<Int32> cells_changed_in_env_view = cells_changed_in_env.view();
      SmallSpan<const Int16> cells_current_nb_material_view = m_work_info.m_cells_current_nb_material.view();
      {
        auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
          Int16 current_cell_nb_mat = cells_current_nb_material_view[index];
          return current_cell_nb_mat != ref_nb_mat;
        };
        auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
          cells_unchanged_in_env_view[output_index] = ids[input_index];
        };
        filterer.applyWithIndex(nb_id, select_lambda, setter_lambda, A_FUNCINFO);
        cells_unchanged_in_env.resize(filterer.nbOutputElement());
      }
      {
        auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
          Int16 current_cell_nb_mat = cells_current_nb_material_view[index];
          return current_cell_nb_mat == ref_nb_mat;
        };
        auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
          cells_changed_in_env_view[output_index] = ids[input_index];
        };
        filterer.applyWithIndex(nb_id, select_lambda, setter_lambda, A_FUNCINFO);
        cells_changed_in_env.resize(filterer.nbOutputElement());
      }
    }

    Integer nb_unchanged_in_env = cells_unchanged_in_env.size();
    info(4) << "Cells unchanged in environment n=" << nb_unchanged_in_env;

    Int16 mat_id = true_mat->componentId();
    if (is_add) {
      mat->cells().addItems(cells_unchanged_in_env, check_if_present);
      connectivity->addCellsToMaterial(mat_id, cells_unchanged_in_env.view(), m_queue);
      _addItemsToEnvironment(true_env, true_mat, cells_unchanged_in_env.view(), false);
    }
    else {
      flagRemovedCells(cells_unchanged_in_env, true);
      _removeItemsInGroup(mat->cells(), cells_unchanged_in_env);
      connectivity->removeCellsToMaterial(mat_id, cells_unchanged_in_env.view(), m_queue);
      _removeItemsFromEnvironment(true_env, true_mat, cells_unchanged_in_env.view(), false);
      flagRemovedCells(cells_unchanged_in_env, false);
    }

    // Prend pour \a ids uniquement la liste des mailles
    // qui n'appartenaient pas encore au milieu dans lequel on
    // ajoute le matériau.
    ids = cells_changed_in_env.view();
  }

  // Met à jour le nombre de milieux et de matériaux de chaque maille.
  // NOTE: il faut d'abord faire l'opération sur les milieux avant
  // les matériaux.
  {
    Int16 env_id = true_env->componentId();
    Int16 mat_id = true_mat->componentId();
    if (is_add) {
      connectivity->addCellsToEnvironment(env_id, ids, m_queue);
      connectivity->addCellsToMaterial(mat_id, ids, m_queue);
    }
    else {
      connectivity->removeCellsToEnvironment(env_id, ids, m_queue);
      connectivity->removeCellsToMaterial(mat_id, ids, m_queue);
    }
  }

  // Comme on a ajouté/supprimé des mailles matériau dans le milieu,
  // il faut transformer les mailles pures en mailles partielles (en cas
  // d'ajout) ou les mailles partielles en mailles pures (en cas de
  // suppression).
  info(4) << "Transform PartialPure for material name=" << true_mat->name();
  _switchCellsForMaterials(true_mat, orig_ids);
  info(4) << "Transform PartialPure for environment name=" << env->name();
  _switchCellsForEnvironments(env, orig_ids);

  // Si je suis mono-mat, alors mat->cells()<=>env->cells() et il ne faut
  // mettre à jour que l'un des deux groupes.
  bool need_update_env = (nb_mat != 1);

  if (is_add) {
    mat->cells().addItems(ids.smallView(), check_if_present);
    if (need_update_env)
      env->cells().addItems(ids.smallView(), check_if_present);
    _addItemsToEnvironment(true_env, true_mat, ids, need_update_env);
  }
  else {
    flagRemovedCells(ids, true);
    _removeItemsInGroup(mat->cells(), ids);
    if (need_update_env)
      _removeItemsInGroup(env->cells(), ids);
    _removeItemsFromEnvironment(true_env, true_mat, ids, need_update_env);
    // Remet \a removed_local_ids_filter à la valeur initiale pour les prochaines opérations
    flagRemovedCells(ids, false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule les mailles à transformer lorsqu'on modifie les mailles
 * d'un milieu.
 */
Int32 IncrementalComponentModifier::
_computeCellsToTransformForEnvironments(SmallSpan<const Int32> ids)
{
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  ConstArrayView<Int16> cells_nb_env = connectivity->cellsNbEnvironment();
  const bool is_add = m_work_info.isAdd();
  SmallSpan<bool> transformed_cells = m_work_info.transformedCells();

  const Int32 n = ids.size();
  auto command = makeCommand(m_queue);
  Accelerator::ReducerSum2<Int32> sum_transformed(command);
  command << RUNCOMMAND_LOOP1(iter, n, sum_transformed)
  {
    auto [i] = iter();
    Int32 lid = ids[i];
    bool do_transform = false;
    // En cas d'ajout, on passe de pure à partiel s'il y a plusieurs milieux.
    // En cas de suppression, on passe de partiel à pure si on est le seul milieu.
    if (is_add)
      do_transform = cells_nb_env[lid] > 1;
    else
      do_transform = cells_nb_env[lid] == 1;
    if (do_transform) {
      transformed_cells[lid] = do_transform;
      sum_transformed.combine(1);
    }
  };
  Int32 total_transformed = sum_transformed.reducedValue();
  return total_transformed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_computeItemsToAdd(ComponentItemListBuilder& list_builder, SmallSpan<const Int32> local_ids)
{
  SmallSpan<const bool> cells_is_partial = m_work_info.m_cells_is_partial;

  Accelerator::GenericFilterer filterer(m_queue);

  MeshMaterialVariableIndexer* var_indexer = list_builder.indexer();

  const Int32 nb_id = local_ids.size();

  SmallSpan<Int32> pure_indexes = list_builder.pureIndexes();
  SmallSpan<Int32> partial_indexes = list_builder.partialIndexes();
  SmallSpan<Int32> partial_local_ids = list_builder.partialLocalIds();
  Int32 nb_pure_added = 0;
  Int32 nb_partial_added = 0;
  Int32 index_in_partial = var_indexer->maxIndexInMultipleArray();

  // TODO: pour l'instant on remplit en deux fois mais il serait
  // possible de le faire en une seule fois en utilisation l'algorithme de Partition.
  // Il faudrait alors inverser les éléments de la deuxième liste pour avoir
  // le même ordre de parcours qu'avant le passage sur accélérateur.

  // Remplit la liste des mailles pures
  {
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      return !cells_is_partial[index];
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      Int32 local_id = local_ids[input_index];
      pure_indexes[output_index] = local_id;
    };
    filterer.applyWithIndex(nb_id, select_lambda, setter_lambda, A_FUNCINFO);
    nb_pure_added = filterer.nbOutputElement();
  }
  // Remplit la liste des mailles partielles
  {
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      return cells_is_partial[index];
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      Int32 local_id = local_ids[input_index];
      partial_indexes[output_index] = index_in_partial + output_index;
      partial_local_ids[output_index] = local_id;
    };
    filterer.applyWithIndex(nb_id, select_lambda, setter_lambda, A_FUNCINFO);
    nb_partial_added = filterer.nbOutputElement();
  }

  list_builder.resize(nb_pure_added, nb_partial_added);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
flagRemovedCells(SmallSpan<const Int32> local_ids, bool value_to_set)
{
  const Int32 nb_item = local_ids.size();
  SmallSpan<bool> removed_cells = m_work_info.removedCells();
  auto command = makeCommand(m_queue);

  ARCANE_CHECK_ACCESSIBLE_POINTER(m_queue, local_ids.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(m_queue, removed_cells.data());

  command << RUNCOMMAND_LOOP1(iter, nb_item)
  {
    auto [i] = iter();
    removed_cells[local_ids[i]] = value_to_set;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_resetTransformedCells(SmallSpan<const Int32> local_ids)
{
  const Int32 nb_item = local_ids.size();
  auto command = makeCommand(m_queue);
  SmallSpan<bool> transformed_cells = m_work_info.transformedCells();
  command << RUNCOMMAND_LOOP1(iter, nb_item)
  {
    auto [i] = iter();
    Int32 lid = local_ids[i];
    transformed_cells[lid] = false;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_removeItemsInGroup(ItemGroup cells, SmallSpan<const Int32> removed_ids)
{
  const Int32 nb_removed = removed_ids.size();
  if (nb_removed == 0)
    return;

  const bool do_old = false;
  if (do_old) {
    cells.removeItems(removed_ids.smallView(), false);
  }
  else {
    // Filtre les entités du groupe \a cells en considérant que
    // m_work_info.removedCells() vaut vrai pour les mailles qui
    // doivent être supprimées.
    ItemGroupImplInternal* impl_internal = cells._internalApi();
    SmallSpan<Int32> items_local_id(impl_internal->itemsLocalId());

    // Lors de l'application du filtre, le tableau d'entrée et de sortie
    // est le même (c'est normalement supporté par le GenericFilterer).
    SmallSpan<Int32> input_ids(items_local_id);
    SmallSpan<const bool> filtered_cells(m_work_info.removedCells());
    Accelerator::GenericFilterer filterer(m_queue);
    auto select_filter = [=] ARCCORE_HOST_DEVICE(Int32 local_id) -> bool {
      return !filtered_cells[local_id];
    };
    filterer.applyIf(input_ids, select_filter, A_FUNCINFO);

    Int32 current_nb_item = items_local_id.size();
    Int32 nb_remaining = filterer.nbOutputElement();
    if ((nb_remaining + nb_removed) != current_nb_item)
      ARCANE_FATAL("Internal error in removing nb_remaining={0} nb_removed={1} original_size={2}",
                   nb_remaining, nb_removed, current_nb_item);
    impl_internal->notifyDirectRemoveItems(removed_ids, nb_remaining);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue la copie entre les valeurs partielles et globales.
 *
 * Cette méthode permet de faire la copie en utilisant une seule RunCommand.
 */
void IncrementalComponentModifier::
_applyCopyBetweenPartialsAndGlobals(const CopyBetweenPartialAndGlobalArgs& args, RunQueue& queue)
{
  ARCANE_CHECK_POINTER(args.m_copy_data);

  auto output_indexes = args.m_local_ids;
  auto input_indexes = args.m_indexes_in_multiple;
  const bool is_global_to_partial = args.m_is_global_to_partial;

  if (is_global_to_partial)
    std::swap(output_indexes, input_indexes);
  SmallSpan<const CopyBetweenDataInfo> host_copy_data(m_work_info.m_host_variables_copy_data);
  SmallSpan<const CopyBetweenDataInfo> copy_data(m_work_info.m_variables_copy_data.to1DSmallSpan());
  const Int32 nb_value = input_indexes.size();
  if (nb_value != output_indexes.size())
    ARCANE_FATAL("input_indexes ({0}) and output_indexes ({1}) are different", nb_value, output_indexes);

  const Int32 nb_copy = copy_data.size();

  for (Int32 i = 0; i < nb_copy; ++i) {
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, host_copy_data[i].m_output.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, host_copy_data[i].m_input.data());
  }
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, input_indexes.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, output_indexes.data());

  // TODO: Gérer la copie de manière à pouvoir utiliser la coalescence
  // TODO: Faire des spécialisations si le dim2_size est de 4 ou 8
  // (voire un multiple) pour éviter la boucle interne.
  auto command = makeCommand(queue);
  command << RUNCOMMAND_LOOP2(iter, nb_copy, nb_value)
  {
    auto [icopy, i] = iter();
    auto input = copy_data[icopy].m_input;
    auto output = copy_data[icopy].m_output;
    Int32 dim2_size = copy_data[icopy].m_data_size;
    Int32 output_base = output_indexes[i] * dim2_size;
    Int32 input_base = input_indexes[i] * dim2_size;
    for (Int32 j = 0; j < dim2_size; ++j)
      output[output_base + j] = input[input_base + j];
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_applyInitializeWithZero(const InitializeWithZeroArgs& args)
{
  ARCANE_CHECK_POINTER(args.m_copy_data);

  const RunQueue& queue = args.m_queue;
  auto output_indexes = args.m_indexes_in_multiple;

  SmallSpan<const CopyBetweenDataInfo> host_copy_data(m_work_info.m_host_variables_copy_data);
  SmallSpan<const CopyBetweenDataInfo> copy_data(m_work_info.m_variables_copy_data.to1DSmallSpan());
  const Int32 nb_value = output_indexes.size();
  const Int32 nb_copy = copy_data.size();

  for (Int32 i = 0; i < nb_copy; ++i) {
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, host_copy_data[i].m_output.data());
  }
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, output_indexes.data());

  // TODO: Gérer la copie de manière à pouvoir utiliser la coalescence
  // TODO: Faire des spécialisations si le dim2_size est de 4 ou 8
  // (voire un multiple) pour éviter la boucle interne.
  auto command = makeCommand(queue);
  command << RUNCOMMAND_LOOP2(iter, nb_copy, nb_value)
  {
    auto [icopy, i] = iter();
    auto output = copy_data[icopy].m_output;
    Int32 dim2_size = copy_data[icopy].m_data_size;
    Int32 output_base = output_indexes[i] * dim2_size;
    for (Int32 j = 0; j < dim2_size; ++j)
      output[output_base + j] = {};
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue la copie des vues pour les variables.
 *
 * Cette méthode permet de faire en une seule RunCommand les copies entre
 * les vues CPU et accélérateurs des variables
 */
void IncrementalComponentModifier::
_applyCopyVariableViews(RunQueue& queue)
{
  SmallSpan<const CopyBetweenDataInfo> host_copy_data(m_work_info.m_host_variables_copy_data);
  SmallSpan<const CopyBetweenDataInfo> copy_data(m_work_info.m_variables_copy_data.to1DSmallSpan());

  const Int32 nb_copy = host_copy_data.size();
  if (nb_copy == 0)
    return;

  // Suppose que toutes les vues ont les mêmes tailles.
  // C'est le cas car les vues sont composées de 'ArrayView<>' et 'Array2View' et ces
  // deux classes ont la même taille.
  // TODO: il serait préférable de prendre le max des tailles et dans la commande
  // de ne faire la copie que si on ne dépasse pas la taille.
  Int32 nb_value = host_copy_data[0].m_input.size();

  for (Int32 i = 0; i < nb_copy; ++i) {
    const CopyBetweenDataInfo& h = host_copy_data[i];
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, h.m_output.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, h.m_input.data());
    if (h.m_input.size() != nb_value)
      ARCANE_FATAL("Invalid nb_value '{0} i={1} expected={2}", h.m_input.size(), nb_value);
  }

  auto command = makeCommand(queue);
  command << RUNCOMMAND_LOOP2(iter, nb_copy, nb_value)
  {
    auto [icopy, i] = iter();
    auto input = copy_data[icopy].m_input;
    auto output = copy_data[icopy].m_output;
    output[i] = input[i];
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
