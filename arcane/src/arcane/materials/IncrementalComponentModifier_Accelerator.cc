// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier_Accelerator.cc                 (C) 2000-2024 */
/*                                                                           */
/* Incremental modification of constituents.                                 */
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

  // Fills the arrays indicating if a constituent is affected by
  // the current modification. If not, we can avoid testing it
  // in the constituent loop.
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

    // If it is possible to have multiple materials per environment, it must be handled
    // for each cell if the environment changes following the addition/removal of material.
    // The two cases are:
    // - in case of addition, the environment changes for a cell if there was no
    //   material before. In this case, the environment is added to the cell.
    // - in case of removal, the environment changes in the cell if there was
    //   only 1 material before. In this case, the environment is removed from the cell.

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

    // Takes for \a ids only the list of cells
    // that did not yet belong to the environment in which we
    // are adding the material.
    ids = cells_changed_in_env.view();
  }

  // Updates the number of environments and materials for each cell.
  // NOTE: the operation must first be performed on the environments before
  // the materials.
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

  // Since we have added/removed material cells in the environment,
  // we must transform pure cells into partial cells (in case
  // of addition) or partial cells into pure cells (in case of
  // removal).
  info(4) << "Transform PartialPure for material name=" << true_mat->name();
  _switchCellsForMaterials(true_mat, orig_ids);
  info(4) << "Transform PartialPure for environment name=" << env->name();
  _switchCellsForEnvironments(env, orig_ids);

  // If I am mono-mat, then mat->cells()<=>env->cells() and only one
  // of the two groups needs to be updated.
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
    // Reset \a removed_local_ids_filter to the initial value for subsequent operations
    flagRemovedCells(ids, false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the cells to transform when modifying the cells
 * of an environment.
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
    // In case of addition, we switch from pure to partial if there are multiple environments.
    // In case of removal, we switch from partial to pure if we are the only environment.
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

  // TODO: for now we fill in two passes, but it would be possible to do it in one go using the Partition algorithm.
  // We would then need to reverse the elements of the second list to have
  // the same traversal order as before going to the accelerator.

  // Fills the list of pure cells
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
  // Fills the list of partial cells
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
    // Filters the entities of the group \a cells considering that
    // m_work_info.removedCells() is true for the cells that
    // must be deleted.
    ItemGroupImplInternal* impl_internal = cells._internalApi();
    SmallSpan<Int32> items_local_id(impl_internal->itemsLocalId());

    // During the application of the filter, the input and output array
    // is the same (this is normally supported by the GenericFilterer).
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
 * \brief Performs the copy between partial and global values.
 *
 * This method allows the copy to be done using a single RunCommand.
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

  // TODO: Handle the copy in a way that allows using coalescence
  // TODO: Make specializations if dim2_size is 4 or 8
  // (or a multiple) to avoid the internal loop.
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

  // TODO: Handle the copy in a way that allows using coalescence
  // TODO: Make specializations if dim2_size is 4 or 8
  // (or a multiple) to avoid the internal loop.
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
 * \brief Performs the copy of views for variables.
 *
 * This method allows the copies between
 * the CPU and accelerator views of the variables to be done in a single RunCommand
 */
void IncrementalComponentModifier::
_applyCopyVariableViews(RunQueue& queue)
{
  SmallSpan<const CopyBetweenDataInfo> host_copy_data(m_work_info.m_host_variables_copy_data);
  SmallSpan<const CopyBetweenDataInfo> copy_data(m_work_info.m_variables_copy_data.to1DSmallSpan());

  const Int32 nb_copy = host_copy_data.size();
  if (nb_copy == 0)
    return;

  // Assumes that all views have the same sizes.
  // This is the case because the views are composed of 'ArrayView<>' and 'Array2View' and these
  // two classes have the same size.
  // TODO: it would be preferable to take the max of the sizes and in the command
  // only perform the copy if we do not exceed the size.
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
