// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier.cc                             (C) 2000-2025 */
/*                                                                           */
/* Incremental modification of constituents.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/IncrementalComponentModifier.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/materials/IMeshMaterialVariable.h"

#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/ConstituentConnectivityList.h"
#include "arcane/materials/internal/AllEnvData.h"

#include "arcane/accelerator/core/ProfileRegion.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalComponentModifier::
IncrementalComponentModifier(AllEnvData* all_env_data, const RunQueue& queue)
: TraceAccessor(all_env_data->traceMng())
, m_all_env_data(all_env_data)
, m_material_mng(all_env_data->m_material_mng)
, m_work_info(queue.allocationOptions(), queue.memoryRessource())
, m_queue(queue)
{
  // 0 if using typed copy (historical mode) and one command per variable
  // 1 if using generic copy and one command per variable
  // 2 if using generic copy and one command for all variables
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL", true)) {
    m_use_generic_copy_between_pure_and_partial = v.value();
  }
  else {
    // By default on an accelerator and in multi-threading, we use the copy
    // with a single queue, as it is the most performant mechanism.
    if (queue.executionPolicy() != Accelerator::eExecutionPolicy::Sequential)
      m_use_generic_copy_between_pure_and_partial = 2;
  }
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_FORCE_MULTIPLE_COMMAND_FOR_MATERIAL_RESIZE", true)) {
    m_force_multiple_command_for_resize = (v.value());
    info() << "Force using multiple command for resize = " << m_force_multiple_command_for_resize;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
initialize(bool is_debug)
{
  m_is_debug = is_debug;
  Int32 max_local_id = m_material_mng->mesh()->cellFamily()->maxLocalId();
  Int32 nb_mat = m_material_mng->materials().size();
  Int32 nb_env = m_material_mng->environments().size();
  m_work_info.initialize(max_local_id, nb_mat, nb_env, m_queue);
  m_work_info.is_verbose = is_debug || (traceMng()->verbosityLevel() >= 5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
finalize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Transforms entities for an environment.
 *
 * Iterates over the environment \a env and
 * converts pure meshes to partial meshes or
 * vice versa. After conversion, the values corresponding to the
 * modified meshes are updated for each variable.
 *
 * If \a is_add is true, it transforms from pure to partial
 * (material addition); otherwise, it transforms from partial to pure
 * (material removal)
 */
void IncrementalComponentModifier::
_switchCellsForMaterials(const MeshMaterial* modified_mat,
                         SmallSpan<const Int32> ids)
{
  const bool is_add = m_work_info.isAdd();
  const bool is_device = m_queue.isAcceleratorPolicy();
  SmallSpan<const bool> is_materials_modified = m_work_info.m_is_materials_modified.view(false);

  for (MeshEnvironment* true_env : m_material_mng->trueEnvironments()) {
    for (MeshMaterial* mat : true_env->trueMaterials()) {
      // Do not process the material currently being modified.
      if (mat == modified_mat)
        continue;

      if (!is_materials_modified[mat->id()])
        continue;

      if (!is_device) {
        m_work_info.pure_local_ids.clearHost();
        m_work_info.partial_indexes.clearHost();
      }

      MeshMaterialVariableIndexer* indexer = mat->variableIndexer();

      info(4) << "MatTransformCells is_add?=" << is_add << " indexer=" << indexer->name()
              << " mat_id=" << mat->id();

      Int32 nb_transformed = _computeCellsToTransformForMaterial(mat, ids);
      info(4) << "nb_transformed=" << nb_transformed;
      if (nb_transformed == 0)
        continue;
      indexer->transformCells(m_work_info, m_queue, false);
      _resetTransformedCells(ids);

      auto pure_local_ids = m_work_info.pure_local_ids.view(is_device);
      auto partial_indexes = m_work_info.partial_indexes.view(is_device);

      Int32 nb_pure = pure_local_ids.size();
      Int32 nb_partial = partial_indexes.size();
      info(4) << "NB_MAT_TRANSFORM pure=" << nb_pure
              << " partial=" << nb_partial << " name=" << mat->name()
              << " is_device?=" << is_device
              << " is_modified?=" << is_materials_modified[mat->id()];

      CopyBetweenPartialAndGlobalArgs args(indexer->index(), pure_local_ids,
                                           partial_indexes,
                                           m_do_copy_between_partial_and_pure,
                                           is_add,
                                           m_queue);
      _copyBetweenPartialsAndGlobals(args);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Transforms entities for environments.
 *
 * Iterates over environments, except the modified environment \a modified_env, and
 * for each one converts pure meshes to partial meshes or
 * vice versa. After conversion, the values corresponding to the
 * modified meshes are updated for each variable.
 *
 * If \a is_add is true, it transforms from pure to partial
 * (in the case of material addition); otherwise, it transforms from partial to pure
 * (in the case of material removal)
 */
void IncrementalComponentModifier::
_switchCellsForEnvironments(const IMeshEnvironment* modified_env,
                            SmallSpan<const Int32> ids)
{
  const bool is_add = m_work_info.isAdd();
  const bool is_device = m_queue.isAcceleratorPolicy();
  SmallSpan<const bool> is_environments_modified = m_work_info.m_is_environments_modified.view(false);

  // Do not copy partial values from environments to global values
  // in case of mesh removal, because this will be done with the material value
  // corresponding to it. This allows the same behavior as without
  // optimization. This is not active by default for compatibility with existing code.
  const bool is_copy = is_add || !(m_material_mng->isUseMaterialValueWhenRemovingPartialValue());

  Int32 nb_transformed = _computeCellsToTransformForEnvironments(ids);
  info(4) << "Compute Cells for environments nb_transformed=" << nb_transformed;
  if (nb_transformed == 0)
    return;

  for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    // Do not process the environment currently being modified.
    if (env == modified_env)
      continue;
    // If I am mono-material, the indexer update was done by the material
    if (env->isMonoMaterial())
      continue;

    const Int32 env_id = env->id();

    if (!is_environments_modified[env_id])
      continue;

    if (!is_device) {
      m_work_info.pure_local_ids.clearHost();
      m_work_info.partial_indexes.clearHost();
    }

    MeshMaterialVariableIndexer* indexer = env->variableIndexer();

    info(4) << "EnvTransformCells is_add?=" << is_add
            << " env_id=" << env_id
            << " indexer=" << indexer->name() << " nb_item=" << ids.size();

    indexer->transformCells(m_work_info, m_queue, true);

    SmallSpan<const Int32> pure_local_ids = m_work_info.pure_local_ids.view(is_device);
    SmallSpan<const Int32> partial_indexes = m_work_info.partial_indexes.view(is_device);
    const Int32 nb_pure = pure_local_ids.size();

    info(4) << "NB_ENV_TRANSFORM nb_pure=" << nb_pure << " name=" << env->name()
            << " is_modified=" << is_environments_modified[env_id];

    if (is_copy) {
      CopyBetweenPartialAndGlobalArgs copy_args(indexer->index(), pure_local_ids,
                                                partial_indexes,
                                                m_do_copy_between_partial_and_pure, is_add,
                                                m_queue);
      _copyBetweenPartialsAndGlobals(copy_args);
    }
  }

  _resetTransformedCells(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the meshes to transform for material \at mat.
 */
Int32 IncrementalComponentModifier::
_computeCellsToTransformForMaterial(const MeshMaterial* mat, SmallSpan<const Int32> ids)
{
  const MeshEnvironment* env = mat->trueEnvironment();
  const Int16 env_id = env->componentId();
  bool is_add = m_work_info.isAdd();

  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  SmallSpan<bool> transformed_cells = m_work_info.transformedCells();
  return connectivity->fillCellsToTransform(ids, env_id, transformed_cells, is_add, m_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Removes meshes of a material from the environment.
 *
 * Removes the meshes provided by \a local_ids from material \a mat
 * in the environment. The material indexer is updated, and if \a update_env_indexer
 * is true, the environment indexer is also updated (which means the environment disappears
 * from the meshes \a local_ids).
 *
 * TODO: optimize this by not iterating over all
 * materials of the environment (removed_local_ids_filter must be removed).
 * If we know the index of each mesh in the MatVarIndex
 * from the indexer, we can directly access it.
 */
void IncrementalComponentModifier::
_removeItemsFromEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                            SmallSpan<const Int32> local_ids, bool update_env_indexer)
{
  info(4) << "MeshEnvironment::removeItemsDirect mat=" << mat->name();

  Int32 nb_to_remove = local_ids.size();

  // TODO: to be done in finalize()
  env->addToTotalNbCellMat(-nb_to_remove);

  mat->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove, m_queue);

  if (update_env_indexer) {
    // Also updates the entities \a local_ids in the environment's indexer.
    // This is only possible if the number of environment materials
    // is greater than or equal to 2 (because otherwise the material and the environment
    // have the same indexer)
    env->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove, m_queue);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds the meshes of an environment material.
 *
 * Adds the meshes given by \a local_ids to the environment material \a mat
 * of the environment. The material indexer is updated, and if \a update_env_indexer
 * is true, the environment's indexer is also updated (meaning the environment appears
 * in the meshes \a local_ids).
 */
void IncrementalComponentModifier::
_addItemsToEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                       SmallSpan<const Int32> local_ids, bool update_env_indexer)
{
  info(4) << "MeshEnvironment::addItemsDirect"
          << " mat=" << mat->name();

  MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
  const Int32 nb_to_add = local_ids.size();

  // Updates the number of materials per mesh and the total number of material meshes.
  env->addToTotalNbCellMat(nb_to_add);

  const Int16 env_id = env->componentId();
  m_work_info.m_cells_is_partial.resize(nb_to_add);
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  connectivity->fillCellsIsPartial(local_ids, env_id, m_work_info.m_cells_is_partial.to1DSmallSpan(), m_queue);

  _addItemsToIndexer(var_indexer, local_ids);

  if (update_env_indexer) {
    // Also updates the entities \a local_ids in the environment's indexer.
    // This is only possible if the number of environment materials
    // is greater than or equal to 2 (because otherwise the material and the environment
    // have the same indexer)
    _addItemsToIndexer(env->variableIndexer(), local_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_addItemsToIndexer(MeshMaterialVariableIndexer* var_indexer,
                   SmallSpan<const Int32> local_ids)
{
  // TODO Keep the instance during all modifications
  ComponentItemListBuilder& list_builder = m_work_info.list_builder;
  list_builder.setIndexer(var_indexer);

  const Int32 nb_id = local_ids.size();
  list_builder.preAllocate(nb_id);

  _computeItemsToAdd(list_builder, local_ids);

  if (traceMng()->verbosityLevel() >= 5)
    info() << "ADD_MATITEM_TO_INDEXER component=" << var_indexer->name()
           << " nb_pure=" << list_builder.pureIndexes().size()
           << " nb_partial=" << list_builder.partialIndexes().size()
           << "\n pure=(" << list_builder.pureIndexes() << ")"
           << "\n partial=(" << list_builder.partialIndexes() << ")";

  // TODO: during this call, we know the max of \a index_in_partial so
  // we can avoid performing a reduction to recalculate it.

  var_indexer->endUpdateAdd(list_builder, m_queue);

  // Resizes the variables
  _resizeVariablesIndexer(var_indexer->index());

  // Now that the new MatVars are created, they must be
  // initialized with the correct values.
  if (m_do_init_new_items) {
    IMeshMaterialMng* mm = m_material_mng;
    bool init_with_zero = mm->isDataInitialisationWithZero();

    Accelerator::ProfileRegion ps(m_queue, "InitializeNewItems", 0xFFFF00);

    SmallSpan<Int32> partial_indexes = list_builder.partialIndexes();
    if (init_with_zero) {
      RunQueue::ScopedAsync sc(&m_queue);
      InitializeWithZeroArgs init_args(var_indexer->index(), partial_indexes, m_queue);

      bool do_one_command = (m_use_generic_copy_between_pure_and_partial == 2);
      UniqueArray<CopyBetweenDataInfo>& copy_data = m_work_info.m_host_variables_copy_data;
      if (do_one_command) {
        copy_data.clear();
        copy_data.reserve(m_material_mng->nbVariable());
        init_args.m_copy_data = &copy_data;
      }

      auto func_zero = [&](IMeshMaterialVariable* mv) {
        mv->_internalApi()->initializeNewItemsWithZero(init_args);
      };
      functor::apply(mm, &IMeshMaterialMng::visitVariables, func_zero);

      if (do_one_command) {
        MDSpan<CopyBetweenDataInfo, MDDim1> x(copy_data.data(), MDIndex<1>(copy_data.size()));
        m_work_info.m_variables_copy_data.copy(x, &m_queue);
        _applyInitializeWithZero(init_args);
      }
      m_queue.barrier();
    }
    else {
      SmallSpan<Int32> partial_local_ids = list_builder.partialLocalIds();

      CopyBetweenPartialAndGlobalArgs args(var_indexer->index(), partial_local_ids,
                                           partial_indexes, true, true, m_queue);
      _copyBetweenPartialsAndGlobals(args);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Resizes the variable index \a var_index.
 */
void IncrementalComponentModifier::
_resizeVariablesIndexer(Int32 var_index)
{
  Accelerator::ProfileRegion ps(m_queue, "ResizeVariableIndexer", 0xFF00FF);
  ResizeVariableIndexerArgs resize_args(var_index, m_queue);
  // Checks if we are only using one command for view copies.
  // For now (November 2024), we only use it by default if
  // we are on an accelerator.
  bool do_one_command = (m_use_generic_copy_between_pure_and_partial == 2);

  if (m_force_multiple_command_for_resize)
    do_one_command = false;

  UniqueArray<CopyBetweenDataInfo>& copy_data = m_work_info.m_host_variables_copy_data;
  if (do_one_command) {
    copy_data.clear();
    copy_data.reserve(m_material_mng->nbVariable());
    resize_args.m_copy_data = &copy_data;
  }

  if (m_force_multiple_command_for_resize) {
    // The multiple command mode is used to identify which variables
    // are still on CPU via PageFault triggering.
    // That is why we put the variable name in the profiling region
    // to get traces with 'nsys', for example. We also need to add
    // a barrier to serialize the operations.
    auto func2 = [&](IMeshMaterialVariable* mv) {
      Accelerator::ProfileRegion ps2(m_queue, String("Resize_") + mv->name());
      auto* mvi = mv->_internalApi();
      mvi->resizeForIndexer(resize_args);
      m_queue.barrier();
    };
    functor::apply(m_material_mng, &MeshMaterialMng::visitVariables, func2);
  }
  else {
    RunQueue::ScopedAsync sc(&m_queue);
    auto func1 = [&](IMeshMaterialVariable* mv) {
      auto* mvi = mv->_internalApi();
      mvi->resizeForIndexer(resize_args);
    };
    functor::apply(m_material_mng, &MeshMaterialMng::visitVariables, func1);
  }

  if (do_one_command) {
    // Copies 'copy_data' into the corresponding array for the eventual device.
    MDSpan<CopyBetweenDataInfo, MDDim1> x(copy_data.data(), MDIndex<1>(copy_data.size()));
    m_work_info.m_variables_copy_data.copy(x, &m_queue);
    _applyCopyVariableViews(m_queue);
  }

  m_queue.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Copies between partial and global values.
 *
 * If \a pure_to_partial is true, then we copy the global values
 * to the partial values; otherwise, we do the reverse.
 * (of material deletion)
 */
void IncrementalComponentModifier::
_copyBetweenPartialsAndGlobals(const CopyBetweenPartialAndGlobalArgs& args)
{
  if (args.m_local_ids.empty())
    return;
  const bool do_copy = args.m_do_copy_between_partial_and_pure;
  const bool is_add_operation = args.m_is_global_to_partial;
  RunQueue queue(args.m_queue);
  RunQueue::ScopedAsync sc(&queue);
  // Since we modified meshes, we must update the corresponding values
  // for each variable.
  //info(4) << "NB_TRANSFORM=" << nb_transform << " name=" << e->name();
  //Integer indexer_index = indexer->index();

  Accelerator::RunQueuePool& queue_pool = m_material_mng->_internalApi()->asyncRunQueuePool();

  // Resizes the variables if necessary
  if (is_add_operation) {
    _resizeVariablesIndexer(args.m_var_index);
  }

  if (do_copy) {
    bool do_one_command = (m_use_generic_copy_between_pure_and_partial == 2);
    UniqueArray<CopyBetweenDataInfo>& copy_data = m_work_info.m_host_variables_copy_data;
    copy_data.clear();
    copy_data.reserve(m_material_mng->nbVariable());

    Int32 index = 0;
    CopyBetweenPartialAndGlobalArgs args2(args);
    args2.m_use_generic_copy = (m_use_generic_copy_between_pure_and_partial >= 1);
    if (do_one_command)
      args2.m_copy_data = &copy_data;
    auto func2 = [&](IMeshMaterialVariable* mv) {
      auto* mvi = mv->_internalApi();
      if (!do_one_command)
        args2.m_queue = queue_pool[index];
      mvi->copyBetweenPartialAndGlobal(args2);
      ++index;
    };
    functor::apply(m_material_mng, &MeshMaterialMng::visitVariables, func2);
    if (do_one_command) {
      // Copies 'copy_data' into the corresponding array for the eventual device.
      MDSpan<CopyBetweenDataInfo, MDDim1> x(copy_data.data(), MDIndex<1>(copy_data.size()));
      m_work_info.m_variables_copy_data.copy(x, &queue);
      _applyCopyBetweenPartialsAndGlobals(args2, queue);
    }
    else
      queue_pool.barrier();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
