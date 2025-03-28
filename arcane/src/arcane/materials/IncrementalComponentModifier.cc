// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier.cc                             (C) 2000-2025 */
/*                                                                           */
/* Modification incrémentale des constituants.                               */
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
  // 0 si on utilise la copie typée (mode historique) et une commande par variable
  // 1 si on utilise la copie générique et une commande par variable
  // 2 si on utilise la copie générique et une commande pour toutes les variables
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_GENERIC_COPY_BETWEEN_PURE_AND_PARTIAL", true)) {
    m_use_generic_copy_between_pure_and_partial = v.value();
  }
  else {
    // Par défaut sur un accélérateur et en multi-threading, on utilise la copie
    // avec une seule file, car c'est le mécanisme le plus performant.
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
 * \brief Transforme les entités pour un milieu.
 *
 * Parcours le milieux \a env et
 * convertie les mailles pures en mailles partielles ou
 * inversement. Après conversion, les valeurs correspondantes aux
 * mailles modifiées sont mises à jour pour chaque variable.
 *
 * Si \a is_add est vrai, alors on transforme de pure en partiel
 * (ajout de matériau) sinon on transforme de partiel en pure
 * (suppression d'un matériau)
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
      // Ne traite pas le matériau en cours de modification.
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
 * \brief Transforme les entités pour les milieux.
 *
 * Parcours les milieux, sauf le milieu modifié \a modified_env et
 * pour chacun convertie les mailles pures en mailles partielles ou
 * inversement. Après conversion, les valeurs correspondantes aux
 * mailles modifiées sont mises à jour pour chaque variable.
 *
 * Si \a is_add est vrai, alors on transforme de pure en partiel
 * (dans le cas d'ajout de matériau) sinon on transforme de partiel
 * en pure (dans le cas de suppression d'un matériau)
 */
void IncrementalComponentModifier::
_switchCellsForEnvironments(const IMeshEnvironment* modified_env,
                            SmallSpan<const Int32> ids)
{
  const bool is_add = m_work_info.isAdd();
  const bool is_device = m_queue.isAcceleratorPolicy();
  SmallSpan<const bool> is_environments_modified = m_work_info.m_is_environments_modified.view(false);

  // Ne copie pas les valeurs partielles des milieux vers les valeurs globales
  // en cas de suppression de mailles, car cela sera fait avec la valeur matériau
  // correspondante. Cela permet d'avoir le même comportement que sans
  // optimisation. Ce n'est pas actif par défaut pour compatibilité avec l'existant.
  const bool is_copy = is_add || !(m_material_mng->isUseMaterialValueWhenRemovingPartialValue());

  Int32 nb_transformed = _computeCellsToTransformForEnvironments(ids);
  info(4) << "Compute Cells for environments nb_transformed=" << nb_transformed;
  if (nb_transformed == 0)
    return;

  for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    // Ne traite pas le milieu en cours de modification.
    if (env == modified_env)
      continue;
    // Si je suis mono matériau, la mise à jour de l'indexeur a été faite par le matériau
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
 * \brief Calcule les mailles à transformer pour le matériau \at mat.
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
 * \brief Supprime les mailles d'un matériau du milieu.
 *
 * Supprime les mailles données par \a local_ids du matériau \a mat
 * du milieu. L'indexeur du matériau est mis à jour et si \a update_env_indexer
 * est vrai, celui du milieu aussi (ce qui signifie que le milieu disparait
 * des mailles \a local_ids).
 *
 * TODO: optimiser cela en ne parcourant pas toutes les mailles
 * matériaux du milieu (il faut supprimer removed_local_ids_filter).
 * Si on connait l'indice de chaque maille dans la liste des MatVarIndex
 * de l'indexeur, on peut directement taper dedans.
 */
void IncrementalComponentModifier::
_removeItemsFromEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                            SmallSpan<const Int32> local_ids, bool update_env_indexer)
{
  info(4) << "MeshEnvironment::removeItemsDirect mat=" << mat->name();

  Int32 nb_to_remove = local_ids.size();

  // TODO: à faire dans finalize()
  env->addToTotalNbCellMat(-nb_to_remove);

  mat->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove, m_queue);

  if (update_env_indexer) {
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    env->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove, m_queue);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute les mailles d'un matériau du milieu.
 *
 * Ajoute les mailles données par \a local_ids au matériau \a mat
 * du milieu. L'indexeur du matériau est mis à jour et si \a update_env_indexer
 * est vrai, celui du milieu aussi (ce qui signifie que le milieu apparait
 * dans les mailles \a local_ids).
 */
void IncrementalComponentModifier::
_addItemsToEnvironment(MeshEnvironment* env, MeshMaterial* mat,
                       SmallSpan<const Int32> local_ids, bool update_env_indexer)
{
  info(4) << "MeshEnvironment::addItemsDirect"
          << " mat=" << mat->name();

  MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
  const Int32 nb_to_add = local_ids.size();

  // Met à jour le nombre de matériaux par maille et le nombre total de mailles matériaux.
  env->addToTotalNbCellMat(nb_to_add);

  const Int16 env_id = env->componentId();
  m_work_info.m_cells_is_partial.resize(nb_to_add);
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  connectivity->fillCellsIsPartial(local_ids, env_id, m_work_info.m_cells_is_partial.to1DSmallSpan(), m_queue);

  _addItemsToIndexer(var_indexer, local_ids);

  if (update_env_indexer) {
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    _addItemsToIndexer(env->variableIndexer(), local_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_addItemsToIndexer(MeshMaterialVariableIndexer* var_indexer,
                   SmallSpan<const Int32> local_ids)
{
  // TODO Conserver l'instance au cours de toutes modifications
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

  // TODO: lors de cet appel, on connait le max de \a index_in_partial donc
  // on peut éviter de faire une réduction pour le recalculer.

  var_indexer->endUpdateAdd(list_builder, m_queue);

  // Redimensionne les variables
  _resizeVariablesIndexer(var_indexer->index());

  // Maintenant que les nouveaux MatVar sont créés, il faut les
  // initialiser avec les bonnes valeurs.
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

      if (do_one_command){
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
 * \brief Redimensionne l'index \a var_index des variables.
 */
void IncrementalComponentModifier::
_resizeVariablesIndexer(Int32 var_index)
{
  Accelerator::ProfileRegion ps(m_queue, "ResizeVariableIndexer", 0xFF00FF);
  ResizeVariableIndexerArgs resize_args(var_index, m_queue);
  // Regarde si on n'utilise qu'une seule commande pour les copies des vues.
  // Pour l'instant (novembre 2024) on ne l'utilise par défaut que si
  // on est sur accélérateur.
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
    // Le mode de commandes multiples sert à identifier quelles variables
    // sont encore sur CPU via le déclenchement de PageFault.
    // C'est pour cela qu'on met le nom de la variable dans la région de profiling
    // pour avoir les traces avec 'nsys' par exemple. Il faut aussi ajouter
    // une barrière pour sérialiser les opérations.
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
    // Copie 'copy_data' dans le tableau correspondant pour le device éventuel.
    MDSpan<CopyBetweenDataInfo, MDDim1> x(copy_data.data(), MDIndex<1>(copy_data.size()));
    m_work_info.m_variables_copy_data.copy(x, &m_queue);
    _applyCopyVariableViews(m_queue);
  }

  m_queue.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Copie entre les valeurs partielles et les valeurs globales.
 *
 * Si \a pure_to_partial est vrai, alors on copie les valeurs globales
 * vers les valeurs partielles, sinon on fait l'inverse.
 * de suppression d'un matériau)
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
  // Comme on a modifié des mailles, il faut mettre à jour les valeurs
  // correspondantes pour chaque variable.
  //info(4) << "NB_TRANSFORM=" << nb_transform << " name=" << e->name();
  //Integer indexer_index = indexer->index();

  Accelerator::RunQueuePool& queue_pool = m_material_mng->_internalApi()->asyncRunQueuePool();

  // Redimensionne les variables si nécessaire
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
      // Copie 'copy_data' dans le tableau correspondant pour le device éventuel.
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
