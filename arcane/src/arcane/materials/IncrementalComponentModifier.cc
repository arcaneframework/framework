// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier.cc                             (C) 2000-2024 */
/*                                                                           */
/* Modification incrémentale des constituants.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/IncrementalComponentModifier.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FunctorUtils.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"

#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/MaterialModifierOperation.h"
#include "arcane/materials/internal/ConstituentConnectivityList.h"
#include "arcane/materials/internal/AllEnvData.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Filter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalComponentModifier::
IncrementalComponentModifier(AllEnvData* all_env_data)
: TraceAccessor(all_env_data->traceMng())
, m_all_env_data(all_env_data)
, m_material_mng(all_env_data->m_material_mng)
, m_copy_queue(makeQueue(m_material_mng->runner()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
initialize()
{
  Int32 max_local_id = m_material_mng->mesh()->cellFamily()->maxLocalId();
  m_work_info.initialize(max_local_id);
  m_work_info.is_verbose = traceMng()->verbosityLevel() >= 5;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
finalize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
apply(MaterialModifierOperation* operation)
{
  bool is_add = operation->isAdd();
  IMeshMaterial* mat = operation->material();
  ConstArrayView<Int32> orig_ids = operation->ids();
  ConstArrayView<Int32> ids = orig_ids;

  auto* true_mat = ARCANE_CHECK_POINTER(dynamic_cast<MeshMaterial*>(mat));

  info(4) << "Using optimisation updateMaterialDirect is_add=" << is_add
          << " mat=" << mat->name() << " nb_item=" << orig_ids.size();

  const IMeshEnvironment* env = mat->environment();
  MeshEnvironment* true_env = true_mat->trueEnvironment();
  Integer nb_mat = env->nbMaterial();

  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();

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
    cells_unchanged_in_env.clear();
    cells_unchanged_in_env.reserve(nb_id);
    cells_changed_in_env.clear();
    cells_changed_in_env.reserve(nb_id);
    cells_current_nb_material.resize(nb_id);
    const Int32 ref_nb_mat = is_add ? 0 : 1;
    const Int16 env_id = true_env->componentId();
    info(4) << "Using optimisation updateMaterialDirect is_add?=" << is_add;

    connectivity->fillCellsNbMaterial(ids, env_id, cells_current_nb_material.view(), m_copy_queue);

    for (Integer i = 0; i < nb_id; ++i) {
      Int32 lid = ids[i];
      Int16 current_cell_nb_mat = cells_current_nb_material[i];
      if (current_cell_nb_mat != ref_nb_mat) {
        cells_unchanged_in_env.add(lid);
      }
      else {
        cells_changed_in_env.add(lid);
      }
    }

    Integer nb_unchanged_in_env = cells_unchanged_in_env.size();
    info(4) << "Cells unchanged in environment n=" << nb_unchanged_in_env;

    Int16 mat_id = true_mat->componentId();
    if (is_add) {
      mat->cells().addItems(cells_unchanged_in_env);
      connectivity->addCellsToMaterial(mat_id, cells_unchanged_in_env);
      _addItemsToEnvironment(true_env, true_mat, cells_unchanged_in_env, false);
    }
    else {
      mat->cells().removeItems(cells_unchanged_in_env);
      connectivity->removeCellsToMaterial(mat_id, cells_unchanged_in_env);
      _removeItemsFromEnvironment(true_env, true_mat, cells_unchanged_in_env, false);
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
      connectivity->addCellsToEnvironment(env_id, ids);
      connectivity->addCellsToMaterial(mat_id, ids);
    }
    else {
      connectivity->removeCellsToEnvironment(env_id, ids);
      connectivity->removeCellsToMaterial(mat_id, ids);
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
    mat->cells().addItems(ids);
    if (need_update_env)
      env->cells().addItems(ids);
    _addItemsToEnvironment(true_env, true_mat, ids, need_update_env);
  }
  else {
    mat->cells().removeItems(ids);
    if (need_update_env)
      env->cells().removeItems(ids);
    _removeItemsFromEnvironment(true_env, true_mat, ids, need_update_env);
  }
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
                         ConstArrayView<Int32> ids)
{
  const bool is_add = m_work_info.isAdd();
  const bool is_device = isAcceleratorPolicy(m_copy_queue.executionPolicy());

  for (MeshEnvironment* true_env : m_material_mng->trueEnvironments()) {
    for (MeshMaterial* mat : true_env->trueMaterials()) {
      // Ne traite pas le matériau en cours de modification.
      if (mat == modified_mat)
        continue;

      if (!is_device) {
        m_work_info.pure_local_ids.clearHost();
        m_work_info.partial_indexes.clearHost();
      }

      MeshMaterialVariableIndexer* indexer = mat->variableIndexer();

      info(4) << "TransformCells (V3) is_add?=" << is_add << " indexer=" << indexer->name();

      _computeCellsToTransformForMaterial(mat, ids);
      indexer->transformCellsV2(m_work_info, m_copy_queue);
      _resetTransformedCells(ids);

      auto pure_local_ids = m_work_info.pure_local_ids.view(is_device);
      auto partial_indexes = m_work_info.partial_indexes.view(is_device);

      info(4) << "NB_MAT_TRANSFORM pure=" << pure_local_ids.size()
              << " partial=" << partial_indexes.size() << " name=" << mat->name()
              << " is_device?=" << is_device;

      MeshVariableCopyBetweenPartialAndGlobalArgs args(indexer->index(),
                                                       pure_local_ids,
                                                       partial_indexes,
                                                       &m_copy_queue);
      m_all_env_data->_copyBetweenPartialsAndGlobals(args, is_add);
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
                            ConstArrayView<Int32> ids)
{
  const bool is_add = m_work_info.isAdd();
  const bool is_device = m_copy_queue.isAcceleratorPolicy();

  // Ne copie pas les valeurs partielles des milieux vers les valeurs globales
  // en cas de suppression de mailles car cela sera fait avec la valeur matériau
  // correspondante. Cela permet d'avoir le même comportement que sans
  // optimisation. Ce n'est pas actif par défaut pour compatibilité avec l'existant.
  const bool is_copy = is_add || !(m_material_mng->isUseMaterialValueWhenRemovingPartialValue());

  for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    // Ne traite pas le milieu en cours de modification.
    if (env == modified_env)
      continue;

    if (!is_device) {
      m_work_info.pure_local_ids.clearHost();
      m_work_info.partial_indexes.clearHost();
    }

    MeshMaterialVariableIndexer* indexer = env->variableIndexer();

    info(4) << "TransformCells (V2) is_add?=" << is_add << " indexer=" << indexer->name();

    _computeCellsToTransformForEnvironments(ids);
    indexer->transformCellsV2(m_work_info, m_copy_queue);
    _resetTransformedCells(ids);

    info(4) << "NB_ENV_TRANSFORM=" << m_work_info.pure_local_ids.size()
            << " name=" << env->name();

    SmallSpan<const Int32> pure_local_ids = m_work_info.pure_local_ids.view(is_device);
    SmallSpan<const Int32> partial_indexes = m_work_info.partial_indexes.view(is_device);

    if (is_copy) {
      MeshVariableCopyBetweenPartialAndGlobalArgs copy_args(indexer->index(), pure_local_ids,
                                                            partial_indexes, &m_copy_queue);
      m_all_env_data->_copyBetweenPartialsAndGlobals(copy_args, is_add);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule les mailles à transformer pour le matériau \at mat.
 */
void IncrementalComponentModifier::
_computeCellsToTransformForMaterial(const MeshMaterial* mat, ConstArrayView<Int32> ids)
{
  const MeshEnvironment* env = mat->trueEnvironment();
  const Int16 env_id = env->componentId();
  bool is_add = m_work_info.isAdd();

  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  SmallSpan<bool> transformed_cells = m_work_info.transformedCells();
  const bool do_new = true;
  if (do_new)
    connectivity->fillCellsToTransform(ids, env_id, transformed_cells, is_add, m_copy_queue);
  else {
    ConstArrayView<Int16> cells_nb_env = connectivity->cellsNbEnvironment();

    for (Int32 local_id : ids) {
      bool do_transform = false;
      CellLocalId cell_id(local_id);
      // En cas d'ajout on passe de pure à partiel s'il y a plusieurs milieux ou
      // plusieurs matériaux dans le milieu.
      // En cas de supression, on passe de partiel à pure si on est le seul matériau
      // et le seul milieu.
      const Int16 nb_env = cells_nb_env[local_id];
      if (is_add) {
        do_transform = (nb_env > 1);
        if (!do_transform)
          do_transform = connectivity->cellNbMaterial(cell_id, env_id) > 1;
      }
      else {
        do_transform = (nb_env == 1);
        if (do_transform)
          do_transform = connectivity->cellNbMaterial(cell_id, env_id) == 1;
      }
      m_work_info.setTransformedCell(cell_id, do_transform);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule les mailles à transformer lorsqu'on modifie les mailles
 * d'un milieu.
 */
void IncrementalComponentModifier::
_computeCellsToTransformForEnvironments(ConstArrayView<Int32> ids)
{
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  ConstArrayView<Int16> cells_nb_env = connectivity->cellsNbEnvironment();
  const bool is_add = m_work_info.isAdd();
  SmallSpan<bool> transformed_cells = m_work_info.transformedCells();

  const Int32 n = ids.size();
  auto command = makeCommand(m_copy_queue);
  command << RUNCOMMAND_LOOP1(iter, n)
  {
    auto [i] = iter();
    Int32 lid = ids[i];
    bool do_transform = false;
    // En cas d'ajout on passe de pure à partiel s'il y a plusieurs milieux.
    // En cas de supression, on passe de partiel à pure si on est le seul milieu.
    if (is_add)
      do_transform = cells_nb_env[lid] > 1;
    else
      do_transform = cells_nb_env[lid] == 1;
    transformed_cells[lid] = do_transform;
  };
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
                            Int32ConstArrayView local_ids, bool update_env_indexer)
{
  info(4) << "MeshEnvironment::removeItemsDirect mat=" << mat->name();

  Int32 nb_to_remove = local_ids.size();

  // Positionne le filtre des mailles supprimées.
  setRemovedCells(local_ids, true);

  // TODO: à faire dans finialize()
  env->addToTotalNbCellMat(-nb_to_remove);

  mat->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove, m_copy_queue);

  if (update_env_indexer) {
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    env->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove, m_copy_queue);
  }

  // Remet \a removed_local_ids_filter à la valeur initiale pour
  // les prochaines opérations
  setRemovedCells(local_ids, false);
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
                       Int32ConstArrayView local_ids, bool update_env_indexer)
{
  info(4) << "MeshEnvironment::addItemsDirect" << " mat=" << mat->name();

  MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
  const Int32 nb_to_add = local_ids.size();

  // Met à jour le nombre de matériaux par maille et le nombre total de mailles matériaux.
  env->addToTotalNbCellMat(nb_to_add);

  const Int16 env_id = env->componentId();
  m_work_info.m_cells_is_partial.resize(nb_to_add);
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  connectivity->fillCellsIsPartial(local_ids, env_id, m_work_info.m_cells_is_partial.view(), m_copy_queue);

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
  ComponentItemListBuilder list_builder(var_indexer);
  const Int32 n = local_ids.size();
  list_builder.preAllocate(n);

  SmallSpan<MatVarIndex> pure_matvar_indexes = list_builder.pureMatVarIndexes();
  SmallSpan<MatVarIndex> partial_matvar_indexes = list_builder.partialMatVarIndexes();
  SmallSpan<Int32> partial_local_ids = list_builder.partialLocalIds();
  Int32 nb_pure_added = 0;
  Int32 nb_partial_added = 0;
  Int32 index_in_partial = var_indexer->maxIndexInMultipleArray();

  const Int32 component_index = var_indexer->index() + 1;

  SmallSpan<const bool> cells_is_partial = m_work_info.m_cells_is_partial.view();

  Accelerator::GenericFilterer filterer(&m_copy_queue);

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
      pure_matvar_indexes[output_index] = MatVarIndex(0, local_id);
    };
    filterer.applyWithIndex(n, select_lambda, setter_lambda, A_FUNCINFO);
    nb_pure_added = filterer.nbOutputElement();
  }
  // Remplit la liste des mailles partielles
  {
    auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      return cells_is_partial[index];
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      Int32 local_id = local_ids[input_index];
      partial_matvar_indexes[output_index] = MatVarIndex(component_index, index_in_partial + output_index);
      partial_local_ids[output_index] = local_id;
    };
    filterer.applyWithIndex(n, select_lambda, setter_lambda, A_FUNCINFO);
    nb_partial_added = filterer.nbOutputElement();
  }

  list_builder.resize(nb_pure_added, nb_partial_added);

  if (traceMng()->verbosityLevel() >= 5)
    info() << "ADD_MATITEM_TO_INDEXER component=" << var_indexer->name()
           << " nb_pure=" << list_builder.pureMatVarIndexes().size()
           << " nb_partial=" << list_builder.partialMatVarIndexes().size()
           << "\n pure=(" << list_builder.pureMatVarIndexes() << ")"
           << "\n partial=(" << list_builder.partialMatVarIndexes() << ")";

  // TODO: lors de cet appel, on connait le max de \a index_in_partial donc
  // on peut éviter de faire une réduction pour le recalculer.

  var_indexer->endUpdateAdd(list_builder, m_copy_queue);

  // Maintenant que les nouveaux MatVar sont créés, il faut les
  // initialiser avec les bonnes valeurs.
  {
    IMeshMaterialMng* mm = m_material_mng;
    auto func = [&](IMeshMaterialVariable* mv) {
      mv->_internalApi()->initializeNewItems(list_builder, m_copy_queue);
    };
    functor::apply(mm, &IMeshMaterialMng::visitVariables, func);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
setRemovedCells(SmallSpan<const Int32> local_ids, bool value_to_set)
{
  const Int32 nb_item = local_ids.size();
  SmallSpan<bool> removed_cells = m_work_info.removedCells();
  auto command = makeCommand(m_copy_queue);

  ARCANE_CHECK_ACCESSIBLE_POINTER(m_copy_queue, local_ids.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(m_copy_queue, removed_cells.data());

  command << RUNCOMMAND_LOOP1(iter, nb_item)
  {
    auto [i] = iter();
    removed_cells[local_ids[i]] = value_to_set;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_resetTransformedCells(ConstArrayView<Int32> local_ids)
{
  const Int32 nb_item = local_ids.size();
  auto command = makeCommand(m_copy_queue);
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

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
