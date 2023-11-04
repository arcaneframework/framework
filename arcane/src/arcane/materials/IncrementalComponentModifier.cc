// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalComponentModifier.cc                             (C) 2000-2023 */
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
  // Met à jour les variables contenant le nombre de milieux et de matériaux
  // par milieu en fonction des valeurs de ConstituentConnectivityList.

  // TODO: ne le faire que pour les mailles dont les matériaux ont été modifiés
  CellGroup all_cells = m_material_mng->mesh()->allCells();
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  VariableCellInt32& cells_nb_env = m_all_env_data->m_nb_env_per_cell;
  ConstArrayView<Int16> incremental_cells_nb_env = connectivity->cellsNbEnvironment();
  ENUMERATE_(Cell,icell,all_cells){
    cells_nb_env[icell] = incremental_cells_nb_env[icell.itemLocalId()];
  }

  // Met à jour le nombre de matériaux par milieu
  // TODO: Faire cela en une passe
  for (MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    VariableCellInt32& cells_nb_mat = env->m_nb_mat_per_cell;
    Int16 env_id = env->componentId();
    ENUMERATE_(Cell,icell,all_cells){
      cells_nb_mat[icell] = connectivity->cellNbMaterial(icell, env_id);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
apply(MaterialModifierOperation* operation)
{
  bool is_add = operation->isAdd();
  IMeshMaterial* mat = operation->material();
  Int32ConstArrayView ids = operation->ids();

  auto* true_mat = ARCANE_CHECK_POINTER(dynamic_cast<MeshMaterial*>(mat));

  info(4) << "Using optimisation updateMaterialDirect operation=" << operation;

  const IMeshEnvironment* env = mat->environment();
  MeshEnvironment* true_env = true_mat->trueEnvironment();
  Integer nb_mat = env->nbMaterial();

  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();

  Int32UniqueArray cells_changed_in_env;

  if (nb_mat != 1) {

    // S'il est possible d'avoir plusieurs matériaux par milieu, il faut gérer
    // pour chaque maille si le milieu évolue suite à l'ajout/suppression de matériau.
    // Les deux cas sont :
    // - en cas d'ajout, le milieu évolue pour une maille s'il n'y avait pas
    //   de matériau avant. Dans ce cas le milieu est ajouté à la maille.
    // - en cas de suppression, le milieu évolue dans la maille s'il y avait
    //   1 seul matériau avant. Dans ce cas le milieu est supprimé de la maille.

    Int32UniqueArray cells_unchanged_in_env;
    const Int32 ref_nb_mat = is_add ? 0 : 1;
    const Int16 env_id = true_env->componentId();
    info(4) << "Using optimisation updateMaterialDirect is_add?=" << is_add;

    for (Integer i = 0, n = ids.size(); i < n; ++i) {
      Int32 lid = ids[i];
      Int32 current_cell_nb_mat = connectivity->cellNbMaterial(CellLocalId(lid), env_id);
      if (current_cell_nb_mat != ref_nb_mat) {
        info(5) << "CELL i=" << i << " lid=" << lid << " unchanged in environment nb_mat=" << current_cell_nb_mat;
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
  _switchComponentItemsForMaterials(true_mat);
  info(4) << "Transform PartialPure for environment name=" << env->name();
  _switchComponentItemsForEnvironments(env);

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
_switchComponentItemsForMaterials(const MeshMaterial* modified_mat)
{
  const bool is_add = m_work_info.isAdd();

  for (MeshEnvironment* true_env : m_material_mng->trueEnvironments()) {
    for (MeshMaterial* mat : true_env->trueMaterials()) {
      // Ne traite pas le matériau en cours de modification.
      if (mat == modified_mat)
        continue;

      m_work_info.pure_local_ids.clear();
      m_work_info.partial_indexes.clear();

      const MeshEnvironment* env = mat->trueEnvironment();
      if (env != true_env)
        ARCANE_FATAL("BAD ENV");
      MeshMaterialVariableIndexer* indexer = mat->variableIndexer();

      info(4) << "TransformCells (V3) is_add?=" << is_add << " indexer=" << indexer->name();

      _computeCellsToTransform(mat);

      indexer->transformCellsV2(m_work_info);

      info(4) << "NB_MAT_TRANSFORM=" << m_work_info.pure_local_ids.size() << " name=" << mat->name();

      m_all_env_data->_copyBetweenPartialsAndGlobals(m_work_info.pure_local_ids,
                                                     m_work_info.partial_indexes,
                                                     indexer->index(), is_add);
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
_switchComponentItemsForEnvironments(const IMeshEnvironment* modified_env)
{
  const bool is_add = m_work_info.isAdd();

  // Ne copie pas les valeurs partielles des milieux vers les valeurs globales
  // en cas de suppression de mailles car cela sera fait avec la valeur matériau
  // correspondante. Cela permet d'avoir le même comportement que sans
  // optimisation. Ce n'est pas actif par défaut pour compatibilité avec l'existant.
  const bool is_copy = is_add || !(m_material_mng->isUseMaterialValueWhenRemovingPartialValue());

  for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    // Ne traite pas le milieu en cours de modification.
    if (env == modified_env)
      continue;

    m_work_info.pure_local_ids.clear();
    m_work_info.partial_indexes.clear();

    MeshMaterialVariableIndexer* indexer = env->variableIndexer();

    info(4) << "TransformCells (V2) is_add?=" << is_add << " indexer=" << indexer->name();

    _computeCellsToTransform();
    indexer->transformCellsV2(m_work_info);

    info(4) << "NB_ENV_TRANSFORM=" << m_work_info.pure_local_ids.size()
            << " name=" << env->name();

    if (is_copy)
      m_all_env_data->_copyBetweenPartialsAndGlobals(m_work_info.pure_local_ids,
                                                     m_work_info.partial_indexes,
                                                     indexer->index(), is_add);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule les mailles à transformer pour le matériau \at mat.
 */
void IncrementalComponentModifier::
_computeCellsToTransform(const MeshMaterial* mat)
{
  const MeshEnvironment* env = mat->trueEnvironment();
  const Int16 env_id = env->componentId();
  CellGroup all_cells = m_material_mng->mesh()->allCells();
  bool is_add = m_work_info.isAdd();

  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  ConstArrayView<Int16> cells_nb_env = connectivity->cellsNbEnvironment();

  ENUMERATE_ (Cell, icell, all_cells) {
    bool do_transform = false;
    // En cas d'ajout on passe de pure à partiel s'il y a plusieurs milieux ou
    // plusieurs matériaux dans le milieu.
    // En cas de supression, on passe de partiel à pure si on est le seul matériau
    // et le seul milieu.
    if (is_add) {
      do_transform = cells_nb_env[icell.itemLocalId()] > 1;
      if (!do_transform)
        do_transform = connectivity->cellNbMaterial(icell, env_id) > 1;
    }
    else {
      do_transform = cells_nb_env[icell.itemLocalId()] == 1;
      if (do_transform)
        do_transform = connectivity->cellNbMaterial(icell, env_id) == 1;
    }
    m_work_info.setTransformedCell(icell, do_transform);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule les mailles à transformer lorsqu'on modifie les mailles
 * d'un milieu.
 */
void IncrementalComponentModifier::
_computeCellsToTransform()
{
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  ConstArrayView<Int16> cells_nb_env = connectivity->cellsNbEnvironment();
  CellGroup all_cells = m_material_mng->mesh()->allCells();
  const bool is_add = m_work_info.isAdd();

  ENUMERATE_ (Cell, icell, all_cells) {
    bool do_transform = false;
    // En cas d'ajout on passe de pure à partiel s'il y a plusieurs milieux.
    // En cas de supression, on passe de partiel à pure si on est le seul milieu.
    if (is_add)
      do_transform = cells_nb_env[icell.itemLocalId()] > 1;
    else
      do_transform = cells_nb_env[icell.itemLocalId()] == 1;
    m_work_info.setTransformedCell(icell, do_transform);
  }
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
  m_work_info.setRemovedCells(local_ids, true);

  // TODO: à faire dans finialize()
  env->addToTotalNbCellMat(-nb_to_remove);

  mat->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove);

  if (update_env_indexer) {
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    env->variableIndexer()->endUpdateRemove(m_work_info, nb_to_remove);
  }

  // Remet \a removed_local_ids_filter à la valeur initiale pour
  // les prochaines opérations
  m_work_info.setRemovedCells(local_ids, false);
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
  info(4) << "MeshEnvironment::addItemsDirect"
          << " mat=" << mat->name();

  MeshMaterialVariableIndexer* var_indexer = mat->variableIndexer();
  Int32 nb_to_add = local_ids.size();

  // Met à jour le nombre de matériaux par maille et le nombre total de mailles matériaux.
  env->addToTotalNbCellMat(nb_to_add);

  _addItemsToIndexer(env, var_indexer, local_ids);

  if (update_env_indexer) {
    // Met aussi à jour les entités \a local_ids à l'indexeur du milieu.
    // Cela n'est possible que si le nombre de matériaux du milieu
    // est supérieur ou égal à 2 (car sinon le matériau et le milieu
    // ont le même indexeur)
    _addItemsToIndexer(env, env->variableIndexer(), local_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalComponentModifier::
_addItemsToIndexer(MeshEnvironment* env, MeshMaterialVariableIndexer* var_indexer,
                   Int32ConstArrayView local_ids)
{
  ComponentItemListBuilder list_builder(var_indexer, var_indexer->maxIndexInMultipleArray());
  ConstituentConnectivityList* connectivity = m_all_env_data->componentConnectivityList();
  ConstArrayView<Int16> nb_env_per_cell = connectivity->cellsNbEnvironment();
  Int16 env_id = env->componentId();

  for (Int32 lid : local_ids) {
    CellLocalId cell_id(lid);
    // On ne prend l'indice global que si on est le seul matériau et le seul
    // milieu de la maille. Sinon, on prend un indice multiple
    if (nb_env_per_cell[cell_id] > 1 || connectivity->cellNbMaterial(cell_id, env_id) > 1)
      list_builder.addPartialItem(lid);
    else
      list_builder.addPureItem(lid);
  }

  if (traceMng()->verbosityLevel() >= 5)
    info() << "ADD_MATITEM_TO_INDEXER component=" << var_indexer->name()
           << " nb_pure=" << list_builder.pureMatVarIndexes().size()
           << " nb_partial=" << list_builder.partialMatVarIndexes().size()
           << "\n pure=(" << list_builder.pureMatVarIndexes() << ")"
           << "\n partial=(" << list_builder.partialMatVarIndexes() << ")";

  var_indexer->endUpdateAdd(list_builder);

  // Maintenant que les nouveaux MatVar sont créés, il faut les
  // initialiser avec les bonnes valeurs.
  {
    IMeshMaterialMng* mm = m_material_mng;
    functor::apply(mm, &IMeshMaterialMng::visitVariables,
                   [&](IMeshMaterialVariable* mv) { mv->_internalApi()->initializeNewItems(list_builder); });
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
