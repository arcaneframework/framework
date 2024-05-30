// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllEnvData.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Informations sur les valeurs des milieux.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ArraySimdPadder.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/internal/ItemGroupImplInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"

#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"

#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/AllEnvData.h"
#include "arcane/materials/internal/MaterialModifierOperation.h"
#include "arcane/materials/internal/ConstituentConnectivityList.h"
#include "arcane/materials/internal/ComponentItemListBuilder.h"

#include "arcane/accelerator/Scan.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/RunCommandEnumerate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvData::
AllEnvData(MeshMaterialMng* mmg)
: TraceAccessor(mmg->traceMng())
, m_material_mng(mmg)
, m_item_internal_data(mmg)
{
  // \a m_component_connectivity_list utilse un compteur de référence
  // et ne doit pas être détruit explicitement
  m_component_connectivity_list = new ConstituentConnectivityList(m_material_mng);
  m_component_connectivity_list_ref = m_component_connectivity_list->toSourceReference();

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ALLENVDATA_DEBUG_LEVEL", true))
    m_verbose_debug_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
endCreate(bool is_continue)
{
  m_item_internal_data.endCreate();
  m_component_connectivity_list->endCreate(is_continue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AllEnvData::
_isFullVerbose() const
{
  return (m_verbose_debug_level > 1 || traceMng()->verbosityLevel() >= 5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeNbEnvAndNbMatPerCell()
{
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  // Calcule le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  for (MeshEnvironment* env : true_environments) {
    env->computeNbMatPerCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeAndResizeEnvItemsInternal()
{
  // Calcule le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  IMesh* mesh = m_material_mng->mesh();
  const IItemFamily* cell_family = mesh->cellFamily();
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  Integer nb_env = true_environments.size();
  Integer total_env_cell = 0;
  Integer total_mat_cell = 0;
  info(4) << "NB_ENV = " << nb_env;
  for (const MeshEnvironment* env : true_environments) {
    CellGroup cells = env->cells();
    Integer env_nb_cell = cells.size();
    info(4) << "EnvName=" << cells.name() << " nb_env_cell=" << env_nb_cell << " nb_mat_cell=" << env->totalNbCellMat();
    total_env_cell += env_nb_cell;
    total_mat_cell += env->totalNbCellMat();
  }

  // Il faut ajouter les infos pour les mailles de type AllEnvCell
  Int32 max_local_id = cell_family->maxLocalId();
  info(4) << "RESIZE TotalEnvCell=" << total_env_cell
          << " TotalMatCell=" << total_mat_cell
          << " MaxLocalId=" << max_local_id;

  // Redimensionne les tableaux des infos
  // ATTENTION : ils ne doivent plus être redimensionnés par la suite sous peine
  // de tout invalider.
  m_item_internal_data.resizeComponentItemInternals(max_local_id, total_env_cell);

  if (arcaneIsCheck()) {
    Int32 computed_nb_mat = 0;
    Int32 computed_nb_env = 0;
    ConstArrayView<Int16> cells_nb_env = m_component_connectivity_list->cellsNbEnvironment();
    ConstArrayView<Int16> cells_nb_mat = m_component_connectivity_list->cellsNbMaterial();
    ENUMERATE_ (Cell, icell, cell_family->allItems()) {
      Int32 lid = icell.itemLocalId();
      computed_nb_env += cells_nb_env[lid];
      computed_nb_mat += cells_nb_mat[lid];
    }
    Int32 computed_size = computed_nb_mat + computed_nb_env;
    Int32 storage_size = total_mat_cell + total_env_cell;
    info(4) << "storage_size=" << storage_size << " computed=" << computed_size
            << " max_local_id=" << max_local_id << " internal_nb_mat=" << total_mat_cell << " internal_nb_env=" << total_env_cell
            << " computed_nb_mat=" << computed_nb_mat << " computed_nb_env=" << computed_nb_env;
    if (storage_size != computed_size)
      ARCANE_FATAL("BAD STORAGE SIZE internal={0} connectivity={1}", storage_size, computed_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reconstruit les connectivités incrémentales à parties des groupes.
 */
void AllEnvData::
_rebuildIncrementalConnectivitiesFromGroups()
{
  RunQueue queue(makeQueue(m_material_mng->runner()));
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());
  auto clist = m_component_connectivity_list;
  clist->removeAllConnectivities();
  for (MeshEnvironment* env : true_environments) {
    clist->addCellsToEnvironment(env->componentId(), env->cells().view().localIds(), queue);
    for (MeshMaterial* mat : env->trueMaterials())
      clist->addCellsToMaterial(mat->componentId(), mat->cells().view().localIds(), queue);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_rebuildMaterialsAndEnvironmentsFromGroups()
{
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());
  const bool is_full_verbose = _isFullVerbose();
  ConstArrayView<Int16> cells_nb_env = m_component_connectivity_list->cellsNbEnvironment();
  for (const MeshEnvironment* env : true_environments) {
    MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
    ComponentItemListBuilderOld list_builder(var_indexer, 0);
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();
    info(4) << "ENV_INDEXER (V2) i=" << var_indexer->index() << " NB_CELL=" << var_nb_cell << " name=" << cells.name()
            << " index=" << var_indexer->index();
    if (is_full_verbose)
      info(5) << "ENV_INDEXER (V2) name=" << cells.name() << " cells=" << cells.view().localIds();

    ENUMERATE_CELL (icell, cells) {
      if (cells_nb_env[icell.itemLocalId()] > 1)
        list_builder.addPartialItem(icell.itemLocalId());
      else
        // Je suis le seul milieu de la maille donc je prends l'indice global
        list_builder.addPureItem(icell.itemLocalId());
    }
    if (is_full_verbose)
      info() << "MAT_NB_MULTIPLE_CELL (V2) mat=" << var_indexer->name()
             << " nb_in_global=" << list_builder.pureMatVarIndexes().size()
             << " (ids=" << list_builder.pureMatVarIndexes() << ")"
             << " nb_in_multiple=" << list_builder.partialMatVarIndexes().size()
             << " (ids=" << list_builder.partialLocalIds() << ")";
    var_indexer->endUpdate(list_builder);
  }

  for (MeshEnvironment* env : true_environments)
    env->computeItemListForMaterials(*m_component_connectivity_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_computeInfosForEnvCells()
{
  IMesh* mesh = m_material_mng->mesh();
  IItemFamily* cell_family = mesh->cellFamily();
  CellGroup all_cells = cell_family->allItems();
  const Int32 nb_cell = all_cells.size();
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  SmallSpan<const Int16> cells_nb_env = m_component_connectivity_list->cellsNbEnvironment();

  // Calcule pour chaque maille sa position dans le tableau des milieux
  // en considérant que les milieux de chaque maille sont rangés consécutivement
  // dans m_env_items_internal.
  const Int32 max_local_id = cell_family->maxLocalId();
  UniqueArray<Int32> env_cell_indexes(platform::getDefaultDataAllocator());
  env_cell_indexes.resize(cells_nb_env.size());

  //! Tableau de travail pour le nombre de matériaux par milieu
  UniqueArray<Int16> cells_nb_material(platform::getDefaultDataAllocator());
  cells_nb_material.resize(max_local_id);

  RunQueue& queue(m_material_mng->runQueue());

  bool do_old = (max_local_id != nb_cell);
  if (do_old) {
    Integer env_cell_index = 0;
    ENUMERATE_CELL (icell, all_cells) {
      Int32 lid = icell.itemLocalId();
      Int32 nb_env = cells_nb_env[lid];
      env_cell_indexes[lid] = env_cell_index;
      env_cell_index += nb_env;
    }
  }
  else {
    // TODO: Cela ne fonctionne que si all_cells est compacté et
    // local_id[i] <=> i.
    Accelerator::GenericScanner scanner(queue);
    SmallSpan<Int32> env_cell_indexes_view(env_cell_indexes);
    Accelerator::ScannerSumOperator<Int32> op;
    scanner.applyExclusive(0, cells_nb_env, env_cell_indexes_view, op, A_FUNCINFO);
  }

  // Positionne les infos pour les EnvCell
  {
    NumArray<Int32, MDDim1> current_pos;
    {
      MDSpan<Int32, MDDim1> s(env_cell_indexes.data(), ArrayIndex<1>{ env_cell_indexes.size() });
      current_pos.copy(s);
    }
    for (MeshEnvironment* env : true_environments) {
      const Int16 env_id = env->componentId();
      const MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
      CellGroup cells = env->cells();

      env->resizeItemsInternal(var_indexer->nbItem());

      info(4) << "COMPUTE (V2) env_cells env=" << env->name() << " nb_cell=" << cells.size()
              << " index=" << var_indexer->index()
              << " max_multiple_index=" << var_indexer->maxIndexInMultipleArray();

      SmallSpan<const MatVarIndex> matvar_indexes(var_indexer->matvarIndexes());

      Int32ConstArrayView local_ids = var_indexer->localIds();

      SmallSpan<Int16> cells_nb_mat_view = cells_nb_material.view();
      m_component_connectivity_list->fillCellsNbMaterial(local_ids, env_id, cells_nb_mat_view, queue);

      auto command = makeCommand(queue);
      SmallSpan<Int32> current_pos_view(current_pos);
      const Int32 nb_id = matvar_indexes.size();
      ComponentItemSharedInfo* env_shared_info = m_item_internal_data.envSharedInfo();

      Span<Int32> env_cells_local_id = cells._internalApi()->itemsLocalId();
      SmallSpan<ConstituentItemIndex> env_id_list = env->componentData()->m_constituent_local_id_list.mutableLocalIds();
      command << RUNCOMMAND_LOOP1(iter, nb_id)
      {
        auto [z] = iter();
        MatVarIndex mvi = matvar_indexes[z];

        Int32 lid = local_ids[z];
        Int32 pos = current_pos_view[lid];
        ++current_pos_view[lid];
        Int16 nb_mat = cells_nb_mat_view[z];

        ConstituentItemIndex cii_pos(pos);
        matimpl::ConstituentItemBase ref_ii(env_shared_info, cii_pos);
        ConstituentItemIndex cii_lid(lid);
        env_id_list[z] = cii_pos;

        ref_ii._setSuperAndGlobalItem(cii_lid, ItemLocalId(lid));
        ref_ii._setNbSubItem(nb_mat);
        ref_ii._setVariableIndex(mvi);
        ref_ii._setComponent(env_id);
        // Le rang 0 met à jour le padding SIMD du groupe associé au matériau
        if (z==0)
          ArraySimdPadder::applySimdPaddingView(env_cells_local_id);
      };
      cells._internalApi()->notifySimdPaddingDone();
    }
    Accelerator::RunQueuePool& queue_pool = m_material_mng->_internalApi()->asyncRunQueuePool();
    for (MeshEnvironment* env : true_environments) {
      env->computeMaterialIndexes(&m_item_internal_data, queue_pool[env->id()]);
    }
    queue_pool.barrier();
  }

  // Positionne les infos pour les AllEnvCell.
  {
    ComponentItemSharedInfo* all_env_shared_info = m_item_internal_data.allEnvSharedInfo();
    auto command = makeCommand(queue);
    SmallSpan<Int32> env_cell_indexes_view(env_cell_indexes);
    command << RUNCOMMAND_ENUMERATE (Cell, cell_id, all_cells)
    {
      Int32 lid = cell_id;
      Int16 n = cells_nb_env[lid];
      matimpl::ConstituentItemBase ref_ii(all_env_shared_info, ConstituentItemIndex(lid));
      ref_ii._setSuperAndGlobalItem({}, cell_id);
      ref_ii._setVariableIndex(MatVarIndex(0, lid));
      ref_ii._setNbSubItem(n);
      if (n != 0)
        ref_ii._setFirstSubItem(ConstituentItemIndex(env_cell_indexes_view[lid]));
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie la cohérence des localIds() du variableIndexer().
 * avec la maille globale associée au milieu
 */
void AllEnvData::
_checkLocalIdsCoherency() const
{
  for (MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    Int32 index = 0;
    Int32ConstArrayView indexer_local_ids = env->variableIndexer()->localIds();
    ENUMERATE_COMPONENTCELL (icitem, env) {
      ComponentCell cc = *icitem;
      Int32 matvar_lid = cc.globalCell().localId();
      Int32 direct_lid = indexer_local_ids[index];
      if (matvar_lid != direct_lid)
        ARCANE_FATAL("Incoherent localId() matvar_lid={0} direct_lid={1} index={2}",
                     matvar_lid, direct_lid, index);
      ++index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remise à jour des structures suite à une modification des mailles
 * de matériaux ou de milieux.
 *
 * Cette méthode reconstruit les informations uniquement à partir des
 * groupes d'entités associés aux matériaux et milieux. Les variables
 * matériaux ne sont pas prise en compte par cette méthode et il est donc
 * possible qu'elles soient invalidées suite à cet appel. Si on souhaite la conservation
 * des valeurs, il faut d'abord sauvegarder les valeurs partielles,
 * appliquer cette méthode puis restaurer les valeurs partielles. La classe
 * MeshMaterialBackup permet cette sauvegarde/restauration.
 *
 * A noter que cette méthode peut être utilisée en reprise en conservant
 * les valeurs des variables matériaux car la structure des groupes
 * est la même après une reprise ce qui n'invalide pas les valeurs partielles.
 */
void AllEnvData::
forceRecompute(bool compute_all)
{
  m_material_mng->incrementTimestamp();

  ConstArrayView<MeshMaterialVariableIndexer*> vars_idx = m_material_mng->_internalApi()->variablesIndexer();
  Integer nb_var = vars_idx.size();
  Int64 mesh_timestamp = m_material_mng->mesh()->timestamp();
  info(4) << "ForceRecompute NB_VAR_IDX=" << nb_var << " compute_all?=" << compute_all
          << " mesh_timestamp=" << mesh_timestamp << " current_mesh_timestamp=" << m_current_mesh_timestamp;

  // Si le maillage a changé, il y a certaines choses qu'il faut toujours recalculer
  bool has_mesh_changed = m_current_mesh_timestamp != mesh_timestamp;
  m_current_mesh_timestamp = mesh_timestamp;

  const bool is_verbose_debug = m_verbose_debug_level > 0;

  // Il faut compter le nombre total de mailles par milieu et par matériau

  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  if (compute_all)
    _rebuildIncrementalConnectivitiesFromGroups();

  // Calcule le nombre de milieux par maille, et pour chaque
  // milieu le nombre de matériaux par maille
  if (compute_all || has_mesh_changed)
    _computeNbEnvAndNbMatPerCell();

  _computeAndResizeEnvItemsInternal();

  bool is_full_verbose = _isFullVerbose();

  if (compute_all)
    _rebuildMaterialsAndEnvironmentsFromGroups();

  for (const MeshEnvironment* env : true_environments) {
    const MeshMaterialVariableIndexer* var_indexer = env->variableIndexer();
    CellGroup cells = var_indexer->cells();
    Integer var_nb_cell = cells.size();
    info(4) << "FINAL_INDEXER i=" << var_indexer->index() << " NB_CELL=" << var_nb_cell << " name=" << cells.name()
            << " index=" << var_indexer->index();
    if (is_full_verbose) {
      Int32UniqueArray my_array(cells.view().localIds());
      info(5) << "FINAL_INDEXER (V2) name=" << cells.name() << " cells=" << my_array;
      info(4) << "FINAL_MAT_NB_MULTIPLE_CELL (V2) mat=" << var_indexer->name()
              << " ids=" << var_indexer->matvarIndexes();
    }
  }

  _computeInfosForEnvCells();

  if (is_verbose_debug) {
    _printAllEnvCells(m_material_mng->mesh()->allCells().view());
    for (IMeshMaterial* material : m_material_mng->materials()) {
      ENUMERATE_COMPONENTITEM (MatCell, imatcell, material) {
        MatCell pmc = *imatcell;
        info() << "CELL IN MAT vindex=" << pmc._varIndex();
      }
    }
  }

  {
    RunQueue& queue(m_material_mng->runQueue());
    for (MeshEnvironment* env : true_environments) {
      env->componentData()->_rebuildPartData(queue);
      for (MeshMaterial* mat : env->trueMaterials())
        mat->componentData()->_rebuildPartData(queue);
    }
  }

  if (arcaneIsCheck())
    m_material_mng->checkValid();

  m_material_mng->syncVariablesReferences(compute_all);

  if (is_verbose_debug) {
    OStringStream ostr;
    m_material_mng->dumpInfos2(ostr());
    info() << ostr.str();
  }

  // Vérifie la cohérence des localIds() du variableIndexer()
  // avec la maille globale associée au milieu
  if (arcaneIsCheck())
    _checkLocalIdsCoherency();

  // Met à jour le AllCellToAllEnvCell s'il a été initialisé si la fonctionnalité est activé
  if (m_material_mng->isCellToAllEnvCellForRunCommand()) {
    auto* all_cell_to_all_env_cell(m_material_mng->_internalApi()->getAllCellToAllEnvCell());
    if (all_cell_to_all_env_cell)
      all_cell_to_all_env_cell->bruteForceUpdate();
    else
      m_material_mng->_internalApi()->createAllCellToAllEnvCell(platform::getDefaultDataAllocator());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
recomputeIncremental()
{
  forceRecompute(false);
  if (arcaneIsCheck())
    _checkConnectivityCoherency();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_printAllEnvCells(CellVectorView ids)
{
  ConstArrayView<Int16> cells_nb_env = m_component_connectivity_list->cellsNbEnvironment();
  ENUMERATE_ALLENVCELL (iallenvcell, m_material_mng->view(ids)) {
    AllEnvCell all_env_cell = *iallenvcell;
    Integer cell_nb_env = all_env_cell.nbEnvironment();
    Cell cell = all_env_cell.globalCell();
    info() << "CELL2 uid=" << ItemPrinter(cell)
           << " nb_env=" << cells_nb_env[cell.localId()]
           << " direct_nb_env=" << cell_nb_env;
    for (Integer z = 0; z < cell_nb_env; ++z) {
      EnvCell ec = all_env_cell.cell(z);
      Integer cell_nb_mat = ec.nbMaterial();
      info() << "CELL3 nb_mat=" << cell_nb_mat << " env_id=" << ec.environmentId();
      for (Integer k = 0; k < cell_nb_mat; ++k) {
        MatCell mc = ec.cell(k);
        info() << "CELL4 mat_item=" << mc._varIndex() << " mat_id=" << mc.materialId();
      }
    }
  }
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
void AllEnvData::
_copyBetweenPartialsAndGlobals(const CopyBetweenPartialAndGlobalArgs& args,
                               bool is_add_operation)
{
  if (args.m_local_ids.empty())
    return;
  bool do_copy = args.m_do_copy_between_partial_and_pure;
  RunQueue queue(args.m_queue);
  RunQueue::ScopedAsync sc(&queue);
  // Comme on a modifié des mailles, il faut mettre à jour les valeurs
  // correspondantes pour chaque variable.
  //info(4) << "NB_TRANSFORM=" << nb_transform << " name=" << e->name();
  //Integer indexer_index = indexer->index();

  Accelerator::RunQueuePool& queue_pool = m_material_mng->_internalApi()->asyncRunQueuePool();

  // Redimensionne les variables si nécessaire
  if (is_add_operation) {
    Int32 index = 0;
    auto func1 = [&](IMeshMaterialVariable* mv) {
      auto* mvi = mv->_internalApi();
      mvi->resizeForIndexer(args.m_var_index, queue_pool[index]);
      ++index;
    };
    functor::apply(m_material_mng, &MeshMaterialMng::visitVariables, func1);
    queue_pool.barrier();
  }

  if (do_copy) {
    Int32 index = 0;
    CopyBetweenPartialAndGlobalArgs args2(args);
    auto func2 = [&](IMeshMaterialVariable* mv) {
      auto* mvi = mv->_internalApi();
      args2.m_queue = queue_pool[index];
      if (is_add_operation)
        mvi->copyGlobalToPartial(args);
      else
        mvi->copyPartialToGlobal(args);
      ++index;
    };
    functor::apply(m_material_mng, &MeshMaterialMng::visitVariables, func2);
    queue_pool.barrier();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AllEnvData::
_checkConnectivityCoherency()
{
  info() << "AllEnvData: checkCoherency()";
  ConstArrayView<Int16> nb_mat_v2 = m_component_connectivity_list->cellsNbMaterial();
  ConstArrayView<MeshEnvironment*> true_environments(m_material_mng->trueEnvironments());

  ItemGroup all_cells = m_material_mng->mesh()->allCells();

  Int32 nb_error = 0;

  // Vérifie le nombre de matériaux par maille
  ENUMERATE_CELL (icell, all_cells) {
    Int32 ref_nb_mat = 0;
    for (MeshEnvironment* env : true_environments) {
      Int16 env_id = env->componentId();
      ref_nb_mat += m_component_connectivity_list->cellNbMaterial(icell, env_id);
    }
    Int32 current_nb_mat = nb_mat_v2[icell.itemLocalId()];
    if (ref_nb_mat != current_nb_mat) {
      ++nb_error;
      if (nb_error < 10)
        error() << "Invalid values for nb_material cell=" << icell->uniqueId()
                << " ref=" << ref_nb_mat << " current=" << current_nb_mat;
    }
  }

  if (nb_error != 0)
    ARCANE_FATAL("Invalid values for component connectivity nb_error={0}", nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
