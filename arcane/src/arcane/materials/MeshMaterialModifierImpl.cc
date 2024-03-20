// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifierImpl.cc                                 (C) 2000-2024 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IData.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/MeshMaterialBackup.h"
#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/AllEnvData.h"
#include "arcane/materials/internal/MeshMaterialModifierImpl.h"
#include "arcane/materials/internal/MaterialModifierOperation.h"
#include "arcane/materials/internal/IncrementalComponentModifier.h"
#include "arcane/materials/internal/ConstituentListPrinter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialModifierImpl::OperationList::
~OperationList()
{
  clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::OperationList::
add(Operation* o)
{
 m_operations.add(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::OperationList::
clear()
{
  for( Operation* o : m_operations )
    delete o;
  m_operations.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialModifierImpl::
MeshMaterialModifierImpl(MeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_queue(makeQueue(m_material_mng->runner()))
{
  _setLocalVerboseLevel(4);
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DEBUG_MATERIAL_MODIFIER", true)) {
    Int32 value = v.value();
    if (value > 0)
      _setLocalVerboseLevel(3);
    if (value > 1)
      m_print_component_list = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
initOptimizationFlags()
{
  int opt_flag_value = m_material_mng->modificationFlags();

  m_allow_optimization = false;

  info() << "OptimizationFlag = " << opt_flag_value;

  if (opt_flag_value!=0){
    info() << "Using optimization !";
    m_allow_optimization = true;
  }

  m_allow_optimize_multiple_operation = (opt_flag_value & (int)eModificationFlags::OptimizeMultiAddRemove)!=0;
  m_allow_optimize_multiple_material = (opt_flag_value & (int)eModificationFlags::OptimizeMultiMaterialPerEnvironment)!=0;
  m_use_incremental_recompute = true;
  if (m_use_incremental_recompute){
    m_allow_optimize_multiple_operation = true;
  }

  info() << "MeshMaterialModifier::optimization: "
         << " allow?=" << m_allow_optimization
         << " allow_multiple?=" << m_allow_optimize_multiple_operation
         << " allow_multiple_mat?=" << m_allow_optimize_multiple_material
         << " use_incremental_recompute?=" << m_use_incremental_recompute;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
addCells(IMeshMaterial* mat, SmallSpan<const Int32> ids)
{
  if (ids.empty())
    return;
  m_operations.add(Operation::createAdd(mat, ids));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
removeCells(IMeshMaterial* mat, SmallSpan<const Int32> ids)
{
  if (ids.empty())
    return;
  m_operations.add(Operation::createRemove(mat, ids));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_addCellsToGroupDirect(IMeshMaterial* mat,SmallSpan<const Int32> ids)
{
  CellGroup cells = mat->cells();
  info(4) << "ADD_CELLS_TO_MATERIAL: mat=" << mat->name()
         << " nb_item=" << ids.size();
  cells.addItems(ids.smallView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_removeCellsToGroupDirect(IMeshMaterial* mat,SmallSpan<const Int32> ids)
{
  CellGroup cells = mat->cells();
  info(4) << "REMOVE_CELLS_TO_MATERIAL: mat=" << mat->name()
         << " nb_item=" << ids.size();
  cells.removeItems(ids.smallView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshMaterialModifierImpl::
_checkMayOptimize()
{
  Integer nb_operation = m_operations.values().size();
  if (nb_operation>1 && !m_allow_optimize_multiple_operation)
    return false;
  for( Integer i=0; i<nb_operation; ++i ){
    Operation* op = m_operations.values()[i];
    IMeshMaterial* mat = op->material();
    if (mat->environment()->nbMaterial()!=1 && !m_allow_optimize_multiple_material){
      linfo() << "_checkMayOptimize(): not allowing optimization because environment has several material";
      return false;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique les opérations enregistrées.
 */
void MeshMaterialModifierImpl::
endUpdate()
{
  ConstituentListPrinter list_printer(m_material_mng);
  if (m_print_component_list){
    info() << "MeshMaterialModifierImpl::endUpdate(): BEGIN"
           << " modification_id=" << m_modification_id;
    list_printer.print();
  }
  _endUpdate();
  if (m_print_component_list){
    info() << "MeshMaterialModifierImpl::endUpdate(): END"
           << " modification_id=" << m_modification_id;
    list_printer.print();
  }
  ++m_modification_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_endUpdate()
{
  bool is_keep_value = m_material_mng->isKeepValuesAfterChange();
  Integer nb_operation = m_operations.values().size();

  linfo() << "END_UPDATE_MAT_Begin keep=" << is_keep_value
          << " nb_operation=" << nb_operation;

  m_material_mng->incrementTimestamp();

  MeshMaterialBackup backup(m_material_mng, false);

  bool no_optimization_done = true;

  ++nb_update;
  AllEnvData* all_env_data = m_material_mng->allEnvData();

  bool is_display = traceMng()->verbosityLevel() >= _localVerboseLevel();

  for (Integer i = 0; i < nb_operation; ++i) {
    const Operation* op = m_operations.values()[i];
    IMeshMaterial* mat = op->material();
    const IMeshComponentInternal* mci = mat->_internalApi();

    // N'appelle la méthode que si l'affichage sera réalisé pour éviter
    // de recopier 'op->ids()' sur l'hôte.
    if (is_display)
      linfo() << "MODIFIER_CELLS_TO_MATERIAL: mat=" << mat->name()
              << " is_add=" << op->isAdd()
              << " mat_index=" << mci->variableIndexer()->index()
              << " op_index=" << i
              << " ids=" << op->ids();
  }

  bool is_optimization_active = m_allow_optimization;
  if (is_optimization_active)
    is_optimization_active = _checkMayOptimize();
  linfo() << "Check optimize ? = " << is_optimization_active;

  // Tableau de travail utilisé lors des modifications incrémentales
  IncrementalComponentModifier incremental_modifier(all_env_data, m_queue);
  if (is_optimization_active && m_use_incremental_recompute) {
    incremental_modifier.initialize();
  }

  if (is_optimization_active) {
    for (Operation* op : m_operations.values()) {
      const IMeshMaterial* mat = op->material();

      if (op->isAdd()) {
        linfo() << "ONLY_ONE_ADD: using optimization mat=" << mat->name();
        ++nb_optimize_add;
      }
      else {
        linfo() << "ONLY_ONE_REMOVE: using optimization mat=" << mat->name();
        ++nb_optimize_remove;
      }

      incremental_modifier.m_work_info.setCurrentOperation(op);

      // Vérifie dans le cas des mailles à ajouter si elles ne sont pas déjà
      // dans le matériau et dans le cas des mailles à supprimer si elles y sont.
      if (arcaneIsCheck())
        op->filterIds();

      incremental_modifier.apply(op);
    }
    no_optimization_done = false;
  }

  if (no_optimization_done) {
    if (is_keep_value) {
      ++nb_save_restore;
      backup.saveValues();
    }

    _applyOperationsNoOptimize();
    _updateEnvironmentsNoOptimize();

    all_env_data->forceRecompute(true);

    if (is_keep_value) {
      backup.restoreValues();
    }
  }
  else {
    incremental_modifier.finalize();
    all_env_data->recomputeIncremental();
  }

  linfo() << "END_UPDATE_MAT End";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_applyOperationsNoOptimize()
{
  for( Operation* o : m_operations.values() ){
    IMeshMaterial* mat = o->material();
    if (o->isAdd())
      _addCellsToGroupDirect(mat,o->ids());
    else
      _removeCellsToGroupDirect(mat,o->ids());
  }
  m_operations.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_updateEnvironmentsNoOptimize()
{
  ConstArrayView<IMeshEnvironment*> envs = m_material_mng->environments();
  Int32UniqueArray cells_to_add;
  Int32UniqueArray cells_to_remove;
  info(4) << "CHECK ENVIRONMENTS";

  // Regarde s'il faut ajouter ou supprimer des mailles d'un milieu
  // en fonction de celles qui ont été ajoutées ou supprimées dans les
  // matériaux.
  for( IMeshEnvironment* env : envs ){
    
    // Pour les milieux ne contenant qu'un seul materiau, il n'y a rien
    // à faire car le groupe materiau est le meme que le groupe milieu
    if (env->nbMaterial()==1)
      continue;

    CellGroup env_cells = env->cells();
    info(4) << "CHECK ENV name=" << env->name() << " nb_cell=" << env_cells.size();
    Integer max_id = env_cells.itemFamily()->maxLocalId();
    Int32UniqueArray cells_marker(max_id,-1);
    ENUMERATE_CELL(icell,env_cells){
      cells_marker[icell.itemLocalId()] = 0;
    }
    cells_to_add.clear();
    cells_to_remove.clear();
    ConstArrayView<IMeshMaterial*> env_materials = env->materials();
    for( IMeshMaterial* mat : env_materials ){
      ENUMERATE_CELL(icell,mat->cells()){
        Int32 mark = cells_marker[icell.itemLocalId()];
        if (mark==(-1)){
          //mark = 0;
          // Maille a ajouter.
          cells_to_add.add(icell.itemLocalId());
        }
        //++mark;
        cells_marker[icell.itemLocalId()] = 1;
      }
    }

    ENUMERATE_CELL(icell,env_cells){
      Int32 mark = cells_marker[icell.itemLocalId()];
      if (mark==0)
        cells_to_remove.add(icell.itemLocalId());
    }
    if (!cells_to_add.empty()){
      info(4) << "ADD_CELLS to env " << env->name() << " n=" << cells_to_add.size();
      env_cells.addItems(cells_to_add);
    }
    if (!cells_to_remove.empty()){
      info(4) << "REMOVE_CELLS to env " << env->name() << " n=" << cells_to_remove.size();
      env_cells.removeItems(cells_to_remove);
    }
      
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
beginUpdate()
{
  m_material_mng->info(4) << "BEGIN_UPDATE_MAT";
  m_operations.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
dumpStats()
{
  info() << "MeshMaterialModifierImpl statistics:";
  info() << " Nb update : " << nb_update;
  info() << " Nb save/restore : " << nb_save_restore;
  info() << " Nb optimized add : " << nb_optimize_add;
  info() << " Nb optimized remove : " << nb_optimize_remove;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
