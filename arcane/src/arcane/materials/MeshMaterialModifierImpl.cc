// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifierImpl.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IData.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

#include "arcane/materials/MeshMaterialModifierImpl.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/MeshMaterialBackup.h"
#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/AllEnvData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialModifierImpl::Operation
{
 public:
  Operation() = default;
  Operation(IMeshMaterial* mat,Int32ConstArrayView ids,bool is_add)
  : m_mat(mat), m_is_add(is_add), m_ids(ids)
  {
  }
 public:
  IMeshMaterial* m_mat = nullptr;
  bool m_is_add = false;
  UniqueArray<Int32> m_ids;
};

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
, nb_update(0)
, nb_save_restore(0)
, nb_optimize_add(0)
, nb_optimize_remove(0)
, m_allow_optimization(false)
, m_allow_optimize_multiple_operation(false)
, m_allow_optimize_multiple_material(false)
{
  _setLocalVerboseLevel(4);
  if (!platform::getEnvironmentVariable("ARCANE_DEBUG_MATERIAL_MODIFIER").null())
    _setLocalVerboseLevel(3);
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

  info() << "MeshMaterialModifier::optimization: "
         << " allow?=" << m_allow_optimization
         << " allow_multiple?=" << m_allow_optimize_multiple_operation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
addCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  if (ids.empty())
    return;
  m_operations.add(new Operation(mat,ids,true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
removeCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  if (ids.empty())
    return;
  m_operations.add(new Operation(mat,ids,false));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_setCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  CellGroup cells = mat->cells();
  info(4) << "SET_CELLS_TO_MATERIAL: mat=" << mat->name()
         << " nb_item=" << ids.size();
  cells.setItems(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_addCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  CellGroup cells = mat->cells();
  info(4) << "ADD_CELLS_TO_MATERIAL: mat=" << mat->name()
         << " nb_item=" << ids.size();
  cells.addItems(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_removeCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  CellGroup cells = mat->cells();
  info(4) << "REMOVE_CELLS_TO_MATERIAL: mat=" << mat->name()
         << " nb_item=" << ids.size();
  cells.removeItems(ids);
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
    IMeshMaterial* mat = op->m_mat;
    if (mat->environment()->nbMaterial()!=1 && !m_allow_optimize_multiple_material){
      linfo() << "_checkMayOptimize(): not allowing optimization because environment has several material";
      return false;
    }
#if 0
    // Pour l'instant n'optimise pas la suppression en multi-mat
    // car cela n'est pas implémenté.
    if (mat->environment()->nbMaterial()!=1 && !op->m_is_add){
      linfo() << "_checkMayOptimize(): not allowing optimization because environment has several material and action is not add";
      return false;
    }
#endif
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
endUpdate()
{
  bool is_keep_value = m_material_mng->isKeepValuesAfterChange();
  Integer nb_operation = m_operations.values().size();

  linfo() << "END_UPDATE_MAT_Begin keep=" << is_keep_value
          << " nb_operation=" << nb_operation;

  m_material_mng->incrementTimestamp();

  MeshMaterialBackup backup(m_material_mng,false);

  bool need_restore = true;
  UniqueArray<Int32> keeped_lids;
  bool no_optimization_done = true;

  ++nb_update;

  for( Integer i=0; i<nb_operation; ++i ){
    Operation* op = m_operations.values()[i];
    IMeshMaterial* mat = op->m_mat;
    IMeshComponentInternal* mci = mat->_internalApi();
    if (op->m_is_add){
      linfo() << "MODIFIER_ADD_CELLS_TO_MATERIAL: mat=" << mat->name()
              << " mat_index=" << mci->variableIndexer()->index()
              << " op_index=" << i
              << " nb_item=" << op->m_ids.size()
              << " ids=" << op->m_ids;
    }
    if (!op->m_is_add){
      linfo() << "MODIFIER_REMOVE_CELLS_TO_MATERIAL: mat=" << mat->name()
              << " mat_index=" << mci->variableIndexer()->index()
              << " op_index=" << i
              << " nb_item=" << op->m_ids.size()
              << " ids=" << op->m_ids;
    }
  }

  bool is_optimization_active = m_allow_optimization;
  if (is_optimization_active)
    is_optimization_active = _checkMayOptimize();
  linfo() << "Check optimize ? = " << is_optimization_active;

  if (is_optimization_active){
    for( Integer i=0; i<nb_operation; ++i ){
      Operation* op = m_operations.values()[i];
      IMeshMaterial* mat = op->m_mat;

      if (op->m_is_add){
        linfo() << "ONLY_ONE_ADD: using optimization mat=" << mat->name();
        keeped_lids = op->m_ids;
        ++nb_optimize_add;
        m_material_mng->allEnvData()->updateMaterialDirect(mat,op->m_ids,eOperation::Add);
      }

      if (!op->m_is_add){
        linfo() << "ONLY_ONE_REMOVE: using optimization mat=" << mat->name();
        keeped_lids = op->m_ids;
        ++nb_optimize_remove;
        m_material_mng->allEnvData()->updateMaterialDirect(mat,op->m_ids,eOperation::Remove);
      }
    }
    no_optimization_done = false;
    need_restore = false;
  }

  if (no_optimization_done){
    if (is_keep_value){
      ++nb_save_restore;
      backup.saveValues();
    }

    _applyOperations();
    _updateEnvironments();

    m_material_mng->allEnvData()->forceRecompute(true);

    if (is_keep_value){
      backup.restoreValues();
    }
  }
  else{
    if (is_keep_value && need_restore){
      ++nb_save_restore;
      backup.saveValues();
    }

    m_material_mng->allEnvData()->forceRecompute(false);

    if (is_keep_value && need_restore){
      backup.restoreValues();
    }
  }

  linfo() << "END_UPDATE_MAT End";
  if (keeped_lids.size()!=0){
    info(4) << "PRINT KEEPED_IDS size=" << keeped_lids.size();
    //m_material_mng->allEnvData()->printAllEnvCells(keeped_lids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_applyOperations()
{
  for( Operation* o : m_operations.values() ){
    IMeshMaterial* mat = o->m_mat;
    if (o->m_is_add)
      _addCells(mat,o->m_ids);
    else
      _removeCells(mat,o->m_ids);
  }
  m_operations.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifierImpl::
_updateEnvironments()
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
