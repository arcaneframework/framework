// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialTesterModule.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Module de test du gestionnaire des matériaux.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/MeshMaterialTesterModule.h"

#include "arcane/utils/List.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/SimdOperation.h"

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/VariableDependInfo.h"
#include "arcane/core/Concurrency.h"
#include "arcane/core/VariableView.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/IMeshEnvironment.h"
#include "arcane/core/materials/IMeshBlock.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/CellToAllEnvCellConverter.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MeshBlockBuildInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialVariableDependInfo.h"
#include "arcane/materials/MatCellVector.h"
#include "arcane/materials/EnvCellVector.h"
#include "arcane/materials/MatConcurrency.h"
#include "arcane/materials/MeshMaterialIndirectModifier.h"
#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"
#include "arcane/materials/ComponentSimd.h"
#include "arcane/materials/MeshMaterialInfo.h"

#include "arcane/accelerator/core/RunQueue.h"

// Inclut le .cc pour avoir la définition des méthodes templates
#include "arcane/tests/StdMeshVariables.cc"

#include <functional>
#include <atomic>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkRunQueues()
{
  info() << "Test checkRunQueues";
  IMeshMaterialMngInternal* mng = m_material_mng->_internalApi();
  eExecutionPolicy mng_policy = mng->runQueue().executionPolicy();
  eExecutionPolicy def_policy = mng->runQueue(eExecutionPolicy::None).executionPolicy();
  eExecutionPolicy seq_policy = mng->runQueue(eExecutionPolicy::Sequential).executionPolicy();
  eExecutionPolicy thread_policy = mng->runQueue(eExecutionPolicy::Thread).executionPolicy();

  if (def_policy != mng_policy)
    ARCANE_FATAL("Bad default execution policy '{0}' '{1}'", def_policy, mng_policy);
  if (seq_policy != eExecutionPolicy::Sequential)
    ARCANE_FATAL("Bad sequential execution policy '{0}'", seq_policy);
  if (thread_policy != eExecutionPolicy::Thread)
    ARCANE_FATAL("Bad thread execution policy '{0}'", thread_policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkNullComponentItem()
{
  EnvCell null_env_cell;
  info() << "NullEnvCell global_cell_id=" << null_env_cell.globalCell().localId();

  info() << "NullEnvCell var_index =" << null_env_cell._varIndex();
  //info() << "NullEnvCell component =" << null_env_cell.component();
  info() << "NullEnvCell component_id =" << null_env_cell.componentId();
  info() << "NullEnvCell null =" << null_env_cell.null();
  info() << "NullEnvCell super_cell =" << null_env_cell.superCell();
  info() << "NullEnvCell level =" << null_env_cell.level();
  info() << "NullEnvCell nb_sub_item=" << null_env_cell.nbSubItem();
  info() << "NullEnvCell component_unique_id=" << null_env_cell.componentUniqueId();
  //info() << "NullEnvCell sub_items =" << null_env_cell.subItems();

  info() << "NullEnvCell all_env_cell =" << null_env_cell.allEnvCell().null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_testDumpProperties()
{
  IMesh* mesh = defaultMesh();
  VariableCellReal v0(VariableBuildInfo(mesh,"VarTestDump0",IVariable::PNoDump));
  MaterialVariableCellReal v1(VariableBuildInfo(mesh,"VarTestDump0"));
  Int32 p0 = v0.variable()->property();
  Int32 p1 = v1.globalVariable().variable()->property();
  info() << "PROP1 = " << p0 << " " << p1;

  MaterialVariableCellReal v2(VariableBuildInfo(mesh,"VarTestDump1"));
  VariableCellReal v3(VariableBuildInfo(mesh,"VarTestDump1",IVariable::PNoDump));
  Int32 p2 = v2.globalVariable().variable()->property();
  Int32 p3 = v3.variable()->property();
  info() << "PROP2 = " << p2 << " " << p3;

  MaterialVariableCellReal v4(VariableBuildInfo(mesh,"VarTestDump2",IVariable::PNoDump));
  Int32 p4 = v4.globalVariable().variable()->property();
  info() << "PROP4 = " << p4;

  if (p0!=p1)
    ARCANE_FATAL("Bad property value p0={0} p1={1}",p0,p1);
  if (p2!=p3)
    ARCANE_FATAL("Bad property value p2={0} p3={1}",p2,p3);

  if ((p0 & IVariable::PNoDump)!=0)
    ARCANE_FATAL("Bad property value p0={0}. Should be Dump",p0);
  if ((p2 & IVariable::PNoDump)!=0)
    ARCANE_FATAL("Bad property value p2={0}. Should be Dump",p2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle le service d'EOS s'il est disponible.
 */
void MeshMaterialTesterModule::
_applyEos(bool is_init)
{
  auto* x = options()->additionalEosService();
  if (!x)
    return;
  ENUMERATE_ENV(ienv,m_material_mng){
    IMeshEnvironment* env = *ienv;
    ENUMERATE_MAT(imat,env){
      IMeshMaterial* mat  = *imat;
      info() << "EOS: mat=" << mat->name();
      ENUMERATE_MATCELL(icell,mat){
        MatCell mc = *icell;
        info() << " v=" << mc.globalCell().uniqueId();
      }
      if (is_init)
        x->initEOS(mat,m_mat_pressure,m_mat_density,m_mat_internal_energy,m_mat_sound_speed);
      else
        x->applyEOS(mat,m_mat_density,m_mat_internal_energy,m_mat_pressure,m_mat_sound_speed);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_initUnitTest()
{
  IUnitTest* unit_test = options()->additionalTestService();
  if (unit_test)
    unit_test->initializeTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
continueInit()
{
  info() << "MESH_MATERIAL_TESTER :: continueInit()";
  _setDependencies();
  _dumpNoDumpRealValues();
  _initUnitTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_setDependencies()
{
  info() << "SET_DEPENDENCIES";

  m_mat_density.setMaterialComputeFunction(this,&MeshMaterialTesterModule::_computeMaterialDepend);
  m_mat_density.addMaterialDepend(m_mat_nodump_real);
  m_mat_density.addMaterialDepend(m_pressure);

  UniqueArray<VariableDependInfo> infos;
  UniqueArray<MeshMaterialVariableDependInfo> mat_infos;
  m_mat_density.materialVariable()->dependInfos(infos,mat_infos);

  for( Integer k=0, n=infos.size(); k<n; ++k )
    info() << "Global depend v=" << infos[k].variable()->fullName();

  for( Integer k=0, n=mat_infos.size(); k<n; ++k )
    info() << "Material depend v=" << mat_infos[k].variable()->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste les itérateurs par partie.
 *
 * Une des deux valeurs \a mat ou \a env doit être non nul.
 * Cela permet via cette méthode de tester à la fois les matériaux et
 * les milieux.
 */
void MeshMaterialTesterModule::
_testComponentPart(IMeshMaterial* mat,IMeshEnvironment* env)
{
  IMeshComponent* component = mat;
  if (!component)
    component = env;

  MaterialVariableCellInt32 test_var(MaterialVariableBuildInfo(m_material_mng,"TestComponentPart"));
  {
    Int32 index = 15;
    ENUMERATE_COMPONENTCELL(iccell,component){
      test_var[iccell] = index;
      ++index;
    }
  }

  Int32 total_full = 0;
  Int32 total_pure = 0;
  Int32 total_impure = 0;
  ENUMERATE_COMPONENTCELL(iccell,component){
    MatVarIndex mvi = iccell._varIndex();
    Int32 v = test_var[iccell];
    if (mvi.arrayIndex()==0)
      total_pure += v;
    else
      total_impure += v;
    total_full += v;
  }
  info() << "COMPONENT=" << component->name() << " TOTAL PURE=" << total_pure << " IMPURE=" << total_impure
         << " FULL=" << total_full;

  ValueChecker vc(A_FUNCINFO);

  if (mat){
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(MatCell,imc,mat){
        total += test_var[imc];
      }
      vc.areEqual(total,total_full,"TotalFull1");
    }
    {
      Int32 total = 0;
      using MyMatPartCell = MatPartCell;
      ENUMERATE_COMPONENTITEM(MyMatPartCell,imc,mat,eMatPart::Impure){
        total += test_var[imc];
      }
      vc.areEqual(total,total_impure,"TotalImpure1");
    }
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(MatPartCell,imc,mat,eMatPart::Pure){
        total += test_var[imc];
      }
      vc.areEqual(total,total_pure,"TotalPure1");
    }
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(MatPartCell,imc,mat->impureMatItems()){
        total += test_var[imc];
    }
      vc.areEqual(total,total_impure,"TotalImpure2");
    }
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(MatPartCell,imc,mat->pureMatItems()){
        total += test_var[imc];
      }
      vc.areEqual(total,total_pure,"TotalPure2");
    }
  }
  if (env){
    {
      Int32 total = 0;
      using MyEnvCell = EnvCell;
      ENUMERATE_COMPONENTITEM(MyEnvCell,imc,env){
        total += test_var[imc];
      }
      vc.areEqual(total,total_full,"TotalFull1");
    }
    {
      Int32 total = 0;
      using MyEnvPartCell = EnvPartCell;
      ENUMERATE_COMPONENTITEM(MyEnvPartCell,imc,env,eMatPart::Impure){
        total += test_var[imc];
      }
      vc.areEqual(total,total_impure,"TotalImpure1");
    }
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(EnvPartCell,imc,env,eMatPart::Pure){
        total += test_var[imc];
      }
      vc.areEqual(total,total_pure,"TotalPure1");
    }
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(EnvPartCell,imc,env->impureEnvItems()){
        total += test_var[imc];
      }
      vc.areEqual(total,total_impure,"TotalImpure2");
    }
    {
      Int32 total = 0;
      ENUMERATE_COMPONENTITEM(EnvPartCell,imc,env->pureEnvItems()){
        total += test_var[imc];
      }
      vc.areEqual(total,total_pure,"TotalPure2");
    }
  }
  {
    Int32 total = 0;
    using MyComponentPartCell = ComponentPartCell;
    ENUMERATE_COMPONENTITEM(MyComponentPartCell,imc,component->impureItems()){
      total += test_var[imc];
    }
    vc.areEqual(total,total_impure,"TotalImpure3");
  }
  {
    Int32 total = 0;
    ENUMERATE_COMPONENTITEM(ComponentPartCell,imc,component->pureItems()){
      total += test_var[imc];
    }
    vc.areEqual(total,total_pure,"TotalPure3");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_doDependencies()
{
  m_pressure.fill(0.0);

  ENUMERATE_MAT(imat,m_material_mng){
    IMeshMaterial* mat = *imat;
    double v = (double)mat->id();
    m_mat_nodump_real.fill(v);
    m_mat_nodump_real.setUpToDate(mat);
    m_mat_density.update(mat);
  }

  // Normalement m_mat_density d'un matériau doit valoir le numéro du matériau
  ENUMERATE_MAT(imat,m_material_mng){
    IMeshMaterial* mat = *imat;
    double v = (double)mat->id();
    ENUMERATE_MATCELL(imc,mat){
      if (m_mat_density[imc]!=v)
        ARCANE_FATAL("Bad value for mat depend v={0} expected={1}",m_mat_density[imc],v);
    }
    ENUMERATE_COMPONENTITEM(MatPartCell,imc,mat,eMatPart::Pure){
      if (m_mat_density[imc]!=v)
        ARCANE_FATAL("Bad value for mat depend v={0} expected={1}",m_mat_density[imc],v);
    }
  }

  // Vérifie que la dépendance sur la variable globale est bien prise en compte
  m_pressure.fill(1.0);
  m_pressure.setUpToDate();
  ENUMERATE_MAT(imat,m_material_mng){
    IMeshMaterial* mat = *imat;
    double v0 = (double)mat->id();
    m_mat_nodump_real.fill(v0);
    m_mat_density.update(mat);
    double v = 1.0 + v0;
    ENUMERATE_MATCELL(imc,mat){
      if (m_mat_density[imc]!=v)
        ARCANE_FATAL("Bad value (1) for global depend v={0} expected={1}",m_mat_density[imc],v);
    }
    ENUMERATE_COMPONENTITEM(MatPartCell,imc,mat,eMatPart::Impure){
      if (m_mat_density[imc]!=v)
        ARCANE_FATAL("Bad value (2) for global depend v={0} expected={1}",m_mat_density[imc],v);
    }
    ENUMERATE_COMPONENTITEM(MatPartCell,imc,mat->impureMatItems()){
      if (m_mat_density[imc]!=v)
        ARCANE_FATAL("Bad value (2) for global depend v={0} expected={1}",m_mat_density[imc],v);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_computeMaterialDepend(IMeshMaterial* mat)
{
  info() << "Compute material depend mat=" << mat->name();
  Integer index = 0;
  ENUMERATE_MATCELL(imc,mat){
    MatCell mc = *imc;
    Cell cell = mc.globalCell();
    m_mat_density[mc] = m_mat_nodump_real[mc] + m_pressure[cell];
    if (index<5){
      info() << "Cell=" << ItemPrinter(cell) << " density=" << m_mat_density[mc]
             << " dump_real=" << m_mat_nodump_real[mc]
             << " pressure=" <<  m_pressure[cell];
      ++index;
    }
  }
  m_mat_density.setUpToDate(mat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_dumpAverageValues()
{
  info() << "_dumpAverageValues()";
  ENUMERATE_ENV(ienv,m_material_mng){
    IMeshEnvironment* env = *ienv;
    ENUMERATE_MAT(imat,env){
      IMeshMaterial* mat = *imat;
      Real sum_density = 0.0;
      ENUMERATE_MATCELL(imatcell,mat){
        //MatCell mc = *icell;
        sum_density += m_mat_density[imatcell];
      }
      info() << "SumMat ITER=" << m_global_iteration() << " MAT=" << mat->name()
             << " density=" << sum_density;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_doSimd()
{
  info() << "_doSimd()";
  MaterialVariableCellReal var_tmp(MaterialVariableBuildInfo(m_material_mng,"TestVarTmpReal"));
  var_tmp.fill(-1.0);
  // TODO: vérifier les valeurs.
  ENUMERATE_ENV(ienv,m_material_mng){
    IMeshEnvironment* env = *ienv;
    auto out_var_tmp = viewOut(var_tmp);
    Real value = (Real)(env->id()) * 4.3;
    ENUMERATE_COMPONENTITEM_LAMBDA(EnvPartSimdCell,ienvcell,env){
      out_var_tmp[ienvcell] = value;
    };
    ENUMERATE_ENVCELL(ienvcell,env){
      if (var_tmp[ienvcell]!=value)
        ARCANE_FATAL("Bad value v={0} expected={1}",var_tmp[ienvcell],value);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_dumpNoDumpRealValues()
{
  OStringStream ostr;
  m_mat_nodump_real.materialVariable()->dumpValues(ostr());
  info() << ostr.str();

  UniqueArray<IMeshMaterialVariable*> vars;
  m_material_mng->fillWithUsedVariables(vars);
  info() << "NB_USED_MATERIAL_VAR=" << vars.size();
  for( Integer i=0, n=vars.size(); i<n; ++i )
    info() << "USED_MATERIAL_VAR name=" << vars[i]->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VectorType> void MeshMaterialTesterModule::
_checkVectorCopy(VectorType& vec_cells)
{
  ValueChecker vc(A_FUNCINFO);
  // Teste la copie.
  // Normalement il s'agit d'une copie par référence donc les vues associées
  // pointent vers la même zone mémoire.
  VectorType vec_cells_copy(vec_cells);
  if (!vec_cells_copy.view()._isSamePointerData(vec_cells.view()))
    ARCANE_FATAL("Bad copy");

  VectorType vec_cells_copy2(vec_cells);
  vc.areEqual(vec_cells_copy2.view()._matvarIndexes(),vec_cells.view()._matvarIndexes(),"bad copy 2");

  VectorType move_vec_cells(std::move(vec_cells_copy2));
  vc.areEqual(move_vec_cells.view()._matvarIndexes().data(),vec_cells.view()._matvarIndexes().data(),"bad move 1");

  {
    // Teste le clone.
    // A la sortie les valeurs des index doivent être les mêmes mais pas les pointeurs.
    VectorType clone_vec(vec_cells_copy.clone());
    vc.areEqual(clone_vec.view()._matvarIndexes(),vec_cells.view()._matvarIndexes(),"bad clone 1");
    if (clone_vec.view()._constituentItemListView() != vec_cells.view()._constituentItemListView())
      ARCANE_FATAL("Bad clone 2");
    if (clone_vec.view()._matvarIndexes().data()==vec_cells.view()._matvarIndexes().data())
      ARCANE_FATAL("bad clone 3");
    if (clone_vec.view()._isSamePointerData(vec_cells.view()))
      ARCANE_FATAL("bad clone 3");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkTemporaryVectors(const CellGroup& test_group)
{

  if (m_mat2){
    MatCellVector mat_cells(test_group,m_mat1);
    const MatCellVector& mcref(mat_cells);
    info() << "SET_DENSITY ON SUB GROUP for material";
    ENUMERATE_MATCELL(imatcell,m_mat1){
      MatCell mc = *imatcell;
      info() << "REF_IDX MAT1 " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
    ENUMERATE_MATCELL(imatcell,m_mat2){
      MatCell mc = *imatcell;
      info() << "REF_IDX MAT2 " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
    info() << "SET_DENSITY";
    ENUMERATE_MATCELL(imatcell,mcref){
      MatCell mc = *imatcell;
      m_mat_density[mc] = 3.2;
      info() << "SET_MAT_DENSITY " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
    _checkVectorCopy(mat_cells);
    ComponentCellVector component_mat_cells(mat_cells);
    _checkVectorCopy(component_mat_cells);
  }

  if (m_mat2){
    MatCellVector mat_cells(test_group.view(),m_mat1);
    const MatCellVector& mcref(mat_cells);
    info() << "SET_DENSITY ON SUB GROUP for material";
    ENUMERATE_MATCELL(imatcell,m_mat1){
      MatCell mc = *imatcell;
      info() << "REF_IDX MAT1 " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
    ENUMERATE_MATCELL(imatcell,m_mat2->matView()){
      MatCell mc = *imatcell;
      info() << "REF_IDX MAT2 " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
    info() << "SET_DENSITY";
    ENUMERATE_MATCELL(imatcell,mcref){
      MatCell mc = *imatcell;
      m_mat_density[mc] = 3.2;
      info() << "SET_MAT_DENSITY " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
    ENUMERATE_MATCELL(imatcell,mat_cells.view()){
      MatCell mc = *imatcell;
      m_mat_density[mc] = 3.2;
      info() << "SET_MAT_DENSITY (VIEW) " << mc._varIndex() << " " << mc.globalCell().uniqueId();
    }
  }

  {
    IMeshEnvironment* env1 = m_mat1->environment();
    EnvCellVector env_cells(test_group,env1);
    const EnvCellVector& ecref(env_cells);
    info() << "SET_DENSITY ON SUB GROUP for environment";
    ENUMERATE_ENVCELL(ienvcell,env1){
      EnvCell mc = *ienvcell;
      info() << "REF_IDX ENV1 " << mc._varIndex() << " uid=" << mc.globalCell().uniqueId();
    }
    info() << "SET_DENSITY";
    ENUMERATE_ENVCELL(ienvcell,ecref){
      EnvCell mc = *ienvcell;
      m_mat_density[ienvcell] = 3.2;
      info() << "SET_ENV_DENSITY " << mc._varIndex() << " uid=" << mc.globalCell().uniqueId();
    }
    ENUMERATE_ENVCELL(ienvcell,ecref.view()){
      EnvCell mc = *ienvcell;
      m_mat_density[ienvcell] = 3.2;
      info() << "SET_ENV_DENSITY (VIEW)" << mc._varIndex() << " uid=" << mc.globalCell().uniqueId();
    }
    _checkVectorCopy(env_cells);
    ComponentCellVector component_env_cells(env_cells);
    _checkVectorCopy(component_env_cells);
  }

  {
    IMeshEnvironment* env1 = m_mat1->environment();
    EnvCellVector env_cells(test_group.view(),env1);
    const EnvCellVector& ecref(env_cells);
    info() << "SET_DENSITY ON SUB GROUP for environment";
    ENUMERATE_ENVCELL(ienvcell,env1){
      EnvCell mc = *ienvcell;
      info() << "REF_IDX ENV1 " << mc._varIndex() << " uid=" << mc.globalCell().uniqueId();
    }
    info() << "SET_DENSITY";
    ENUMERATE_ENVCELL(ienvcell,ecref){
      EnvCell mc = *ienvcell;
      m_mat_density[ienvcell] = 3.2;
      info() << "SET_ENV_DENSITY " << mc._varIndex() << " uid=" << mc.globalCell().uniqueId();
    }
  }
}

      
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkEqual(Integer expected_value,Integer value)
{
  if (value!=expected_value){
    ARCANE_FATAL("Bad value v={0} expected={1}",value,expected_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshMaterialTesterModule::
_fillTestVar(IMeshMaterial* mat,MaterialVariableCellInt64& var)
{
  Integer index = 1;
  Integer total = 0;
  ENUMERATE_MATCELL(imcell,mat){
    var[imcell] = index;
    total += (Integer)var[imcell];
    ++index;
  }
  return total;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshMaterialTesterModule::
_fillTestVar(ComponentItemVectorView view,MaterialVariableCellInt64& var)
{
  Integer index = 1;
  Integer total = 0;
  ENUMERATE_COMPONENTCELL(iccell,view){
    var[iccell] = index;
    total += (Integer)var[iccell];
    ++index;
  }
  return total;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshMaterialTesterModule::
_checkParallelMatItem(MatItemVectorView mat_view,MaterialVariableCellInt64& var)
{
  std::atomic<Integer> new_total;
  auto func = [&](MatItemVectorView view)
  {
    info() << "ParallelLoop with MatItemVectorView size=" << view.nbItem();
    ENUMERATE_MATCELL(iccell,view){
      new_total += (Integer)var[iccell];
    }
  };

  new_total = 0;
  Parallel::Foreach(mat_view,func);
  return (Integer)new_total;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkSubViews(const CellGroup& test_group)
{
  IMeshEnvironment* env1 = m_mat1->environment();
  EnvCellVector env_cells(test_group.view(),env1);
  MatCellVector mat_cells(test_group.view(),m_mat1);

  // Vérifie l'utilisation des sous-vues.
  {
    ComponentItemVectorView test_group_view(env_cells.view());
    Integer nb_item = test_group_view.nbItem();
    for( Integer i=0; i<20; ++i ){
      Integer begin = i * 9;
      Integer size = 12;
      size = math::min(size,nb_item-begin);
      if (size<=0)
        break;
      ComponentItemVectorView v2 = test_group_view._subView(begin,size);
      ENUMERATE_COMPONENTCELL(iccell,v2){
        ComponentCell ccell = *iccell;
        info() << " ComponentCell c=" << ccell._varIndex();
      }
    }
  }

  MaterialVariableCellInt64 mat_test_sub_view(MaterialVariableBuildInfo(m_material_mng,"TestSubViewInt64"));
  Integer direct_total = 0;
  Integer direct_mat_total = 0;
  {
    Integer index = 1;
    ENUMERATE_MATCELL(imcell,mat_cells){
      mat_test_sub_view[imcell] = index;
      ++index;
    }
    index = 1;
    ENUMERATE_ENVCELL(iecell,env_cells){
      mat_test_sub_view[iecell] = index;
      direct_total += index;
      ++index;
    }
    ENUMERATE_MATCELL(imcell,mat_cells){
      direct_mat_total += (Integer)mat_test_sub_view[imcell];
    }
    info() << "DIRECT_ENV_TOTAL = " << direct_total
           << "DIRECT_MAT_TOTAL = " << direct_mat_total;
  }

  {
    ComponentItemVectorView test_group_view(env_cells.view());
    Integer nb_item = test_group_view.nbItem();
    
    for( Integer block_size=1; block_size<20; ++block_size){
      Integer new_total = 0;
      for( Integer begin=0; begin<nb_item; begin += block_size ){
        Integer size = block_size;
        size = math::min(size,nb_item-begin);
        if (size<=0)
          break;
        ComponentItemVectorView v2 = test_group_view._subView(begin,size);
        ENUMERATE_COMPONENTCELL(iccell,v2){
          new_total += (Integer)mat_test_sub_view[iccell];
        }
      }
      if (new_total!=direct_total){
        ARCANE_FATAL("Bad total v={0} expected={1} block_size={2}",
                     new_total,direct_total,block_size);
      }
    }
  }

  // Test en parallèle
  {
    ComponentItemVectorView test_group_view(env_cells.view());

    ParallelLoopOptions options;
    options.setGrainSize(10);

    Integer nb_item = test_group_view.nbItem();
    info() << "ParallelTest with lambda full_size=" << nb_item;

    // Test avec ComponentItemVectorView
    {
      std::atomic<Integer> new_total;
      new_total = 0;
      auto func = [&](ComponentItemVectorView view)
      {
        info() << "ParallelLoop with component size=" << view.nbItem();
        ENUMERATE_COMPONENTCELL(iccell,view){
          new_total += (Integer)mat_test_sub_view[iccell];
        }
      };
      Parallel::Foreach(test_group_view,options,func);

      if (new_total!=direct_total){
        ARCANE_FATAL("Bad total v={0} expected={1}",(Integer)new_total,direct_total);
      }
    }

    // Test avec EnvItemVectorView
    {
      EnvItemVectorView env_test_group_view(env_cells.view());
      std::atomic<Integer> new_total;
      new_total = 0;
      auto func = [&](EnvItemVectorView view)
        {
          info() << "ParallelLoop with environment size=" << view.nbItem();
          ENUMERATE_ENVCELL(iccell,view){
            new_total += (Integer)mat_test_sub_view[iccell];
          }
        };
      Parallel::Foreach(env_test_group_view,options,func);

      if (new_total!=direct_total){
        ARCANE_FATAL("Bad total v={0} expected={1}",(Integer)new_total,direct_total);
      }
    }

    // Test avec MatItemVectorView
    {
      MatItemVectorView mat_view(mat_cells.view());
      Integer ref_val = _fillTestVar(mat_view,mat_test_sub_view);
      Integer new_val = _checkParallelMatItem(mat_view,mat_test_sub_view);
      _checkEqual(ref_val,new_val);
    }

    // Test avec IMeshMaterial
    {
      Integer ref_val = _fillTestVar(m_mat1,mat_test_sub_view);
      Integer new_val = _checkParallelMatItem(m_mat1->matView(),mat_test_sub_view);
      _checkEqual(ref_val,new_val);
    }

    // Test avec MatItemVectorView vide
    {
      MatCellVector empty_mat_cells(CellGroup(),m_mat1);
      MatItemVectorView mat_view(empty_mat_cells.view());
      Integer ref_val = _fillTestVar(mat_view,mat_test_sub_view);
      Integer new_val = _checkParallelMatItem(mat_view,mat_test_sub_view);
      _checkEqual(ref_val,new_val);
    }

  }

  // Test en parallèle avec un pointeur sur membre
  // (teste uniquement cette belle syntaxe que propose le C++ avec std::bind)
  {
    ComponentItemVectorView test_group_view(env_cells.view());
    Integer nb_item = test_group_view.nbItem();
    info() << "NB_ITEM=" << nb_item;
    auto f0 = std::bind(std::mem_fn(&MeshMaterialTesterModule::_subViewFunctor),this,std::placeholders::_1);
    Parallel::Foreach(test_group_view,ParallelLoopOptions(),f0);
    // Syntaxe avec lambda
    Parallel::Foreach(test_group_view,ParallelLoopOptions(),
    [&](ComponentItemVectorView view){ this->_subViewFunctor(view); }
    );
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_subViewFunctor(ComponentItemVectorView view)
{
  ENUMERATE_COMPONENTCELL(iccell,view){
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkFillArrayFromTo(IMeshMaterial* mat,MaterialVariableCellReal& var)
{
  ValueChecker vc(A_FUNCINFO);

  {
    RealUniqueArray values;
    Integer nb_cell = mat->cells().size();
    // Récupère les valeurs de \a var dans \a values puis les mets dans
    // var_tmp et vérifie que tout est OK.
    Integer index = 0;
    values.resize(nb_cell);
    var.fillToArray(mat,values);
    ENUMERATE_MATCELL(imatcell,mat){
      vc.areEqual(var[imatcell],values[index],"Bad value for fillToArray()");
      ++index;
    }
    MaterialVariableCellReal var_tmp(MaterialVariableBuildInfo(m_material_mng,"TestVarTmpReal"));

    index = 0;
    var_tmp.fillFromArray(mat,values);
    ENUMERATE_MATCELL(imatcell,mat){
      vc.areEqual(values[index],var_tmp[imatcell],"Bad value for fillFromArray()");
      ++index;
    }
  }

  {
    std::map<Int32,MatCell> matvar_indexes;
    Int32UniqueArray indexes;
    {
      Integer wanted_index = 0;
      Integer iterator_index = 0;
      ENUMERATE_MATCELL(imatcell,mat){
        if (iterator_index==wanted_index){
          matvar_indexes.insert(std::make_pair(iterator_index,*imatcell));
          indexes.add(iterator_index);
        }
        ++iterator_index;
      }
      info() << "Indexes=" << indexes;
    }

    // Idem test précédent mais sur un sous-ensemble des valeurs

    RealUniqueArray values;
    Integer nb_index = indexes.size();
    values.resize(nb_index);

    var.fillToArray(mat,values,indexes);
    for( Integer i=0; i<nb_index; ++i )
      vc.areEqual(var[matvar_indexes[i]],values[i],"Bad value for fillToArray() (2)");

    MaterialVariableCellReal var_tmp(MaterialVariableBuildInfo(m_material_mng,"TestVarTmpReal2"));

    var_tmp.fillFromArray(mat,values,indexes);
    for( Integer i=0; i<nb_index; ++i )
      vc.areEqual(values[i],var_tmp[matvar_indexes[i]],"Bad value for fillFromArray() (2)");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ContainerType> void MeshMaterialTesterModule::
applyGeneric(const ContainerType& container, MaterialVariableCellReal& var, Real value)
{
  ENUMERATE_GENERIC_CELL (igencell, container) {
    var[igencell] = value;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
compute()
{
  IUnitTest* unit_test = options()->additionalTestService();
  if (unit_test)
    unit_test->executeTest();

  // Si non nul, indique qu'il faut vérifier les valeurs suite à un repartitionnement
  if (m_check_spectral_values_iteration!=0){
    info() << "Check spectral values after loadbalancing";
    _setOrCheckSpectralValues(m_check_spectral_values_iteration,true);
    m_check_spectral_values_iteration = 0;
  }

  // Active la variable une itération sur deux pour tester l'activation et désactivation
  // au cours du temps.
  m_mat_not_used_real.globalVariable().setUsed((m_global_iteration()%2)==0);

  _dumpAverageValues();
  _doDependencies();
  _doSimd();
  _testComponentPart(m_mat1,nullptr);
  if (m_mat2)
    _testComponentPart(m_mat2,nullptr);

  ENUMERATE_ENV(ienv,m_material_mng){
    _testComponentPart(nullptr,(*ienv));
  }

  // Teste la création de variable et les accesseurs.
  using namespace Materials;
  MaterialVariableCellReal mat_pressure(MaterialVariableBuildInfo(m_material_mng,"Pressure"));
  mat_pressure.fill(0.0);

  FaceGroup xmin_group = defaultMesh()->faceFamily()->findGroup("XMIN");
  ENUMERATE_FACE(iface,xmin_group){
    Face face = *iface;
    Cell c = face.boundaryCell();
    Real d = m_density[c];
    m_density[c] = d + 1.0;
  }

  {
    // Teste le constructeur de recopie
    MaterialVariableCellReal mat_pressure2(mat_pressure);
    if (mat_pressure2.materialVariable()!=mat_pressure.materialVariable())
      ARCANE_FATAL("Bad clone");
  }

  {
    // Teste le changement de référence.
    MaterialVariableCellReal mat_test_refersto(MaterialVariableBuildInfo(m_material_mng,"TestRefersToVar"));
    mat_test_refersto.refersTo(mat_pressure);
    if (mat_test_refersto.materialVariable()!=mat_pressure.materialVariable())
      ARCANE_FATAL("Bad refersTo");
    Real total = 0.0;
    Real total_ref = 0.0;
    ENUMERATE_MATCELL(imatcell,m_mat1){
      total += mat_test_refersto[imatcell];
      total_ref += mat_pressure[imatcell];
    }
    if (total!=total_ref)
      ARCANE_FATAL("Bad value for total using refersTo");

  }

  // Teste le ENUMERATE_GENERIC_CELL
  {
    MatCellVector mat_cells(ownCells(),m_mat1);
    const MatCellVector& mcref(mat_cells);
    MaterialVariableCellReal var(MaterialVariableBuildInfo(m_material_mng,"VarTestGeneric"));
    applyGeneric(mcref,var,4.5);
    ENUMERATE_MATCELL(imatcell,mcref){
      if (var[imatcell]!=4.5)
        ARCANE_FATAL("Bad value 1 for TestGeneric");
    }
    applyGeneric(allCells(),var,3.2);
    ENUMERATE_CELL(icell,allCells()){
      if (var[icell]!=3.2)
        ARCANE_FATAL("Bad value 2 for TestGeneric");
    }
    applyGeneric(m_mat1,var,7.6);
    ENUMERATE_MATCELL(imatcell,m_mat1){
      if (var[imatcell]!=7.6)
        ARCANE_FATAL("Bad value 3 for TestGeneric");
    }
    applyGeneric(m_mat1->environment(),var,4.2);
    ENUMERATE_ENVCELL(imatcell,m_mat1->environment()){
      if (var[imatcell]!=4.2)
        ARCANE_FATAL("Bad value 4 for TestGeneric");
    }
  }

  // Teste le remplissage des valeurs partielles.
  _checkFillPartialValues();

  IMeshMaterialVariable* nv = m_material_mng->findVariable(m_pressure.variable()->fullName());
  if (!nv)
    fatal() << "Can not find MeshVariable (F1)";

  IMeshMaterialVariable* nv2 = m_material_mng->findVariable("Pressure");
  if (!nv2)
    fatal() << "Can not find MeshVariable (F2)";

  ENUMERATE_MATCELL(imatcell,m_mat1){
    MatCell mmc = *imatcell;
    MatVarIndex mvi = mmc._varIndex();
    info() << "CELL IN MAT1 i=" << imatcell.index() << " vindex=" << mvi.arrayIndex() << " i=" << mvi.valueIndex();
    mat_pressure[mmc] += 0.2;
  }

  if (m_mat2){
    ENUMERATE_MATCELL(imatcell,m_mat2){
      MatCell mmc = *imatcell;
      MatVarIndex mvi = mmc._varIndex();
      info() << "CELL IN MAT2 vindex=" << mvi.arrayIndex() << " i=" << mvi.valueIndex();
      //mat_pressure[mmc] -= 0.2;
      mat_pressure[imatcell] -= 0.2;
    }
  }

  _checkFillArrayFromTo(m_mat1,mat_pressure);
  if (m_mat2)
    _checkFillArrayFromTo(m_mat2,mat_pressure);

  ENUMERATE_ENV(ienv,m_material_mng){
    IMeshEnvironment* env = *ienv;
    ENUMERATE_ENVCELL(ienvcell,env){
      EnvCell ev = *ienvcell;
      MatVarIndex mvi = ev._varIndex();
      info() << "CELL IN ENV vindex=" << mvi.arrayIndex() << " i=" << mvi.valueIndex();
      mat_pressure[ev] += 3.0;
      mat_pressure[ienvcell] += 3.0;
    }
    ENUMERATE_COMPONENTCELL(icmpcell,env){
      ComponentCell cv = *icmpcell;
      MatVarIndex mvi = cv._varIndex();
      info() << "CELL IN ENV WITH COMPONENT vindex=" << mvi.arrayIndex() << " i=" << mvi.valueIndex();
      mat_pressure[cv] += 3.0;
      EnvCell env_cell(cv);
      if (env_cell._varIndex()!=cv._varIndex())
        ARCANE_FATAL("Bad cell");
    }
  }
  _computeDensity();
  _checkArrayVariableSynchronize();

  for( Integer i=0, n=m_material_mng->materials().size(); i<n; ++i ){
    IMeshMaterial* mat = m_material_mng->materials()[i];
    m_density_post_processing[i]->copy(m_mat_density.globalVariable());
    _copyPartialToGlobal(mat,*m_density_post_processing[i],m_mat_density);
  }

  // Supprime des mailles pour test
  {
    info() << "CheckRemove: Cells in MAT1=" << m_mat1->cells().size();
    ENUMERATE_MATCELL(imatcell,m_mat1){
      MatCell mmc = *imatcell;
      MatVarIndex mvi = mmc._varIndex();
      info() << "CheckRemove: CELL IN MAT1 i=" << imatcell.index() << " vindex=" << mvi.arrayIndex() << " i=" << mvi.valueIndex()
             << " lid=" << mmc.envCell().globalCell();
    }

    if (m_mat2)
      info() << "Cells in MAT2=" << m_mat2->cells().size();

    Int32UniqueArray remove_lids;
    
    Int64 last_uid = m_nb_starting_cell() - (m_global_iteration()*30);
    info() << "LAST_UID_TO_REMOVE=" << last_uid;

    ENUMERATE_CELL(icell,allCells()){
      Cell c = *icell;
      if (c.uniqueId()>last_uid)
        remove_lids.add(c.localId());
    }

    info() << "Removing cells n=" << remove_lids.size();
    mesh()->modifier()->setDynamic(true);
    mesh()->modifier()->removeCells(remove_lids);
    if (parallelMng()->isParallel()){
      // En parallèle, comme on supprime les mailles un peu n'importe comment,
      // on supprime les tests tant que le maillage n'est pas à jour.
      // TODO: regarder pourquoi le test checkValidMesh() plante.
      Integer check_level = mesh()->checkLevel();
      mesh()->setCheckLevel(0);
      mesh()->modifier()->endUpdate();
      {
        MeshMaterialIndirectModifier mmim(m_material_mng);
        mmim.beginUpdate();
        info() << "MESH_MATERIAL_TEST: UpdateGhostLayers";
        mesh()->modifier()->updateGhostLayers();
        if ((m_global_iteration() % 2)==0){
          mmim.endUpdateWithSort();
          // TODO: vérifier que tout est trié
        }
        else
          mmim.endUpdate();
      }
      mesh()->setCheckLevel(check_level);
      //mesh()->checkValidMesh();
    }
    else
      mesh()->modifier()->endUpdate();

    info() << "End removing cells nb_cell=" << mesh()->nbCell();
  }
  if (m_mesh_partitioner){
    Integer iteration = m_global_iteration();
    // Lance un repartitionnement toute les 3 itérations.
    if ((iteration%3)==0){
      info() << "Registering mesh partition";
      subDomain()->timeLoopMng()->registerActionMeshPartition(m_mesh_partitioner);
      m_check_spectral_values_iteration = (iteration*2)+1; 
     _setOrCheckSpectralValues(m_check_spectral_values_iteration,false);
    }
  }
  {
    // Initialise la densité et l'energie interne dans les nouvelles mailles.
    ENUMERATE_MAT(imat,m_material_mng){
      Materials::IMeshMaterial* mat = *imat;
      ENUMERATE_MATCELL(icell,mat){
        MatCell mc = *icell;
        if (m_mat_density[mc] == 0.0)
          m_mat_density[mc] = 50.0;
        if (m_mat_internal_energy[mc] == 0.0)
          m_mat_internal_energy[mc] = 1.0;
      }
    }
    _applyEos(false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkFillPartialValues()
{
  // Teste le remplissage des valeurs partielles par les valeurs globales.
  info() << "Check MaterialVariableCellReal";
  MaterialVariableCellReal mat_var(MaterialVariableBuildInfo(m_material_mng,"TestFillPartialMat"));
  _checkFillPartialValuesHelper(mat_var);

  info() << "Check EnvironmentVariableCellReal";
  EnvironmentVariableCellReal env_var(MaterialVariableBuildInfo(m_material_mng,"TestFillPartialEnv"));
  _checkFillPartialValuesHelper(env_var);
  if (m_material_mng->isAllocateScalarEnvironmentVariableAsMaterial()){
    MaterialVariableCellReal mat_env_var(MaterialVariableBuildInfo(m_material_mng,"TestFillPartialEnv"));
    info() << "Ok for creating Material variable with same name as Environment variable";
  }

  info() << "Check MaterialVariableCellReal";
  MaterialVariableCellArrayReal mat_var2(MaterialVariableBuildInfo(m_material_mng,"TestFillPartialMat2"));
  mat_var2.resize(5);
  _checkFillPartialValuesHelper(mat_var2);

  info() << "Check EnvironmentVariableCellArrayReal";
  EnvironmentVariableCellArrayReal env_var2(MaterialVariableBuildInfo(m_material_mng,"TestFillPartialEnv2"));
  env_var2.resize(12);
  _checkFillPartialValuesHelper(env_var2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VarType> void MeshMaterialTesterModule::
_checkFillPartialValuesHelper(VarType& mat_var)
{
  info() << "Check fillPartialValuesWithGlobalValues()";
  _fillVar(mat_var,3.0);
  mat_var.materialVariable()->fillPartialValuesWithGlobalValues();
  _checkFillPartialValuesWithGlobal(mat_var,m_material_mng->components());

  info() << "Check fillPartialValuesWithSuperValues(LEVEL_ALLENVIRONMENT)";
  _fillVar(mat_var,7.0);
  mat_var.fillPartialValuesWithSuperValues(LEVEL_ALLENVIRONMENT);
  _checkFillPartialValuesWithGlobal(mat_var,m_material_mng->components());

  info() << "Check fillPartialValuesWithSuperValues(LEVEL_ENVIRONMENT)";
  _fillVar(mat_var,-2.0);
  mat_var.fillPartialValuesWithSuperValues(LEVEL_ENVIRONMENT);
  _checkFillPartialValuesWithSuper(mat_var,m_material_mng->environmentsAsComponents());

  info() << "Check fillPartialValuesWithSuperValues(LEVEL_MATERIAl)";
  _fillVar(mat_var,-25.0);
  mat_var.fillPartialValuesWithSuperValues(LEVEL_MATERIAL);
  _checkFillPartialValuesWithSuper(mat_var,m_material_mng->materialsAsComponents());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void _setValue(Real& var_ref,Real value)
{
  var_ref = value;
}
void _setValue(RealArrayView var_ref,Real value)
{
  Integer n = var_ref.size();
  for( Integer i=0; i<n; ++i ){
    var_ref[i] = value*((Real)(i+1));
  }
}
}
template<typename VarType> void MeshMaterialTesterModule::
_fillVar(VarType& var_type,Real base_value)
{
  MeshComponentList components = m_material_mng->components();
  MatVarSpace var_space = var_type.space();
  Int32 index = 0;
  ENUMERATE_COMPONENT(ic,components){
    IMeshComponent* c = *ic;
    if (!c->hasSpace(var_space))
      continue;
    ENUMERATE_COMPONENTCELL(iccell,c){
      ++index;
      _setValue(var_type[iccell],(base_value + (Real)index));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VarType> void MeshMaterialTesterModule::
_checkFillPartialValuesWithGlobal(const VarType& var_type,MeshComponentList components)
{
  ValueChecker vc(A_FUNCINFO);
  MatVarSpace var_space = var_type.space();

  ENUMERATE_COMPONENT(ic,components){
    IMeshComponent* c = *ic;
    if (!c->hasSpace(var_space))
      continue;
    ENUMERATE_COMPONENTCELL(iccell,c){
      Cell c = (*iccell).globalCell();
      auto ref_value = var_type[c];
      auto my_value = var_type[iccell];
      vc.areEqual(my_value,ref_value,"Bad fill value with global");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VarType> void MeshMaterialTesterModule::
_checkFillPartialValuesWithSuper(const VarType& var_type,MeshComponentList components)
{
  ValueChecker vc(A_FUNCINFO);
  MatVarSpace var_space = var_type.space();

  ENUMERATE_COMPONENT(ic,components){
    IMeshComponent* c = *ic;
    if (!c->hasSpace(var_space))
      continue;
    ENUMERATE_COMPONENTCELL(iccell,c){
      ComponentCell c = (*iccell).superCell();
      auto ref_value = var_type[c];
      auto my_value = var_type[iccell];
      vc.areEqual(my_value,ref_value,"Bad fill value with super");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VarType1,typename VarType2,typename VarType3> void MeshMaterialTesterModule::
_setOrCheckSpectralValues2(VarType1& var_real,VarType2& var_int32,VarType3& var_scalar_int32,
                           Int64 iteration,bool is_check)
{
  typedef std::function<void(Int64,Int64,ComponentItemLocalId)> FunctorType;

  Int32 spectral1_dim2 = var_real.globalVariable().arraySize();
  Int32 spectral2_dim2 = var_int32.globalVariable().arraySize();
  info() << "SET_OR_CHECK size1=" << spectral1_dim2 << " size2=" << spectral2_dim2;
  FunctorType set_func = [&](Int64 uid,Int64 iteration,ComponentItemLocalId var_index)
  {
    Int64 component_idx = var_index.localId().arrayIndex();
    Int64 base = uid + iteration + (component_idx+1);
    for( Integer i=0; i<spectral1_dim2; ++i )
      var_real[var_index][i] = Convert::toReal(2.0 + (Real)(base * spectral1_dim2 + i));
    for( Integer i=0; i<spectral2_dim2; ++i )
      var_int32[var_index][i] = (Int32)(3 + (base * spectral2_dim2 + i*2 ));
    var_scalar_int32[var_index] = (Int32)(3 + (base * spectral2_dim2));
  };

  ValueChecker vc(A_FUNCINFO);
  vc.setThrowOnError(false);

  FunctorType check_func = [&](Int64 uid,Int64 iteration,ComponentItemLocalId var_index)
  {
    Int64 component_idx = var_index.localId().arrayIndex();
    Int64 base = uid + iteration + (component_idx+1);
    for( Integer i=0; i<spectral1_dim2; ++i ){
      Real ref1 = Convert::toReal(2.0 + (Real)(base * spectral1_dim2 + i));
      //info() << "CHECK1 var=" << var_real.name() << " idx=" << var_index << " v1=" << ref1 << " v2=" << var_real[var_index][i];
      vc.areEqual(ref1,var_real[var_index][i],String::format("spectral1:{0}",var_real.name()));
    }
    for( Integer i=0; i<spectral2_dim2; ++i ){
      Int32 ref2 = (Int32)(3 + (base * spectral2_dim2 + i*2 ));
      //info() << "CHECK2 var=" << var_real.name() << " idx=" << var_index << " v1=" << ref2 << " v2=" << var_int32[var_index][i];
      vc.areEqual(ref2,var_int32[var_index][i],String::format("spectral2:{0}",var_int32.name()));
    }
    Int32 ref3 = (Int32)(3 + (base * spectral2_dim2));
    vc.areEqual(ref3,var_scalar_int32[var_index],"scalar1");
    if (vc.nbError()!=0){
      error() << "Error for cell uid=" << uid << " var_index=" << var_index;
      vc.throwIfError();
    }
  };

  FunctorType func = (is_check) ? check_func : set_func;

  bool has_mat = var_real.materialVariable()->space()!=MatVarSpace::Environment;
  ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng,allCells()){
    AllEnvCell all_env_cell = *iallenvcell;
    Cell global_cell = all_env_cell.globalCell();
    Int64 cell_uid = global_cell.uniqueId();
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      func(cell_uid,iteration,ienvcell);
      if (has_mat){
        ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
          func(cell_uid,iteration,imatcell);
        }
      }
    }
    func(cell_uid,iteration,all_env_cell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_setOrCheckSpectralValues(Int64 iteration,bool is_check)
{
  _setOrCheckSpectralValues2(m_mat_spectral1,m_mat_spectral2,m_mat_int32,iteration,is_check);
  _setOrCheckSpectralValues2(m_env_spectral1,m_env_spectral2,m_env_int32,iteration,is_check);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkArrayVariableSynchronize()
{
  info() << "_checkArrayVariableSynchronize(): SYNCHRONIZE_MATERIALS";
  m_material_mng->synchronizeMaterialsInCells();
  m_material_mng->checkMaterialsInCells();

  Int64 iteration = m_global_iteration();

  _setOrCheckSpectralValues(iteration,false);

  // On utilise la synchro par liste une itération sur deux.
  if ((iteration % 2)==0){
    MeshMaterialVariableSynchronizerList mlist(m_material_mng);
    m_mat_spectral1.synchronize(mlist);
    m_mat_spectral2.synchronize(mlist);
    m_env_spectral1.synchronize(mlist);
    m_env_spectral2.synchronize(mlist);
    m_mat_int32.synchronize();
    m_env_int32.synchronize();
    mlist.apply();
  }
  else{
    m_mat_spectral1.synchronize();
    m_mat_spectral2.synchronize();
    m_env_spectral1.synchronize();
    m_env_spectral2.synchronize();
    m_mat_int32.synchronize();
    m_env_int32.synchronize();
  }

  _setOrCheckSpectralValues(iteration,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_computeDensity()
{
  VariableCellReal tmp_cell_mat_density(VariableBuildInfo(defaultMesh(),"TmpCellMatDensity"));
  VariableNodeReal tmp_node_mat_density(VariableBuildInfo(defaultMesh(),"TmpNodeMatDensity"));
  
  Int32UniqueArray mat_to_add_array;
  Int32UniqueArray mat_to_remove_array;
  // Calcul les mailles dans lesquelles il faut ajouter ou supprimer des matériaux
  {
    Materials::MeshMaterialModifier modifier(m_material_mng);
    ENUMERATE_ENV(ienv,m_material_mng){
      IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT(imat,env){
        IMeshMaterial* mat = *imat;
        mat_to_add_array.clear();
        mat_to_remove_array.clear();
        _fillDensity(*imat,tmp_cell_mat_density,tmp_node_mat_density,mat_to_add_array,mat_to_remove_array,true);
        info() << "FILL_DENSITY_INFO ITER=" << m_global_iteration()
               << " mat=" << mat->name()
               << " nb_to_add=" << mat_to_add_array.size()
               << " nb_to_remove=" << mat_to_remove_array.size();
        
        modifier.removeCells(mat,mat_to_remove_array);
        modifier.addCells(mat,mat_to_add_array);
      }
    }
  }

  // Met à jour les valeurs.
  {
    ENUMERATE_ENV(ienv,m_material_mng){
      IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT(imat,env){
        _fillDensity(*imat,tmp_cell_mat_density,tmp_node_mat_density,mat_to_add_array,mat_to_remove_array,false);
      }
    }
  }
  // Pour que les synchronisations fonctionnent bien,
  // il faut que les matériaux soient les mêmes dans toutes les mailles.
  m_material_mng->synchronizeMaterialsInCells();
  info() << "Synchronize density";
  m_mat_density.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_copyPartialToGlobal(IMeshMaterial* mat,VariableCellReal& global_density,
                     MaterialVariableCellReal& partial_variable)
{
  ENUMERATE_MATCELL(imatcell,mat){
    MatCell mc = *imatcell;
    Cell global_cell = mc.globalCell();
    global_density[global_cell] = partial_variable[mc];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Dans cet exemple, on travaille sur les valeurs partielles d'un matériau.
 * Cette méthode est appelée 2 fois:
 * - la première fois, \a is_compute_mat vaut true et on
 * indique dans \a mat_to_add_array et \a mat_to_remove_array la liste
 * des mailles qui vont être ajoutées ou supprimer pour ce matériau.
 * Cette liste est déterminée en fonction de la valeur de la densité
 * partielle dans les mailles voisines.
 * - la seconde fois, on met à jour la valeur partielle. On ne peut
 * pas le faire lors du premier appel car on ne peut pas remplir les
 * valeurs partielles dans les mailles qui ne possèdent pas encore le
 * matériaux (TODO: prévoir un mécanisme pour éviter cela).
 */
void MeshMaterialTesterModule::
_fillDensity(IMeshMaterial* mat,VariableCellReal& tmp_cell_mat_density,
             VariableNodeReal& tmp_node_mat_density,
             Int32Array& mat_to_add_array,Int32Array& mat_to_remove_array,
             bool is_compute_mat)
{
  Int32 mat_id = mat->id();
  tmp_cell_mat_density.fill(0.0);
  tmp_node_mat_density.fill(0.0);
  info() << "FILL MAT=" << mat->name();

  // Copy dans tmp_cell_mat_density la valeur partielle de la densité pour \a mat
  _copyPartialToGlobal(mat,tmp_cell_mat_density,m_mat_density);

  // La valeur aux noeuds est la moyenne de la valeur aux mailles autour
  ENUMERATE_NODE(inode,allNodes()){
    Real v = 0.0;
    for( CellLocalId icell : inode->cellIds() )
      v += tmp_cell_mat_density[icell];
    v /= (Real)inode->nbCell();
    v *= 0.8;
    tmp_node_mat_density[inode] = v;
  }
  
  // Phase1, calcule les mailles où le matériau sera créé ou supprimé.
  // Cela se fait en fonction de certaines valeurs (arbitraires) de la densité.
  if (is_compute_mat){
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Real current_density = tmp_cell_mat_density[icell];
      Real new_density = 0.0;
      bool has_material = (m_present_material[cell] & (1<<mat_id));
      for( NodeLocalId inode : cell.nodeIds() )
        new_density += tmp_node_mat_density[inode];
      new_density /= (Real)cell.nbNode();
      if (new_density>=0.5 && !has_material && current_density==0.0){
        mat_to_add_array.add(cell.localId());
        info() << "NEW CELL FOR MAT " << mat_id << " uid=" << ItemPrinter(cell) << " new=" << new_density;
      }
      else if (new_density<0.4 && has_material && current_density>0.4){
        mat_to_remove_array.add(cell.localId());
        info() << "REMOVE CELL FOR MAT " << mat_id << " uid=" << ItemPrinter(cell) << " new=" << new_density
               << " old=" << current_density;
      }
      tmp_cell_mat_density[icell] = new_density;
    }
  }
  else{
    // Phase2: met à jour les valeurs maintenant que le matériau a été
    // ajouté dans toutes les mailles

    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Real new_density = 0.0;
      for( NodeLocalId inode : cell.nodeIds() )
        new_density += tmp_node_mat_density[inode];
      new_density /= (Real)cell.nbNode();
      tmp_cell_mat_density[icell] = new_density;
    }

    ENUMERATE_MATCELL(imatcell,mat){
      MatCell mc = *imatcell;
      Cell global_cell = mc.globalCell();
      //Real density = tmp_cell_mat_density[global_cell];
      //info() << "ASSIGN cell=" << global_cell.uniqueId() << " density=" << density;
      m_mat_density[mc] = tmp_cell_mat_density[global_cell];
    }
  }
  {
    StdMeshVariables<MeshMaterialVariableTraits> xm(meshHandle(),"Base1","Base2");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkCreation2(Integer a,Integer n)
{
  Int64 z = 0;
  info() << "I=" << a << " N=" << n;
  for( Integer i=0; i<n; ++i ){
    MaterialVariableCellReal mat_pressure(MaterialVariableBuildInfo(m_material_mng,"Pressure"));
    z += mat_pressure.materialVariable()->name().length();
  }
  info() << "Z=" << z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test la création à la volée des variables, avec multi-threading.
 */
void MeshMaterialTesterModule::
_checkCreation()
{
  Integer n = 0;
  ParallelLoopOptions options;
  options.setGrainSize(20);
  arcaneParallelFor(0,1000,options,[&](Integer a,Integer n) { _checkCreation2(a,n); });

  info() << "CHECK CREATE N=" << n;

  ValueChecker vc(A_FUNCINFO);
  {
    MaterialVariableCellReal mat_pressure(MaterialVariableBuildInfo(m_material_mng,"Pressure"));
    MaterialVariableCellArrayReal mat_pressure_array(MaterialVariableBuildInfo(m_material_mng, "PressureArray"));
    Int32 nb_dim2 = 5;
    mat_pressure_array.resize(nb_dim2);
    mat_pressure.fill(0.0);
    ENUMERATE_MATCELL(imatcell,m_mat1){
      MatCell mmc = *imatcell;
      mat_pressure[mmc] += 0.2;
      for (Int32 i = 0; i < nb_dim2; ++i)
        mat_pressure_array[mmc][i] = 0.3 + static_cast<Real>(i);
    }
    // Teste la création à partir d'une variable existante.
    MaterialVariableCellReal mat_pressure_ref2(mat_pressure.materialVariable());
    MaterialVariableCellArrayReal mat_pressure_array_ref2(mat_pressure_array.materialVariable());
    ENUMERATE_MATCELL (imatcell, m_mat1) {
      MatCell mmc = *imatcell;
      vc.areEqual(mat_pressure[mmc], mat_pressure_ref2[mmc], "Bad Pressure");
      for (Int32 i = 0; i < nb_dim2; ++i)
        vc.areEqual(mat_pressure_array[mmc][i], mat_pressure_array_ref2[mmc][i], "Bad Pressure Array");
    }
    bool is_ok = false;
    try{
      MaterialVariableCellInt32 mat_pressure_ref3(mat_pressure.materialVariable());
    }
    catch(const FatalErrorException& ex){
      is_ok = true;
    }
    if (!is_ok)
      ARCANE_FATAL("Should launch exception for 'MaterialVariableCellInt32' conversion");
    is_ok = false;
    try{
      MaterialVariableCellArrayInt32 mat_pressure_ref3(mat_pressure.materialVariable());
    }
    catch(const FatalErrorException& ex){
      is_ok = true;
    }
    if (!is_ok)
      ARCANE_FATAL("Should launch exception for 'MaterialVariableCellArrayInt32' conversion");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
