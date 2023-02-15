// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialAcceleratorUnitTest.cc                          (C) 2000-2023 */
/*                                                                           */
/* Service de test unitaire du support accelerateurs des matériaux/milieux.  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/FactoryService.h"
#include "arcane/VariableView.h"

#include "arcane/materials/ComponentSimd.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MatCellVector.h"
#include "arcane/materials/EnvCellVector.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshEnvironmentVariableRef.h"
#include "arcane/materials/EnvItemVector.h"

#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/MaterialVariableViews.h"
#include "arcane/accelerator/RunCommandMaterialEnumerate.h"
#include "arcane/accelerator/AsyncRunQueuePool.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Materials;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test unitaire du support accelerateurs des matériaux/milieux.
 */
class MeshMaterialAcceleratorUnitTest
: public BasicUnitTest
{
 public:

  explicit MeshMaterialAcceleratorUnitTest(const ServiceBuildInfo& cb);
  ~MeshMaterialAcceleratorUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

  IMeshMaterialMng* m_mm_mng;
  IMeshEnvironment* m_env1;

  MaterialVariableCellReal m_mat_a_ref;
  MaterialVariableCellReal m_mat_b_ref;
  MaterialVariableCellReal m_mat_c_ref;
  MaterialVariableCellReal m_mat_d_ref;
  MaterialVariableCellReal m_mat_e_ref;

  MaterialVariableCellReal m_mat_a;
  MaterialVariableCellReal m_mat_b;
  MaterialVariableCellReal m_mat_c;
  MaterialVariableCellReal m_mat_d;
  MaterialVariableCellReal m_mat_e;

  UniqueArray<Int32> m_env1_pure_value_index;
  UniqueArray<Int32> m_env1_partial_value_index;
  EnvCellVector* m_env1_as_vector;
  Int32 m_nb_z;

  void _initializeVariables();

 public:

  // Les méthodes suivantes doivent être publiques pour
  // sur accélérateur

  void _executeTest1(Integer nb_z);
  void _executeTest2(Integer nb_z);
  void _executeTest3(Integer nb_z);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(MeshMaterialAcceleratorUnitTest,
                                           IUnitTest,MeshMaterialAcceleratorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialAcceleratorUnitTest::
MeshMaterialAcceleratorUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_mm_mng(nullptr)
, m_env1(nullptr)
, m_mat_a_ref(VariableBuildInfo(mesh(),"MatA_ref"))
, m_mat_b_ref(VariableBuildInfo(mesh(),"MatB_ref"))
, m_mat_c_ref(VariableBuildInfo(mesh(),"MatC_ref"))
, m_mat_d_ref(VariableBuildInfo(mesh(),"MatD_ref"))
, m_mat_e_ref(VariableBuildInfo(mesh(),"MatE_ref"))
, m_mat_a(VariableBuildInfo(mesh(),"MatA"))
, m_mat_b(VariableBuildInfo(mesh(),"MatB"))
, m_mat_c(VariableBuildInfo(mesh(),"MatC"))
, m_mat_d(VariableBuildInfo(mesh(),"MatD"))
, m_mat_e(VariableBuildInfo(mesh(),"MatE"))
, m_env1_as_vector(nullptr)
, m_nb_z(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialAcceleratorUnitTest::
~MeshMaterialAcceleratorUnitTest()
{
  delete m_env1_as_vector;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
initializeTest()
{
  IApplication* app = subDomain()->application();
  const auto& acc_info = app->acceleratorRuntimeInitialisationInfo();
  initializeRunner(m_runner,traceMng(),acc_info);

  m_mm_mng = IMeshMaterialMng::getReference(mesh());

  // Lit les infos des matériaux du JDD et les enregistre dans le gestionnaire
  UniqueArray<String> mat_names = { "MAT1", "MAT2", "MAT3" };
  for( String v : mat_names.range() ){
    m_mm_mng->registerMaterialInfo(v);
  }

  {
    Materials::MeshEnvironmentBuildInfo env_build("ENV1");
    env_build.addMaterial("MAT1");
    env_build.addMaterial("MAT2");
    m_mm_mng->createEnvironment(env_build);
  }
  {
    Materials::MeshEnvironmentBuildInfo env_build("ENV2");
    env_build.addMaterial("MAT2");
    env_build.addMaterial("MAT3");
    m_mm_mng->createEnvironment(env_build);
  }

  m_mm_mng->endCreate(false);

  IMeshEnvironment* env1 = m_mm_mng->environments()[0];
  IMeshEnvironment* env2 = m_mm_mng->environments()[1];

  m_env1 = env1;

  IMeshMaterial* mat1 = env1->materials()[0];
  IMeshMaterial* mat2 = env2->materials()[1];

  {
    Int32UniqueArray env1_indexes;
    Int32UniqueArray mat2_indexes;
    Int32UniqueArray sub_group_indexes;
    Integer nb_cell = ownCells().size();
    Int64 total_nb_cell = nb_cell;
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Int64 cell_index = cell.uniqueId();
      if (cell_index<((2*total_nb_cell)/3)){
        env1_indexes.add(icell.itemLocalId());
      }
      if (cell_index<(total_nb_cell/2) && cell_index>(total_nb_cell/3)){
        mat2_indexes.add(icell.itemLocalId());
      }
      if ((cell_index%2)==0)
        sub_group_indexes.add(icell.itemLocalId());
    }

    // Ajoute les mailles du milieu 1
    {
      Materials::MeshMaterialModifier modifier(m_mm_mng);
      Materials::IMeshEnvironment* env = mat1->environment();
      // Ajoute les mailles du milieu
      //modifier.addCells(env,env1_indexes);
      Int32UniqueArray mat1_indexes;
      Int32UniqueArray mat2_indexes;
      Integer nb_cell = env1_indexes.size();
      for( Integer z=0; z<nb_cell; ++z ){
        bool add_to_mat1 = (z<(nb_cell/2) && z>(nb_cell/4));
        bool add_to_mat2 = (z>=(nb_cell/2) || z<(nb_cell/3));
        if (add_to_mat1){
          mat1_indexes.add(env1_indexes[z]);
        }
        if (add_to_mat2)
          mat2_indexes.add(env1_indexes[z]);
      }
      // Ajoute les mailles du matériau 1
      modifier.addCells(mat1,mat1_indexes);
      Integer nb_mat = env->nbMaterial();
      if (nb_mat>1)
        // Ajoute les mailles du matériau 2
        modifier.addCells(m_mm_mng->environments()[0]->materials()[1],mat2_indexes);
    }
    // Ajoute les mailles du milieu 2
    if (mat2){
      Materials::MeshMaterialModifier modifier(m_mm_mng);
      //modifier.addCells(m_mat2->environment(),mat2_indexes);
      modifier.addCells(mat2,mat2_indexes);
    }
  }

  for( IMeshEnvironment* env : m_mm_mng->environments().range() ){
    info() << "** ** ENV name=" << env->name() << " nb_item=" << env->view().nbItem();
    Integer nb_pure_env = 0;
    ENUMERATE_ENVCELL(ienvcell,env){
      if ( (*ienvcell).allEnvCell().nbEnvironment()==1 )
        ++nb_pure_env;
    }
    info() << "** ** NB_PURE=" << nb_pure_env;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
executeTest()
{
  m_env1_pure_value_index.clear();
  m_env1_partial_value_index.clear();
  Integer nb_unknown = 0;
  {
    ENUMERATE_ENVCELL(ienvcell,m_env1){
      EnvCell env_cell = *ienvcell;
      MatVarIndex mvi = env_cell._varIndex();
      Integer nb_env = env_cell.allEnvCell().nbEnvironment();
      if ( nb_env==1 )
        m_env1_pure_value_index.add(mvi.valueIndex());
      else if (nb_env>1)
        m_env1_partial_value_index.add(mvi.valueIndex());
      else
        ++nb_unknown;
    }
  }
  m_env1_as_vector = new EnvCellVector(m_env1->cells(),m_env1);

  Integer nb_z = 10000;
  if (arcaneIsDebug())
    nb_z /= 100;
  Integer nb_z2 = nb_z / 5;

  Int32 env_idx = m_env1->variableIndexer()->index() + 1;
  info() << "ENV_IDX=" << env_idx
         << " nb_pure=" << m_env1_pure_value_index.size()
         << " nb_partial=" << m_env1_partial_value_index.size()
         << " nb_unknown=" << nb_unknown
         << " nb_z=" << nb_z << " nb_z2=" << nb_z2;

  _initializeVariables();
  {
    _executeTest1(nb_z);
  }
  {
    _executeTest2(nb_z);
  }
  {
    _executeTest3(nb_z);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
_initializeVariables()
{
  MaterialVariableCellReal& a_ref(m_mat_a_ref);
  MaterialVariableCellReal& b_ref(m_mat_b_ref);
  MaterialVariableCellReal& c_ref(m_mat_c_ref);
  MaterialVariableCellReal& d_ref(m_mat_d_ref);
  MaterialVariableCellReal& e_ref(m_mat_e_ref);

  MaterialVariableCellReal& a(m_mat_a);
  MaterialVariableCellReal& b(m_mat_b);
  MaterialVariableCellReal& c(m_mat_c);
  MaterialVariableCellReal& d(m_mat_d);
  MaterialVariableCellReal& e(m_mat_e);

  ENUMERATE_ENVCELL(i,m_env1){
    Real z = (Real)i.index();
    b_ref[i] = z*2.3;
    c_ref[i] = z*3.1;
    d_ref[i] = z*2.5;
    e_ref[i] = z*4.2;
    a_ref[i] = 0;
    a[i] = 0;
    b[i] = b_ref[i];
    c[i] = c_ref[i];
    d[i] = d_ref[i];
    e[i] = e_ref[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_ENUMERATE(EnvCell, ...
 * avec en paramètres l'environnement et cherchant à accèder
 * aux variables multimat par l'envcell (i.e. le MatVarIndex en fait)
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest1(Integer nb_z)
{
  MaterialVariableCellReal& a_ref(m_mat_a_ref);
  MaterialVariableCellReal& b_ref(m_mat_b_ref);
  MaterialVariableCellReal& c_ref(m_mat_c_ref);
  MaterialVariableCellReal& d_ref(m_mat_d_ref);
  MaterialVariableCellReal& e_ref(m_mat_e_ref);

  // Ref CPU
  for (Integer z=0, iz=nb_z; z<iz; ++z) {
    ENUMERATE_ENVCELL(i,m_env1){
      a_ref[i] = b_ref[i] + c_ref[i] * d_ref[i] + e_ref[i];
    }
  }

  // GPU
  {
    auto queue = makeQueue(m_runner);
    auto cmd = makeCommand(queue);

    auto out_a = ax::viewOut(cmd, m_mat_a);
    auto in_b = ax::viewIn(cmd, m_mat_b);
    auto in_c = ax::viewIn(cmd, m_mat_c);
    auto in_d = ax::viewIn(cmd, m_mat_d);
    auto in_e = ax::viewIn(cmd, m_mat_e);  

    for (Integer z=0, iz=nb_z; z<iz; ++z) {
      cmd << RUNCOMMAND_ENUMERATE(EnvCell, evi, m_env1) {
        auto [mvi, cid] = evi();
        out_a[mvi] = in_b[mvi] + in_c[mvi] * in_d[mvi] + in_e[mvi];
      };
    }
  }

  // Test
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ENV(ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      ENUMERATE_ENVCELL(iev,env)
      {
        vc.areEqual(m_mat_a[iev], m_mat_a_ref[iev],"Test1_mat_a");
        vc.areEqual(m_mat_b[iev], m_mat_b_ref[iev],"Test1_mat_b");
        vc.areEqual(m_mat_c[iev], m_mat_c_ref[iev],"Test1_mat_c");
        vc.areEqual(m_mat_d[iev], m_mat_d_ref[iev],"Test1_mat_d");
        vc.areEqual(m_mat_e[iev], m_mat_e_ref[iev],"Test1_mat_e");
      }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_ENUMERATE(EnvCell, ...
 * avec en paramètres une collection d'envcell et cherchant à accèder
 * aux variables multimat par l'envcell (i.e. le MatVarIndex en fait)
 * mais aussi aux variables globales par le cell ID
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest2(Integer nb_z)
{
  MaterialVariableCellReal& a_ref(m_mat_a_ref);
  MaterialVariableCellReal& b_ref(m_mat_b_ref);
  MaterialVariableCellReal& c_ref(m_mat_c_ref);
  MaterialVariableCellReal& d_ref(m_mat_d_ref);
  MaterialVariableCellReal& e_ref(m_mat_e_ref);

  // Ref CPU
  for (Integer z=0, iz=nb_z; z<iz; ++z) {
    ENUMERATE_ENV(ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      EnvCellVectorView envcellsv = env->envView();
      ENUMERATE_ENVCELL(iev,envcellsv)
      {
        Cell cell = (*iev).globalCell();
        a_ref[iev] = b_ref[iev] * e_ref[cell];
      }
      ENUMERATE_ENVCELL(iev,envcellsv)
      {
        Cell cell = (*iev).globalCell();
        c_ref[iev] += a_ref[iev] / d_ref[cell];
      }
    }
  }

  // GPU
  {
    auto queue = makeQueue(m_runner);
    auto cmd = makeCommand(queue);

    auto inout_a = ax::viewInOut(cmd, m_mat_a);
    auto in_b = ax::viewIn(cmd, m_mat_b);
    auto out_c = ax::viewOut(cmd, m_mat_c);
    auto in_d = ax::viewIn(cmd, m_mat_d.globalVariable());
    auto in_e = ax::viewIn(cmd, m_mat_e.globalVariable());

    for (Integer z=0, iz=nb_z; z<iz; ++z) {
      ENUMERATE_ENV(ienv, m_mm_mng) {
        IMeshEnvironment* env = *ienv;
        EnvCellVectorView envcellsv = env->envView();    
        {
          cmd << RUNCOMMAND_ENUMERATE(EnvCell, evi, envcellsv) {
            auto [mvi, cid] = evi();
            inout_a[mvi] = in_b[mvi] * in_e[cid];
          };
        }
        {
          cmd << RUNCOMMAND_ENUMERATE(EnvCell, evi, envcellsv) {
            auto [mvi, cid] = evi();
            out_c[mvi] += inout_a[mvi] / in_d[cid];
          };
        }
      }
    }
  }

  // Test
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ENV(ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      ENUMERATE_ENVCELL(iev,env)
      {
        vc.areEqual(m_mat_a[iev], m_mat_a_ref[iev],"Test1_mat_a");
        vc.areEqual(m_mat_b[iev], m_mat_b_ref[iev],"Test1_mat_b");
        vc.areEqual(m_mat_c[iev], m_mat_c_ref[iev],"Test1_mat_c");
        vc.areEqual(m_mat_d[iev], m_mat_d_ref[iev],"Test1_mat_d");
        vc.areEqual(m_mat_e[iev], m_mat_e_ref[iev],"Test1_mat_e");
      }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_ENUMERATE(EnvCell, ...
 * Même chose que le test2 mais avec l'utilisation d'un pool
 * de run queue asynchrone
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest3(Integer nb_z)
{
  MaterialVariableCellReal& a_ref(m_mat_a_ref);
  MaterialVariableCellReal& b_ref(m_mat_b_ref);
  MaterialVariableCellReal& c_ref(m_mat_c_ref);
  MaterialVariableCellReal& d_ref(m_mat_d_ref);
  MaterialVariableCellReal& e_ref(m_mat_e_ref);

  // Ref CPU
  for (Integer z=0, iz=nb_z; z<iz; ++z) {
    ENUMERATE_ENV(ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      EnvCellVectorView envcellsv = env->envView();
      ENUMERATE_ENVCELL(iev,envcellsv)
      {
        Cell cell = (*iev).globalCell();
        a_ref[iev] = b_ref[iev] * e_ref[cell];
      }
      ENUMERATE_ENVCELL(iev,envcellsv)
      {
        Cell cell = (*iev).globalCell();
        c_ref[iev] += a_ref[iev] / d_ref[cell];
      }
    }
  }

  // GPU
  {
    auto async_queues = makeAsyncQueuePool(m_runner, m_mm_mng->environments().size());

    for (Integer z=0, iz=nb_z; z<iz; ++z) {
      ENUMERATE_ENV(ienv, m_mm_mng) {
        IMeshEnvironment* env = *ienv;
        EnvCellVectorView envcellsv = env->envView();

        auto cmd = makeCommand(async_queues[env->id()]);

        auto inout_a = ax::viewInOut(cmd, m_mat_a);
        auto in_b = ax::viewIn(cmd, m_mat_b);
        auto out_c = ax::viewOut(cmd, m_mat_c);
        auto in_d = ax::viewIn(cmd, m_mat_d.globalVariable());
        auto in_e = ax::viewIn(cmd, m_mat_e.globalVariable());

        {
          cmd << RUNCOMMAND_ENUMERATE(EnvCell, evi, envcellsv) {
            auto [mvi, cid] = evi();
            inout_a[mvi] = in_b[mvi] * in_e[cid];
          };
        }
        {
          cmd << RUNCOMMAND_ENUMERATE(EnvCell, evi, envcellsv) {
            auto [mvi, cid] = evi();
            out_c[mvi] += inout_a[mvi] / in_d[cid];
          };
        }
      }
      async_queues.waitAll();
    }
  }

  // Test
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ENV(ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      ENUMERATE_ENVCELL(iev,env)
      {
        vc.areEqual(m_mat_a[iev], m_mat_a_ref[iev],"Test1_mat_a");
        vc.areEqual(m_mat_b[iev], m_mat_b_ref[iev],"Test1_mat_b");
        vc.areEqual(m_mat_c[iev], m_mat_c_ref[iev],"Test1_mat_c");
        vc.areEqual(m_mat_d[iev], m_mat_d_ref[iev],"Test1_mat_d");
        vc.areEqual(m_mat_e[iev], m_mat_e_ref[iev],"Test1_mat_e");
      }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
