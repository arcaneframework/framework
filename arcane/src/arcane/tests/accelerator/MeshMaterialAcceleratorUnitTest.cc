// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialAcceleratorUnitTest.cc                          (C) 2000-2026 */
/*                                                                           */
/* Service de test unitaire du support accélérateurs des matériaux/milieux.  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshModifier.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/IMemoryAllocator.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/AllCellToAllEnvCellConverter.h"
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
#include "arcane/materials/MeshEnvironmentVariableRef.h"
#include "arcane/materials/EnvItemVector.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/internal/AllCellToAllEnvCellContainer.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/MaterialVariableViews.h"
#include "arcane/accelerator/RunCommandMaterialEnumerate.h"
#include "arcane/accelerator/AsyncRunQueuePool.h"
#include "arcane/accelerator/Reduce.h"

#include "arcane/accelerator/RunCommandEnumerate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Materials;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test unitaire du support accélérateurs des constituants.
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

  IMeshMaterialMng* m_mm_mng = nullptr;
  IMeshEnvironment* m_env1 = nullptr;

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

  EnvironmentVariableCellReal m_env_a;
  EnvironmentVariableCellReal m_env_b;
  EnvironmentVariableCellReal m_env_c;

  UniqueArray<Int32> m_env1_pure_value_index;
  UniqueArray<Int32> m_env1_partial_value_index;
  CellGroup m_sub_env_group1;

  void _initializeVariables(ComponentItemVectorView component);

 public:

  // Les méthodes suivantes doivent être publiques pour
  // sur accélérateur

  void _executeTest1(Integer nb_z, EnvCellVectorView env1);
  void _executeTest2(Integer nb_z);
  void _executeTest3(Integer nb_z);
  void _executeTest4(Integer nb_z, bool use_new_impl);
  void _executeTest5(Integer nb_z, MatCellVectorView mat);
  void _executeTest6();
  void _executeTest7(RunQueue& queue);
  void _testComponentSetSpecificExecutionPolicy();
  void _checkEnvValues1();
  void _checkMatValues1();
  void _checkEnvironmentValues();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(MeshMaterialAcceleratorUnitTest,
                                           IUnitTest, MeshMaterialAcceleratorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialAcceleratorUnitTest::
MeshMaterialAcceleratorUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_mm_mng(nullptr)
, m_env1(nullptr)
, m_mat_a_ref(VariableBuildInfo(mesh(), "MatA_ref"))
, m_mat_b_ref(VariableBuildInfo(mesh(), "MatB_ref"))
, m_mat_c_ref(VariableBuildInfo(mesh(), "MatC_ref"))
, m_mat_d_ref(VariableBuildInfo(mesh(), "MatD_ref"))
, m_mat_e_ref(VariableBuildInfo(mesh(), "MatE_ref"))
, m_mat_a(VariableBuildInfo(mesh(), "MatA"))
, m_mat_b(VariableBuildInfo(mesh(), "MatB"))
, m_mat_c(VariableBuildInfo(mesh(), "MatC"))
, m_mat_d(VariableBuildInfo(mesh(), "MatD"))
, m_mat_e(VariableBuildInfo(mesh(), "MatE"))
, m_env_a(VariableBuildInfo(mesh(), "EnvA"))
, m_env_b(VariableBuildInfo(mesh(), "EnvB"))
, m_env_c(VariableBuildInfo(mesh(), "EnvC"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialAcceleratorUnitTest::
~MeshMaterialAcceleratorUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
initializeTest()
{
  m_runner = subDomain()->acceleratorMng()->runner();
  m_mm_mng = IMeshMaterialMng::getReference(mesh());

  // Lit les infos des matériaux du JDD et les enregistre dans le gestionnaire
  UniqueArray<String> mat_names = { "MAT1", "MAT2", "MAT3", "MAT4" };
  for (String v : mat_names) {
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
  {
    Materials::MeshEnvironmentBuildInfo env_build("ENV3");
    env_build.addMaterial("MAT1");
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
    Integer nb_cell = allCells().size();
    Int64 total_nb_cell = nb_cell;
    ENUMERATE_CELL (icell, allCells()) {
      if (icell.itemLocalId() != 0) { // on ne veut pas de la première maille pour tester un cas tordu en //
        Cell cell = *icell;
        Int64 cell_index = cell.uniqueId();
        if (cell_index < ((2 * total_nb_cell) / 3)) {
          env1_indexes.add(icell.itemLocalId());
        }
        if (cell_index < (total_nb_cell / 2) && cell_index > (total_nb_cell / 3)) {
          mat2_indexes.add(icell.itemLocalId());
        }
        if ((cell_index % 2) == 0)
          sub_group_indexes.add(icell.itemLocalId());
      }
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
      for (Integer z = 0; z < nb_cell; ++z) {
        bool add_to_mat1 = (z < (nb_cell / 2) && z > (nb_cell / 4));
        bool add_to_mat2 = (z >= (nb_cell / 2) || z < (nb_cell / 3));
        if (add_to_mat1) {
          mat1_indexes.add(env1_indexes[z]);
        }
        if (add_to_mat2)
          mat2_indexes.add(env1_indexes[z]);
      }
      // Ajoute les mailles du matériau 1
      modifier.addCells(mat1, mat1_indexes);
      Integer nb_mat = env->nbMaterial();
      if (nb_mat > 1)
        // Ajoute les mailles du matériau 2
        modifier.addCells(m_mm_mng->environments()[0]->materials()[1], mat2_indexes);
    }
    // Ajoute les mailles du milieu 2
    if (mat2) {
      Materials::MeshMaterialModifier modifier(m_mm_mng);
      //modifier.addCells(m_mat2->environment(),mat2_indexes);
      modifier.addCells(mat2, mat2_indexes);
    }
  }

  for (IMeshEnvironment* env : m_mm_mng->environments()) {
    info() << "** ** ENV name=" << env->name() << " nb_item=" << env->view().nbItem();
    Integer nb_pure_env = 0;
    ENUMERATE_ENVCELL (ienvcell, env) {
      if ((*ienvcell).allEnvCell().nbEnvironment() == 1)
        ++nb_pure_env;
    }
    info() << "** ** NB_PURE=" << nb_pure_env;
  }

  // Créé un groupe contenant un sous-ensemble des mailles pour test EnvCellVector
  {
    UniqueArray<Int32> sub_indexes;
    ENUMERATE_ (Cell, icell, allCells()) {
      CellLocalId c = *icell;
      if ((c.localId() % 3) == 0)
        sub_indexes.add(c);
    }
    m_sub_env_group1 = mesh()->cellFamily()->createGroup("SubGroup1", sub_indexes);
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
    ENUMERATE_ENVCELL (ienvcell, m_env1) {
      EnvCell env_cell = *ienvcell;
      MatVarIndex mvi = env_cell._varIndex();
      Integer nb_env = env_cell.allEnvCell().nbEnvironment();
      if (nb_env == 1)
        m_env1_pure_value_index.add(mvi.valueIndex());
      else if (nb_env > 1)
        m_env1_partial_value_index.add(mvi.valueIndex());
      else
        ++nb_unknown;
    }
  }

  Integer nb_z = 200;
  if (arcaneIsDebug())
    nb_z /= 100;
  Integer nb_z2 = nb_z / 5;

  Int32 env_idx = m_env1->_internalApi()->variableIndexerIndex() + 1;
  info() << "ENV_IDX=" << env_idx
         << " nb_pure=" << m_env1_pure_value_index.size()
         << " nb_partial=" << m_env1_partial_value_index.size()
         << " nb_unknown=" << nb_unknown
         << " nb_z=" << nb_z << " nb_z2=" << nb_z2;

  _initializeVariables(m_env1->envView());
  EnvCellVector sub_ev1(m_sub_env_group1, m_env1);
  {
    _executeTest1(nb_z, m_env1->envView());
    _executeTest1(nb_z, sub_ev1);
  }
  _testComponentSetSpecificExecutionPolicy();
  {
    _executeTest2(nb_z);
    _executeTest3(nb_z);
    _executeTest4(nb_z, false);
    _executeTest4(nb_z, true);
  }
  {
    IMeshEnvironment* env2 = m_mm_mng->environments()[1];
    IMeshMaterial* mat2 = env2->materials()[1];
    _executeTest5(nb_z, mat2->matView());
  }
  {
    _executeTest6();
  }
  {
    RunQueue queue = makeQueue(m_runner);
    if (!queue.isAcceleratorPolicy()) {
      // Le mode concurrent n'est pas supporté avec les accélérateurs
      // (uniquement le multi-threading ou le séquentiel)
      queue.setConcurrentCommandCreation(true);
      if (!queue.isConcurrentCommandCreation())
        ARCANE_FATAL("Can not create concurrent commands");
      // Teste l'exécution multhread de la création de MatCellVector/EnvCellVector
      ParallelLoopOptions loop_options;
      loop_options.setGrainSize(1);
      arcaneParallelFor(1, 20, loop_options,
                        [&](Integer a, Integer n) {
                          for (Int32 i = a; i < (a + n); ++i)
                            _executeTest7(queue);
                        });
    }
    _executeTest7(queue);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
_initializeVariables(ComponentItemVectorView component)
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
  bool is_env = component.component()->isEnvironment();

  ENUMERATE_COMPONENTCELL (i, component) {
    Real z = (Real)i.index();
    b_ref[i] = z * 2.3;
    c_ref[i] = z * 3.1;
    d_ref[i] = z * 2.5;
    e_ref[i] = z * 4.2;
    a_ref[i] = 0;
    a[i] = 0;
    b[i] = b_ref[i];
    c[i] = c_ref[i];
    d[i] = d_ref[i];
    e[i] = e_ref[i];
    if (is_env) {
      m_env_a[i] = a_ref[i];
      m_env_b[i] = b_ref[i];
      m_env_c[i] = c_ref[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_MAT_ENUMERATE(EnvCell, ...
 * avec en paramètres l'environnement et cherchant à accèder
 * aux variables multimat par l'envcell (i.e. le MatVarIndex en fait)
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest1(Integer nb_z, EnvCellVectorView env1)
{
  _initializeVariables(m_env1->envView());

  // Ref CPU
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    ENUMERATE_ENVCELL (i, env1) {
      m_mat_a_ref[i] = m_mat_b_ref[i] + m_mat_c_ref[i] * m_mat_d_ref[i] + m_mat_e_ref[i];
    }
  }

  // GPU
  {
    auto queue = makeQueue(m_runner);

    for (Integer z = 0, iz = nb_z; z < iz; ++z) {
      auto cmd = makeCommand(queue);

      auto out_a = viewOut(cmd, m_mat_a);
      auto in_b = viewIn(cmd, m_mat_b);
      auto in_c = viewIn(cmd, m_mat_c);
      auto in_d = viewIn(cmd, m_mat_d);
      auto in_e = viewIn(cmd, m_mat_e);
      cmd << RUNCOMMAND_MAT_ENUMERATE(EnvCell, evi, env1)
      {
        out_a[evi] = in_b[evi] + in_c[evi] * in_d[evi] + in_e[evi];
      };
      out_a = {};
      in_b = {};
    }
  }

  _checkEnvValues1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_MAT_ENUMERATE(MatCell, ...
 * avec en paramètres le matériau et cherchant à accèder
 * aux variables multimat par la matcell (i.e. le MatVarIndex en fait)
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest5(Integer nb_z, MatCellVectorView mat1)
{
  info() << "Execute Test 5";
  _initializeVariables(mat1);

  // Ref CPU
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    ENUMERATE_MATCELL (i, mat1) {
      m_mat_a_ref[i] = m_mat_b_ref[i] + m_mat_c_ref[i] * m_mat_d_ref[i] + m_mat_e_ref[i];
    }
  }

  // GPU
  auto queue = makeQueue(m_runner);
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    auto cmd = makeCommand(queue);

    auto out_a = ax::viewOut(cmd, m_mat_a);
    auto in_b = ax::viewIn(cmd, m_mat_b);
    auto in_c = ax::viewIn(cmd, m_mat_c);
    auto in_d = ax::viewIn(cmd, m_mat_d);
    auto in_e = ax::viewIn(cmd, m_mat_e);
    cmd << RUNCOMMAND_MAT_ENUMERATE(MatCell, evi, mat1)
    {
      out_a[evi] = in_b[evi] + in_c[evi] * in_d[evi] + in_e[evi];
    };
  }

  _checkMatValues1();

  // Test récupération des MatCell
  {
    auto cmd = makeCommand(queue);
    auto in_b = ax::viewIn(cmd, m_mat_b);
    auto in_c = ax::viewIn(cmd, m_mat_c);
    auto out_a = ax::viewOut(cmd, m_mat_a);
    cmd << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, vi, mat1)
    {
      auto [mvi, cid] = vi();
      MatCell mc = mat1.matCell(vi.index());
      Real x = in_b[mvi] + in_c[mvi];
      Real y = in_b[mc] + in_c[mc];
      out_a[mc] = x + y;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_MAT_ENUMERATE(EnvCell, ...
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
  UniqueArray<Real> ref_reduced_values(m_mm_mng->environments().size());
  // Ref CPU
  CellToAllEnvCellConverter allenvcell_converter(m_mm_mng);
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    ENUMERATE_ENV (ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      EnvCellVectorView envcellsv = env->envView();
      ENUMERATE_ENVCELL (iev, envcellsv) {
        Cell cell = (*iev).globalCell();
        a_ref[iev] = b_ref[iev] * e_ref[cell];
        AllEnvCell all_env_cell = allenvcell_converter[cell];
        ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
          EnvCell env_cell = *ienvcell;
          Int32 env_id = env_cell.environmentId();
          a_ref[iev] += env_id;
        }
      }
      Real total_cref = 0.0;
      ENUMERATE_ENVCELL (iev, envcellsv) {
        Cell cell = (*iev).globalCell();
        Real value_to_add = a_ref[iev] * d_ref[cell];
        c_ref[iev] += value_to_add;
        total_cref += value_to_add;
      }
      if (z == 0) {
        ref_reduced_values[env->id()] = total_cref;
        info() << "REF_REDUCED_VALUE=" << total_cref << " env=" << env->name();
      }
    }
  }

  // GPU
  {
    auto queue = makeQueue(m_runner);

    for (Integer z = 0, iz = nb_z; z < iz; ++z) {

      ENUMERATE_ENV (ienv, m_mm_mng) {
        IMeshEnvironment* env = *ienv;
        EnvCellVectorView envcellsv = env->envView();
        {
          auto cmd = makeCommand(queue);

          auto inout_a = viewInOut(cmd, m_mat_a);
          auto in_b = viewIn(cmd, m_mat_b);
          auto in_e = viewIn(cmd, m_mat_e.globalVariable());

          auto inout_env_a = viewInOut(cmd, m_env_a);
          auto in_env_b = viewIn(cmd, m_env_b);
          cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv)
          {
            auto [mvi, cid] = evi();
            inout_a[mvi] = in_b[mvi] * in_e[cid];
            inout_env_a[mvi] = in_env_b[mvi] * in_e[cid];
            AllEnvCell all_env_cell = allenvcell_converter[cid];
            ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
              EnvCell env_cell = *ienvcell;
              Int32 env_id = env_cell.environmentId();
              inout_a[mvi] += env_id;
              inout_env_a[mvi] += env_id;
            }
          };
        }
        {
          auto cmd = makeCommand(queue);
          auto inout_env_a = viewInOut(cmd, m_env_a);
          auto out_c = viewOut(cmd, m_mat_c);
          auto in_d = viewIn(cmd, m_mat_d.globalVariable());
          auto out_env_c = viewOut(cmd, m_env_c);
          ax::ReducerSum2<Real> reducer2(cmd);
          cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv, reducer2)
          {
            auto [mvi, cid] = evi();
            Real value_to_add = inout_env_a[mvi] * in_d[cid];
            EnvCell ev0(envcellsv.envCell(evi.index()));
            out_c[mvi] += value_to_add;
            out_env_c[ev0] += inout_env_a[mvi] * in_d[cid];
            reducer2.combine(value_to_add);
          };
          Real reduced_value = reducer2.reducedValue();
          if (z == 0) {
            Real ref_value = ref_reduced_values[env->id()];
            info() << "REDUCED_VALUE=" << reduced_value << " ref=" << ref_value << " env=" << env->name();
            if (!math::isNearlyEqualWithEpsilon(reduced_value, ref_value, 1.0e-14))
              ARCANE_FATAL("Bad value v={0} expected={1}", reduced_value, ref_value);
          }
        }
      }
    }
  }

  _checkEnvValues1();
  _checkEnvironmentValues();
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
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    ENUMERATE_ENV (ienv, m_mm_mng) {
      IMeshEnvironment* env = *ienv;
      EnvCellVectorView envcellsv = env->envView();
      ENUMERATE_ENVCELL (iev, envcellsv) {
        Cell cell = (*iev).globalCell();
        a_ref[iev] = b_ref[iev] * e_ref[cell];
      }
      ENUMERATE_ENVCELL (iev, envcellsv) {
        Cell cell = (*iev).globalCell();
        c_ref[iev] += a_ref[iev] * d_ref[cell];
      }
    }
  }

  // GPU
  {
    auto async_queues = makeAsyncQueuePool(m_runner, m_mm_mng->environments().size());

    for (Integer z = 0, iz = nb_z; z < iz; ++z) {
      ENUMERATE_ENV (ienv, m_mm_mng) {
        IMeshEnvironment* env = *ienv;
        EnvCellVectorView envcellsv = env->envView();

        {
          auto cmd = makeCommand(async_queues[env->id()]);
          auto inout_a = viewInOut(cmd, m_mat_a);
          auto in_b = viewIn(cmd, m_mat_b);
          auto in_e = viewIn(cmd, m_mat_e.globalVariable());

          cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv)
          {
            auto [mvi, cid] = evi();
            EnvCell ev = envcellsv.envCell(evi.index());
            inout_a[mvi] = in_b[ev] * in_e[cid];
          };
        }
        {
          auto cmd = makeCommand(async_queues[env->id()]);
          auto inout_a = viewInOut(cmd, m_mat_a);
          auto out_c = viewOut(cmd, m_mat_c);
          auto in_d = viewIn(cmd, m_mat_d.globalVariable());

          cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv)
          {
            auto [mvi, cid] = evi();
            out_c[mvi] += inout_a[mvi] * in_d[cid];
          };
        }
      }
      async_queues.waitAll();
    }
  }

  _checkEnvValues1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(...
 * Test qui boucle sur les mailles puis sur les env/mal cells
 * de chaque maille.
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest4(Integer nb_z, bool use_new_impl)
{
  MaterialVariableCellReal& a_ref(m_mat_a_ref);
  MaterialVariableCellReal& b_ref(m_mat_b_ref);
  MaterialVariableCellReal& c_ref(m_mat_c_ref);

  // Ref CPU
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    CellToAllEnvCellConverter allenvcell_converter(m_mm_mng);
    ENUMERATE_CELL (icell, allCells()) {
      Cell cell = *icell;
      AllEnvCell all_env_cell = allenvcell_converter[cell];

      Real sum2 = 0.;
      ENUMERATE_CELL_ENVCELL (iev, all_env_cell) {
        sum2 += b_ref[iev] + b_ref[icell];
      }

      Real sum3 = 0.;
      if (all_env_cell.nbEnvironment() > 1) {
        ENUMERATE_CELL_ENVCELL (iev, all_env_cell) {
          Real contrib2 = (b_ref[iev] + b_ref[icell]) - (sum2 + 1.);
          c_ref[iev] = contrib2 * c_ref[icell];
          sum3 += contrib2;
        }
      }
      a_ref[icell] = sum3;
    }
  }

  // GPU
  {
    auto queue = makeQueue(m_runner);

    if (use_new_impl) {
      for (Integer z = 0, iz = nb_z; z < iz; ++z) {
        //![SampleAllEnvCell]
        Arcane::Materials::AllEnvCellVectorView all_env_view = m_mm_mng->view(allCells());
        auto cmd = makeCommand(queue);
        auto in_b = viewIn(cmd, m_mat_b);
        auto out_c = viewOut(cmd, m_mat_c);
        auto in_c_g = viewIn(cmd, m_mat_c.globalVariable());
        auto out_a_g = viewOut(cmd, m_mat_a);
        cmd << RUNCOMMAND_MAT_ENUMERATE(AllEnvCell, all_env_cell_iter, all_env_view)
        {
          Arcane::Materials::AllEnvCell all_env_cell = all_env_cell_iter;
          Arcane::CellLocalId cid = all_env_cell.globalCellId();
          Real sum2 = 0.0;
          for (Arcane::Materials::EnvCell ev : all_env_cell.subEnvItems()) {
            sum2 += in_b[ev] + in_b[cid];
          }

          Real sum3 = 0.0;
          if (all_env_cell.nbEnvironment() > 1) {
            for (Arcane::Materials::EnvCell ev : all_env_cell.subEnvItems()) {
              Real contrib2 = (in_b[ev] + in_b[all_env_cell]) - (sum2 + 1.);
              out_c[ev] = contrib2 * in_c_g[cid];
              sum3 += contrib2;
            }
          }
          out_a_g[cid] = sum3;
        };
        //![SampleAllEnvCell]
      }
    }
    else {
      m_mm_mng->enableCellToAllEnvCellForRunCommand(true, true);
      CellToAllEnvCellAccessor cell2allenvcell(m_mm_mng);

      for (Integer z = 0, iz = nb_z; z < iz; ++z) {
        auto cmd = makeCommand(queue);
        auto in_b = viewIn(cmd, m_mat_b);
        auto out_c = viewOut(cmd, m_mat_c);
        auto in_c_g = viewIn(cmd, m_mat_c.globalVariable());
        auto out_a_g = viewOut(cmd, m_mat_a);
        cmd << RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(cell2allenvcell, cid, allCells())
        {

          Real sum2 = 0.;
          ENUMERATE_CELL_ALLENVCELL(iev, cid, cell2allenvcell)
          {
            sum2 += in_b[*iev] + in_b[cid];
          }

          Real sum3 = 0.;
          if (cell2allenvcell.nbEnvironment(cid) > 1) {
            ENUMERATE_CELL_ALLENVCELL(iev, cid, cell2allenvcell)
            {
              Real contrib2 = (in_b[*iev] + in_b[cid]) - (sum2 + 1.);
              out_c[*iev] = contrib2 * in_c_g[cid];
              sum3 += contrib2;
            }
          }
          out_a_g[cid] = sum3;
        };
      }
    }
  }

  _checkEnvValues1();

  // Some further functions testing, not really usefull here, but it improves cover
  {
    AllCellToAllEnvCellContainer useless(m_mm_mng);
    useless.initialize();
  }

  // Call to forceRecompute to test bruteForceUpdate
  m_mm_mng->forceRecompute();

  // Remove one cell to test other branch of bruteForceUpdate
  Int32UniqueArray lid(1);
  lid[0] = 1;
  mesh()->modifier()->removeCells(lid);
  mesh()->modifier()->endUpdate();
  m_mm_mng->forceRecompute();

  // Force last path of bruteForceUpdate testing
  Int32UniqueArray env3_indexes;
  ENUMERATE_CELL (icell, allCells()) {
    env3_indexes.add(icell.itemLocalId());
  }

  IMeshEnvironment* env3 = m_mm_mng->environments()[2];
  {
    Materials::MeshMaterialModifier modifier(m_mm_mng);
    modifier.addCells(env3->materials()[0], env3_indexes);
  }
  m_mm_mng->forceRecompute();
  ENUMERATE_ENVCELL (i, env3) {
    Real z = (Real)i.index();
    m_mat_a_ref[i] = z * 3.6;
    m_mat_b_ref[i] = z * 1.8;
    m_mat_c_ref[i] = z * 1.1;
    m_mat_a[i] = m_mat_a_ref[i];
    m_mat_b[i] = m_mat_b_ref[i];
    m_mat_c[i] = m_mat_c_ref[i];
  }

  // Another round to test numerical pbs
  // Ref CPU
  for (Integer z = 0, iz = nb_z; z < iz; ++z) {
    CellToAllEnvCellConverter allenvcell_converter(m_mm_mng);
    ENUMERATE_CELL (icell, allCells()) {
      Cell cell = *icell;
      AllEnvCell all_env_cell = allenvcell_converter[cell];

      Real sum2 = 0.;
      ENUMERATE_CELL_ENVCELL (iev, all_env_cell) {
        sum2 += b_ref[iev] + b_ref[icell];
      }

      Real sum3 = 0.;
      if (all_env_cell.nbEnvironment() > 1) {
        ENUMERATE_CELL_ENVCELL (iev, all_env_cell) {
          Real contrib2 = (b_ref[iev] + b_ref[icell]) - (sum2 + 1.);
          c_ref[iev] = contrib2 * c_ref[icell];
          sum3 += contrib2;
        }
      }
      a_ref[icell] = sum3;
    }
  }

  // GPU
  {
    auto queue = makeQueue(m_runner);

    m_mm_mng->enableCellToAllEnvCellForRunCommand(true, true);
    CellToAllEnvCellAccessor cell2allenvcell(m_mm_mng);

    for (Integer z = 0, iz = nb_z; z < iz; ++z) {
      auto cmd = makeCommand(queue);

      auto in_b = ax::viewIn(cmd, m_mat_b);
      auto out_c = ax::viewOut(cmd, m_mat_c);
      auto in_c_g = ax::viewIn(cmd, m_mat_c.globalVariable());
      auto out_a_g = ax::viewOut(cmd, m_mat_a);
      cmd << RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(cell2allenvcell, cid, allCells())
      {

        Real sum2 = 0.;
        ENUMERATE_CELL_ALLENVCELL(iev, cid, cell2allenvcell)
        {
          sum2 += in_b[*iev] + in_b[cid];
        }

        Real sum3 = 0.;
        if (cell2allenvcell.nbEnvironment(cid) > 1) {
          ENUMERATE_CELL_ALLENVCELL(iev, cid, cell2allenvcell)
          {
            Real contrib2 = (in_b[*iev] + in_b[cid]) - (sum2 + 1.);
            out_c[*iev] = contrib2 * in_c_g[cid];
            sum3 += contrib2;
          }
        }
        out_a_g[cid] = sum3;
      };
    }
  }

  _checkEnvValues1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test du RUNCOMMAND_MAT_ENUMERATE(EnvCell, ...
 * avec en paramètres l'environnement et cherchant à accèder
 * aux variables multimat par l'envcell (i.e. le MatVarIndex en fait)
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest6()
{
  //![SampleEnvAndGlobalCell]
  Arcane::Materials::IMeshEnvironment* env1 = m_env1;
  Int32 nb_cell = env1->cells().size();
  Arcane::NumArray<Int32, MDDim1> cells_local_id(nb_cell);
  Arcane::Materials::MaterialVariableCellReal& mat_a = m_mat_a;

  {
    auto queue = makeQueue(m_runner);
    auto command = makeCommand(queue);
    auto cells_local_id_view = viewOut(command, cells_local_id);
    auto out_mat_a = viewOut(command, mat_a);
    command << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, iter, m_env1)
    {
      EnvAndGlobalCellIteratorValue evi = iter; // Valeur de l'itérateur
      auto [iter_mvi, iter_cid] = evi();
      EnvCellLocalId mvi = iter_mvi; // Numéro local de la maille milieu
      Arcane::CellLocalId cid = iter_cid; // Numéro de la maille globale de la maille milieu courante
      Int32 iter_index = evi.index(); // Index de l'itération
      cells_local_id_view[iter_index] = cid;
      out_mat_a[mvi] = 1.2;
    };
  }
  //![SampleEnvAndGlobalCell]
  {
    ENUMERATE_ENVCELL (ienvcell, m_env1) {
      EnvCell env_cell(*ienvcell);
      Int32 index = ienvcell.index();
      Int32 computed_lid = cells_local_id[index];
      Int32 ref_lid = env_cell.globalCell().localId();
      if (computed_lid != ref_lid)
        ARCANE_FATAL("Bad computed localId() ref={0} computed={1} index={2}",
                     ref_lid, computed_lid, index);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tests passages CellVector vers EnvCellVectorView ou MatCellVectorView
 */
void MeshMaterialAcceleratorUnitTest::
_executeTest7(RunQueue& queue)
{
  ValueChecker vc(A_FUNCINFO);

  // Créé un CellVector contenant une maille sur 2
  IItemFamily* cell_family = mesh()->cellFamily();
  const Int32 nb_cell_to_add = cell_family->maxLocalId() / 2;
  CellVector cell_vector1;
  {
    UniqueArray<Int32> ids_to_use(MemoryUtils::getDeviceOrHostAllocator());
    auto cmd = makeCommand(queue);
    ids_to_use.resize(nb_cell_to_add);
    auto out_c = ids_to_use.smallSpan();
    // Prend une maille sur 2 pour le vecteur.
    cmd << RUNCOMMAND_LOOP1(iter, nb_cell_to_add)
    {
      auto [i] = iter();
      out_c[i] = i * 2;
    };
    CellVector cell_vector2(cell_family, ids_to_use);
    cell_vector1 = cell_vector2;
  }
  Int32 nb_cell_in_vector = cell_vector1.size();
  vc.areEqual(nb_cell_in_vector, nb_cell_to_add, "CellInVector");

  {
    EnvCellVector env_vector(cell_vector1.view(), m_env1);
    EnvCellVectorView sub_env_view(env_vector.view());
    Int32 nb_sub_item = sub_env_view.nbItem();
    info() << "NB_SUB_ITEM (env)=" << nb_sub_item;
    // TODO: Vérifier la validité

    ENUMERATE_ENVCELL (ienvcell, sub_env_view) {
      EnvCell ec = *ienvcell;
      if ((ec.globalCellId() % 2) != 0)
        ARCANE_FATAL("Bad env cell lid={0}", ec.globalCellId());
    }
  }
  {
    MatCellVector mat_vector(cell_vector1.view(), m_env1->materials()[1]);
    MatCellVectorView sub_mat_view(mat_vector.view());
    Int32 nb_sub_item = sub_mat_view.nbItem();
    info() << "NB_SUB_ITEM (mat)=" << nb_sub_item;
    ENUMERATE_MATCELL (imatcell, sub_mat_view) {
      MatCell mc = *imatcell;
      if ((mc.globalCellId() % 2) != 0)
        ARCANE_FATAL("Bad mat cell lid={0}", mc.globalCellId());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
_testComponentSetSpecificExecutionPolicy()
{
  ValueChecker vc(A_FUNCINFO);
  // Vérifie le changement politique d'exécution d'un IMeshComponent
  m_env1->setSpecificExecutionPolicy(Accelerator::eExecutionPolicy::Sequential);
  EnvCellVector sub_env1(m_sub_env_group1, m_env1);
  m_env1->setSpecificExecutionPolicy(Accelerator::eExecutionPolicy::None);
  EnvCellVector sub_env2(m_sub_env_group1, m_env1);
  Int32 nb_item1 = sub_env1.view().nbItem();
  Int32 nb_item2 = sub_env2.view().nbItem();
  vc.areEqual(nb_item1, nb_item2, "NbItem");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  void _checkOneValue(Real value, Real ref_value, const char* var_name)
  {
    Real epsilon = 1.0e-15;
    if (!math::isNearlyEqualWithEpsilon(value, ref_value, epsilon)) {
      Real diff = ref_value - value;
      if (ref_value != 0.0)
        diff /= ref_value;
      ARCANE_FATAL("Bad value for '{0}' : ref={1} v={2} diff={3}",
                   var_name, ref_value, value, diff);
    }
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
_checkEnvValues1()
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ENV (ienv, m_mm_mng) {
    IMeshEnvironment* env = *ienv;
    ENUMERATE_ENVCELL (iev, env) {
      _checkOneValue(m_mat_a[iev], m_mat_a_ref[iev], "Test1_mat_a");
      _checkOneValue(m_mat_b[iev], m_mat_b_ref[iev], "Test1_mat_b");
      _checkOneValue(m_mat_c[iev], m_mat_c_ref[iev], "Test1_mat_c");
      _checkOneValue(m_mat_d[iev], m_mat_d_ref[iev], "Test1_mat_d");
      _checkOneValue(m_mat_e[iev], m_mat_e_ref[iev], "Test1_mat_e");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
_checkMatValues1()
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_MAT (imat, m_mm_mng) {
    IMeshMaterial* mat = *imat;
    ENUMERATE_MATCELL (imat, mat) {
      _checkOneValue(m_mat_a[imat], m_mat_a_ref[imat], "Test1_mat_a");
      _checkOneValue(m_mat_b[imat], m_mat_b_ref[imat], "Test1_mat_b");
      _checkOneValue(m_mat_c[imat], m_mat_c_ref[imat], "Test1_mat_c");
      _checkOneValue(m_mat_d[imat], m_mat_d_ref[imat], "Test1_mat_d");
      _checkOneValue(m_mat_e[imat], m_mat_e_ref[imat], "Test1_mat_e");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialAcceleratorUnitTest::
_checkEnvironmentValues()
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ENV (ienv, m_mm_mng) {
    IMeshEnvironment* env = *ienv;
    ENUMERATE_ENVCELL (iev, env) {
      _checkOneValue(m_env_a[iev], m_mat_a_ref[iev], "Test1_env_a");
      _checkOneValue(m_env_b[iev], m_mat_b_ref[iev], "Test1_env_b");
      _checkOneValue(m_env_c[iev], m_mat_c_ref[iev], "Test1_env_c");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
