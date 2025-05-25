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

MeshMaterialTesterModule::
MeshMaterialTesterModule(const ModuleBuildInfo& mbi)
: ArcaneMeshMaterialTesterObject(mbi)
, m_material_mng(IMeshMaterialMng::getReference(mbi.meshHandle()))
, m_density(VariableBuildInfo(this, "Density"))
, m_pressure(VariableBuildInfo(this, "Pressure"))
, m_mat_density2(VariableBuildInfo(this, "Density", IVariable::PNoDump))
, m_mat_nodump_real(VariableBuildInfo(this, "NoDumpReal", IVariable::PNoDump))
, m_present_material(VariableBuildInfo(this, "PresentMaterial"))
, m_mat_int32(VariableBuildInfo(this, "PresentMaterial"))
, m_mat_not_used_real(VariableBuildInfo(this, "NotUsedRealVariable"))
, m_nb_starting_cell(VariableBuildInfo(this, "NbStartingCell"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialTesterModule::
~MeshMaterialTesterModule()
{
  for (VariableCellReal* v : m_density_post_processing)
    delete v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("MeshMaterialTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshMaterialTester.buildInit"));
    time_loop->setEntryPoints(ITimeLoop::WBuild, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshMaterialTester.startInit"));
    clist.add(TimeLoopEntryPointInfo("MeshMaterialTester.continueInit"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshMaterialTester.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("MeshMaterialTester");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
buildInit()
{
  // La création des milieux et des matériaux doit se faire dans un point
  // d'entrée de type 'build' pour que la liste des variables créées par les
  // milieux et les matériaux soient accessibles dans le post-traitement.
  info() << "MESH_MATERIAL_TESTER :: buildInit()";

  Materials::IMeshMaterialMng* mm = IMeshMaterialMng::getReference(defaultMesh());

  m_material_mng->setModificationFlags(options()->modificationFlags());

  m_material_mng->setMeshModificationNotified(true);

  // En parallèle, test la création de variables milieux aussi sur les matériaux
  if (parallelMng()->isParallel())
    m_material_mng->setAllocateScalarEnvironmentVariableAsMaterial(true);

  if (subDomain()->isContinue() && options()->recreateFromDump()) {
    mm->recreateFromDump();
  }
  else {
    UniqueArray<MeshMaterialInfo*> materials_info;
    // Lit les infos des matériaux du JDD et les enregistre dans le gestionnaire
    for (Integer i = 0, n = options()->material().size(); i < n; ++i) {
      String mat_name = options()->material[i].name;
      info() << "Found material name=" << mat_name;
      materials_info.add(mm->registerMaterialInfo(mat_name));
    }

    MeshBlockBuildInfo mbbi("BLOCK1", allCells());
    UniqueArray<IMeshEnvironment*> saved_envs;

    // Créé les milieux
    for (Integer i = 0, n = options()->environment().size(); i < n; ++i) {
      String env_name = options()->environment[i].name;
      info() << "Found environment name=" << env_name;
      Materials::MeshEnvironmentBuildInfo env_build(env_name);
      for (Integer k = 0, kn = options()->environment[i].material.size(); k < kn; ++k) {
        String mat_name = options()->environment[i].material[k];
        info() << "Add material " << mat_name << " for environment " << env_name;
        env_build.addMaterial(mat_name);
      }
      IMeshEnvironment* env = mm->createEnvironment(env_build);
      saved_envs.add(env);
      // Le bloc ne contient que 2 milieux
      if (i < 2) {
        info() << "Add environment " << env_name << " to block1";
        mbbi.addEnvironment(env);
      }
    }

    // Création du bloc BLOCK1 sur le groupe de toutes les mailles
    // et contenant les milieux ENV1 et ENV2
    m_block1 = mm->createBlock(mbbi);

    {
      // Création d'un deuxième bloc de manière incrémentalle.
      Integer nb_env = saved_envs.size();
      if (nb_env >= 2) {
        MeshBlockBuildInfo mbbi2("BLOCK2", allCells());
        mbbi2.addEnvironment(saved_envs[0]);
        IMeshBlock* block2 = mm->createBlock(mbbi2);
        Integer nb_env1 = block2->nbEnvironment();
        mm->addEnvironmentToBlock(block2, saved_envs[1]);
        info() << "Finished incremental creation of block";
        Integer nb_env2 = block2->nbEnvironment();
        if (nb_env2 != (nb_env1 + 1))
          ARCANE_FATAL("Bad number of environment");
        if (block2->environments()[nb_env1] != saved_envs[1])
          ARCANE_FATAL("Bad last environment");
        // Supprime le premier milieu du bloc
        IMeshEnvironment* first_env = block2->environments()[0];
        IMeshEnvironment* second_env = block2->environments()[1];
        mm->removeEnvironmentToBlock(block2, first_env);
        nb_env2 = block2->nbEnvironment();
        if (nb_env2 != nb_env1)
          ARCANE_FATAL("Bad number of environment after remove");
        if (block2->environments()[0] != second_env)
          ARCANE_FATAL("Bad first environment");
      }
    }

    mm->endCreate(subDomain()->isContinue());

    info() << "List of materials:";
    for (MeshMaterialInfo* m : materials_info) {
      info() << "MAT=" << m->name();
      for (String s : m->environmentsName())
        info() << " In ENV=" << s;
    }
  }

  // Récupère deux matériaux de deux milieux différents pour test.
  ConstArrayView<Materials::IMeshEnvironment*> envs = mm->environments();
  Integer nb_env = envs.size();
  m_mat1 = mm->environments()[0]->materials()[0];
  if (nb_env > 1)
    m_mat2 = mm->environments()[1]->materials()[0];

  m_global_deltat.assign(1.0);

  for (Integer i = 0, n = m_material_mng->materials().size(); i < n; ++i) {
    IMeshMaterial* mat = m_material_mng->materials()[i];
    VariableCellReal* var = new VariableCellReal(VariableBuildInfo(defaultMesh(), String("Density_") + mat->name()));
    m_density_post_processing.add(var);
  }

  m_mesh_partitioner = options()->loadBalanceService();
  if (m_mesh_partitioner)
    info() << "Activating load balance test";
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
startInit()
{
  m_mat_not_used_real.globalVariable().setUsed(false);

  info() << "MESH_MATERIAL_TESTER :: startInit()";
  m_material_mng->forceRecompute();

  ValueChecker vc(A_FUNCINFO);

  m_nb_starting_cell.assign(parallelMng()->reduce(Parallel::ReduceSum, ownCells().size()));

  Int32UniqueArray env1_indexes;
  Int32UniqueArray mat2_indexes;
  Int32UniqueArray sub_group_indexes;
  Integer nb_cell = ownCells().size();
  IParallelMng* pm = parallelMng();
  Int64 total_nb_cell = pm->reduce(Parallel::ReduceMax, nb_cell);
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    Int64 cell_index = cell.uniqueId();
    if (cell_index > (total_nb_cell / 2))
      break;
    if (cell_index < (total_nb_cell / 5)) {
      env1_indexes.add(icell.itemLocalId());
    }
    if (cell_index < (total_nb_cell / 3) && cell_index > (total_nb_cell / 6)) {
      mat2_indexes.add(icell.itemLocalId());
    }
    if ((cell_index % 2) == 0)
      sub_group_indexes.add(icell.itemLocalId());
  }

  // Ajoute les mailles du milieu 1
  {
    Materials::MeshMaterialModifier modifier(m_material_mng);
    modifier.setDoCopyBetweenPartialAndPure(false);
    modifier.setDoInitNewItems(false);
    modifier.setPersistantWorkBuffer(false);
    Materials::IMeshEnvironment* env = m_mat1->environment();
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
    modifier.addCells(m_mat1, mat1_indexes);
    Integer nb_mat = env->nbMaterial();
    if (nb_mat > 1)
      // Ajoute les mailles du matériau 2
      modifier.addCells(env->materials()[1], mat2_indexes);
  }
  CellGroup test_group = defaultMesh()->cellFamily()->createGroup("SUB_GROUP_TEST", sub_group_indexes);
  // Ajoute les mailles du milieu 2
  if (m_mat2) {
    Materials::MeshMaterialModifier modifier(m_material_mng);
    //modifier.addCells(m_mat2->environment(),mat2_indexes);
    modifier.addCells(m_mat2, mat2_indexes);
  }

  const Integer spectral_size = 5;
  m_mat_spectral1.resize(spectral_size);
  m_mat_spectral2.resize(spectral_size * 2);
  m_env_spectral1.resize(spectral_size * 3);
  m_env_spectral2.resize(spectral_size * 4);
  // NOTE: m_env_empty_int64array ne doit pas avoir de resize
  // pour pouvoir tester la gestion des variables avec une dim2Size() nulle.

  // TODO tester que les valeurs partielles sont correctes
  m_mat_density.fillPartialValues(3.0);
  m_env_int32.fillPartialValues(5);
  m_mat_int32.fillPartialValues(8);

  info() << "Liste des mailles de test_group";
  ENUMERATE_CELL (icell, test_group) {
    info(6) << "Cell=" << ItemPrinter(*icell);
  }

  ENUMERATE_CELL (icell, allCells()) {
    m_density[icell] = 1.0;
    m_mat_density[icell] = 2.0;
    Integer idx2 = icell.itemLocalId() % spectral_size;
    m_mat_spectral1[icell][idx2] = 3.0 + (Real)(icell.itemLocalId() * spectral_size);
  }

  CellGroup mat1_cells = m_mat1->cells();
  ENUMERATE_CELL (icell, mat1_cells) {
    m_density[icell] = 2.0;
  }
  {
    VariableCellReal& gvar = m_mat_density.globalVariable();
    info() << "GVAR_NAME = " << gvar.name();
    ENUMERATE_CELL (icell, mat1_cells) {
      if (gvar[icell] != 2.0)
        fatal() << "Bad value for global variable v=" << gvar[icell];
    }
  }

  if (m_mat2) {
    CellGroup mat2_cells = m_mat2->cells();
    ENUMERATE_CELL (icell, mat2_cells) {
      m_density[icell] = 1.5;
    }
  }

  _checkTemporaryVectors(test_group);
  _checkSubViews(test_group);

  {
    OStringStream oss;
    m_material_mng->dumpInfos(oss());
    info() << oss.str();
  }

  // A supprimer
  //m_mat_density.updateFromInternal();

  m_present_material.fill(0);
  m_mat_density.fill(0.0);
  m_mat_nodump_real.fill(0.0);

  constexpr IMeshMaterial* null_mat = nullptr;
  constexpr IMeshEnvironment* null_env = nullptr;
  // Itération sur tous les milieux puis tous les matériaux
  // et toutes les mailles de ce matériau
  ENUMERATE_ENV (ienv, m_material_mng) {
    IMeshEnvironment* env = *ienv;
    info() << "ENV name=" << env->name();
    vc.areEqual(env->isEnvironment(), true, "IsEnvEnvOK");
    vc.areEqual(env->isMaterial(), false, "IsEnvMatOK");
    vc.areEqual(env->asEnvironment(), env, "ToEnvEnvOK");
    vc.areEqual(env->asMaterial(), null_mat, "ToEnvMatOK");
    ENUMERATE_MAT (imat, env) {
      Materials::IMeshMaterial* mat = *imat;
      info() << "MAT name=" << mat->name();
      vc.areEqual(mat->isEnvironment(), false, "IsMatEnvOK");
      vc.areEqual(mat->isMaterial(), true, "IsMatMatOK");
      vc.areEqual(mat->asEnvironment(), null_env, "ToMatEnvOK");
      vc.areEqual(mat->asMaterial(), mat, "ToMatMatOK");
      ENUMERATE_MATCELL (icell, mat) {
        MatCell mmcell = *icell;
        //info() << "Cell name=" << mmcell._varIndex();
        m_mat_density[mmcell] = 200.0;
        m_mat_nodump_real[mmcell] = 1.2;
        ComponentCell x1(mmcell);
        if (x1._varIndex() != mmcell._varIndex())
          ARCANE_FATAL("Bad convertsion MatCell -> ComponentCell");
        MatCell x2(x1);
        if (x1._varIndex() != x2._varIndex())
          ARCANE_FATAL("Bad convertsion ComponentCell -> MatCell");
      }
    }
  }

  // Idem mais à partir d'un bloc.
  ENUMERATE_ENV (ienv, m_block1) {
    IMeshEnvironment* env = *ienv;
    info() << "BLOCK_ENV name=" << env->name();
    ENUMERATE_MAT (imat, env) {
      Materials::IMeshMaterial* mat = *imat;
      info() << "BLOCK_MAT name=" << mat->name();
    }
  }

  // Idem mais itération sur des milieux sous forme de composants
  ENUMERATE_COMPONENT (icmp, m_material_mng->environmentsAsComponents()) {
    IMeshComponent* cmp = *icmp;
    info() << "ENV COMPONENT name=" << cmp->name();
  }

  // Itération sur les matériaux sous forme de composant.
  ENUMERATE_COMPONENT (icmp, m_material_mng->materialsAsComponents()) {
    IMeshComponent* cmp = *icmp;
    info() << "MAT COMPONENT name=" << cmp->name();
  }

  // Itération directement avec tous les matériaux du gestionnaire
  ENUMERATE_MAT (imat, m_material_mng) {
    Materials::IMeshMaterial* mat = *imat;
    info() << "MAT name=" << mat->name()
           << " density_var_name=" << m_mat_density.materialVariable()->materialVariable(mat)->name();
    ENUMERATE_MATCELL (icell, mat) {
      MatCell mmcell = *icell;
      m_mat_density[mmcell] = 200.0;
    }
  }

  if (0) {
    ENUMERATE_ENV (ienv, m_material_mng) {
      IMeshEnvironment* env = *ienv;
      info() << "Env name=" << env->name();
      ENUMERATE_ENVCELL (ienvcell, env) {
        EnvCell ev = *ienvcell;
        info() << "EnvCell nb_mat=" << ev.nbMaterial() << " cell_uid=" << ItemPrinter(ev.globalCell())
               << " component_uid=" << ev.componentUniqueId()
               << " var_index=" << ev._varIndex();
      }
    }
  }

  if (1) {
    CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
    ENUMERATE_FACE (iface, allFaces()) {
      Face face = *iface;
      Cell back_cell = face.backCell();
      if (!back_cell.null()) {
        AllEnvCell all_env_back_cell = all_env_cell_converter[back_cell];
        info() << "NB_ENV=" << all_env_back_cell.nbEnvironment();
        ComponentCell x1 = all_env_back_cell;
        AllEnvCell x2(x1);
        if (x1._varIndex() != all_env_back_cell._varIndex())
          ARCANE_FATAL("Bad convertsion AllEnvCell -> ComponentCell");
        if (x1._varIndex() != x2._varIndex())
          ARCANE_FATAL("Bad convertsion ComponentCell -> EnvCell");
      }
    }
  }

  // Itération sur tous les milieux et tous les matériaux d'une maille.
  ENUMERATE_ALLENVCELL (iallenvcell, m_material_mng, allCells()) {
    AllEnvCell all_env_cell = *iallenvcell;
    Cell global_cell = all_env_cell.globalCell();
    ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
      Real env_density = 0.0;
      ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
        MatCell mc = *imatcell;
        env_density += m_mat_density[imatcell];
        Int32 idx = mc.materialId();
        m_present_material[global_cell] = m_present_material[global_cell] | (1 << idx);
      }
      m_mat_density[ienvcell] = env_density;
      m_mat_nodump_real[ienvcell] = 3.5;
    }
  }

  if (1) {
    ENUMERATE_ALLENVCELL (iallenvcell, m_material_mng, allCells()) {
      AllEnvCell all_env_cell = *iallenvcell;
      info() << "Cell uid=" << ItemPrinter(all_env_cell.globalCell()) << " nb_env=" << all_env_cell.nbEnvironment();
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        EnvCell ec = *ienvcell;
        info() << "Cell   nb_mat=" << ec.nbMaterial()
               << " env=" << ec.environment()->name()
               << " (id=" << ec.environmentId() << ")";
        ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
          MatCell mc = *imatcell;
          info() << "Cell     mat=" << mc.material()->name()
                 << " (id=" << mc.materialId() << ")"
                 << " density=" << m_mat_density[imatcell]
                 << " index=" << mc._varIndex()
                 << " component_uid=" << mc.componentUniqueId();
          //env_density += m_mat_density[imatcell];
        }
        for (ComponentCell mc : ec.subItems()) {
          info() << "Cell     mat=" << mc.component()->name()
                 << " (id=" << mc.componentId() << ")"
                 << " density=" << m_mat_density[mc]
                 << " index=" << mc._varIndex()
                 << " component_uid=" << mc.componentUniqueId();
          //env_density += m_mat_density[imatcell];
        }
        //m_mat_density[ienvcell] = env_density;
      }

      ENUMERATE_CELL_COMPONENTCELL (ienvcell, all_env_cell) {
        ComponentCell ec = *ienvcell;
        info() << "Cell   nb_mat=" << ec.nbSubItem()
               << " env=" << ec.component()->name()
               << " (id=" << ec.componentId() << ")";
      }
    }

    const bool test_depend = false;
    // Ne doit pas être exécuté mais juste compilé pour vérifier la syntaxe
    if (test_depend) {
      m_mat_density.addDependCurrentTime(m_density);
      m_mat_density.addDependCurrentTime(m_mat_density2);
      m_mat_density.addDependPreviousTime(m_mat_density2);
      m_mat_density.removeDepend(m_mat_density2);
      m_mat_density.setComputeFunction(this, &MeshMaterialTesterModule::startInit);
    }
  }

  // Teste la récupération de la valeur de la densité partielle par maille et par matériau ou par milieu
  if (1) {
    CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);

    ENUMERATE_MAT (imat, m_material_mng) {
      IMeshMaterial* mm = *imat;
      Int32 id = mm->id();
      ENUMERATE_CELL (icell, allCells()) {
        Cell cell = *icell;
        AllEnvCell all_env_cell = all_env_cell_converter[cell];
        Real density = m_mat_density.matValue(all_env_cell, id);
        info() << "IndirectMatAccess Cell uid=" << cell.uniqueId() << " mat_id=" << id << " density=" << density;
      }
    }

    ENUMERATE_ENV (ienv, m_material_mng) {
      IMeshEnvironment* me = *ienv;
      Int32 id = me->id();
      ENUMERATE_CELL (icell, allCells()) {
        Cell cell = *icell;
        AllEnvCell all_env_cell = all_env_cell_converter[cell];
        Real density = m_mat_density.envValue(all_env_cell, id);
        info() << "IndirectEnvAccess Cell uid=" << cell.uniqueId() << " env_id=" << id << " density=" << density;
      }
    }
  }

  _checkCreation();
  _setDependencies();
  _dumpNoDumpRealValues();
  _initUnitTest();
  _checkRunQueues();

  _applyEos(true);
  _testDumpProperties();
  _checkNullComponentItem();
}

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

  {
    MaterialVariableCellReal mat_pressure(MaterialVariableBuildInfo(m_material_mng,"Pressure"));
    mat_pressure.fill(0.0);
    ENUMERATE_MATCELL(imatcell,m_mat1){
      MatCell mmc = *imatcell;
      mat_pressure[mmc] += 0.2;
    }
  }
}
#if 0
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialTesterModule::
_checkSynchronize()
{
  MaterialVariableCellInt32 m_mat_int32;
  MaterialVariableCellInt64 m_mat_int64;
  MaterialVariableCellInt64 m_mat_int64;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Exemple pour la documentation. Doit compiler mais ne sera pas exécuté.
class Sample
: public BasicModule
{
 public:
  Sample(const ModuleBuildInfo& mbi)
  : BasicModule(mbi), m_mat_density(MaterialVariableBuildInfo(0,"TestDensity"))
  {}
  MaterialVariableCellReal m_mat_density;

  //![SampleDependenciesComputeFunction]
  void _computeDensity(IMeshMaterial* mat)
  {
    ENUMERATE_MATCELL(imc,mat){
      MatCell mc = *imc;
      m_mat_density[mc] = 1.0;
    }

    // Indique que la variable est à jour.
    m_mat_density.setUpToDate(mat);
  }
  //![SampleDependenciesComputeFunction]

  void _sample()
  {
    //![SampleMaterial]
    //![SampleMaterialCreate]
    // Création ou récupération du gestionnaire depuis un maillage.
    IMeshMaterialMng* material_mng = IMeshMaterialMng::getReference(defaultMesh());
    //![SampleMaterialCreate]

    //![SampleMaterialCreate2]
    // Exemple de création de 3 matériaux:
    material_mng->registerMaterialInfo("MAT1");
    material_mng->registerMaterialInfo("MAT2");
    material_mng->registerMaterialInfo("MAT3");
    //![SampleMaterialCreate2]

    //![SampleMaterialCreate3]
    // Création du milieu ENV1 contenant les matériaux MAT1 et MAT2
    MeshEnvironmentBuildInfo ebi1("ENV1");
    ebi1.addMaterial("MAT1");
    ebi1.addMaterial("MAT2");
    IMeshEnvironment* env1 = material_mng->createEnvironment(ebi1);

    // Création du milieu ENV2 contenant le matériau MAT2
    MeshEnvironmentBuildInfo ebi2("ENV2");
    ebi2.addMaterial("MAT2");
    IMeshEnvironment* env2 = material_mng->createEnvironment(ebi2);

    // Création du milieu ENV3 contenant les matériaux MAT3 et MAT1
    MeshEnvironmentBuildInfo ebi3("ENV3");
    ebi3.addMaterial("MAT3");
    ebi3.addMaterial("MAT1");
    IMeshEnvironment* env3 = material_mng->createEnvironment(ebi3);
    
    // Création du bloc BLOCK1 sur le groupe de toutes les mailles
    // et contenant les milieux ENV1 et ENV2
    MeshBlockBuildInfo mb1("BLOCK1",allCells());
    mb1.addEnvironment(env1);
    mb1.addEnvironment(env2);
    IMeshBlock* block = material_mng->createBlock(mb1);

    // Indique au gestionnaire que l'initialisation est terminée
    material_mng->endCreate();
    //![SampleMaterialCreate3]

    //![SampleMaterialCreate4]
    info() << env1->id();                 // Affiche '0'
    info() << env1->materials()[0]->id(); // Affiche '0'
    info() << env1->materials()[1]->id(); // Affiche '1'
    info() << env2->id();                 // Affiche '1'
    info() << env2->materials()[0]->id(); // Affiche '2'
    info() << env3->id();                 // Affiche '2'
    info() << env3->materials()[0]->id(); // Affiche '3'
    info() << env3->materials()[1]->id(); // Affiche '4'
    info() << block->id();                // Affiche '0'
    //![SampleMaterialCreate4]

    //![SampleMaterialAddMat]
    {
      // Créé l'instance de modification. Les modifications
      // seront effectives lors de l'appel au destructeur de
      // cette classe.
      MeshMaterialModifier modifier(material_mng);
      // Ajoute les mailles du matériau 1 ou 2 en fonction
      // de leur localId()
      Int32UniqueArray mat1_indexes;
      Int32UniqueArray mat2_indexes;
      Integer nb_cell = allCells().size();
      ENUMERATE_CELL(icell,allCells()){
        Int32 local_id = icell.itemLocalId();
        Integer z = icell.index();
        bool add_to_mat1 = (z<(nb_cell/2) && z>(nb_cell/4));
        bool add_to_mat2 = (z>=(nb_cell/2) || z<(nb_cell/3));
        if (add_to_mat1)
          mat1_indexes.add(local_id);
        if (add_to_mat2)
          mat2_indexes.add(local_id);
      }
      // Ajoute les mailles du matériau 1
      modifier.addCells(env1->materials()[0],mat1_indexes);
      // Ajoute les mailles du matériau 2
      modifier.addCells(env1->materials()[1],mat2_indexes);
    }
    // A partir d'ici, les matériaux sont mis à jour.
    info() << env1->materials()[0]->cells().size(); // Nombre de mailles du matériau
    //![SampleMaterialAddMat]

    //![SampleMaterialCreateVariable]
    IMesh* mesh = defaultMesh();
    MaterialVariableBuildInfo mvbinfo(material_mng,"Density");
    MaterialVariableCellReal mat_density(mvbinfo);
    MaterialVariableCellReal mat_pressure(VariableBuildInfo(mesh,"Pressure"));
    //![SampleMaterialCreateVariable]

    //![SampleMaterialIterEnv]
    // Itération sur tous les milieux, puis tous les matériaux et
    // toutes les mailles de ce matériau
    ENUMERATE_ENV(ienv,material_mng){
      IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT(imat,env){
        IMeshMaterial* mat = *imat;
        ENUMERATE_MATCELL(imatcell,mat){
          MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
      ENUMERATE_ENVCELL(ienvcell,env){
        EnvCell mmcell = *ienvcell;
        info() << "Cell env=" << mmcell.environmentId();
      }
    }
    //![SampleMaterialIterEnv]

    //![SampleBlockEnvironmentIter]
    // Itération sur tous les mailles des matériaux des milieux d'un bloc.
    ENUMERATE_ENV(ienv,block){
      IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT(imat,env){
        IMeshMaterial* mat = *imat;
        ENUMERATE_MATCELL(imatcell,mat){
          MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
    }
    //![SampleBlockEnvironmentIter]

    //![SampleMaterialIterCell]
    // Itération sur tous les milieux et tous les matériaux d'une maille.
    ENUMERATE_ALLENVCELL(iallenvcell,material_mng,allCells()){
      AllEnvCell all_env_cell = *iallenvcell;
      ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
        EnvCell env_cell = *ienvcell;
        info() << "Cell env=" << env_cell.environmentId();
        ENUMERATE_CELL_MATCELL(imatcell,env_cell){
          MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
    }
    //![SampleMaterialIterCell]

    //![SampleBlockMaterialIterCell]
    // Itération sur tous les milieux et tous les matériaux d'une maille.
    ENUMERATE_ALLENVCELL(iallenvcell,block){
      AllEnvCell all_env_cell = *iallenvcell;
      ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
        EnvCell env_cell = *ienvcell;
        info() << "Cell env=" << env_cell.environmentId();
        ENUMERATE_CELL_MATCELL(imatcell,env_cell){
          MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
    }
    //![SampleBlockMaterialIterCell]


    //![SampleMaterialIterFromGroup]
    CellGroup cells;
    IMeshMaterial* mat = env1->materials()[0];
    MatCellVector mat_cells(cells,mat);
    ENUMERATE_MATCELL(imatcell,mat_cells){
      mat_density[imatcell] = 2.3;
    }
    IMeshEnvironment* env = env1;
    EnvCellVector env_cells(cells,env);
    ENUMERATE_ENVCELL(imatcell,env_cells){
      mat_density[imatcell] = 3.1;
    }
    //![SampleMaterialIterFromGroup]

    //![SampleMaterial]


    //![SampleComponentIter]
    // Itération sur tous les milieux, puis tous les matériaux et
    // toutes les mailles de ce matériau via la ComponentCell
    ENUMERATE_ENV(ienv,material_mng){
      IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT(imat,env){
        IMeshMaterial* mat = *imat;
        ENUMERATE_COMPONENTCELL(iccell,mat){
          ComponentCell cc = *iccell;
          info() << "Cell mat=" << cc.componentId();
          mat_density[cc] = 3.1; // Met à jour la densité du matériau
        }
      }
      ENUMERATE_COMPONENTCELL(iccell,env){
        ComponentCell cc = *iccell;
        info() << "Cell env=" << cc.componentId();
        mat_density[cc] = 2.5; // Met à jour la densité du milieu
      }
    }
    //![SampleComponentIter]

    {
      //![SampleComponentSuperItem]
      MatCell mc;
      ComponentCell cc = mc;
      // Retourne la maille milieu (EnvCell) du matériau:
      ComponentCell cc2 = cc.superCell();
      // Itère sur les mailles matériaux du milieu
      ENUMERATE_CELL_COMPONENTCELL(icc,cc2){
      }

      // Retourne la maille AllEnvCell du milieu:
      ComponentCell cc3 = cc2.superCell();
      // Itère sur les mailles milieu de la maille.
      ENUMERATE_CELL_COMPONENTCELL(icc,cc3){
      }
      //![SampleComponentSuperItem]
    }

    {
      Real init_val = 0.0;
      MaterialVariableCellReal& var = mat_density;
      // Initialise la valeur globale
      var.globalVariable().fill(init_val);
      ENUMERATE_ENV(ienv,material_mng){
        // Initialise les valeurs milieux
        ENUMERATE_ENVCELL(ienvcell,(*ienv)){
          var[ienvcell] = init_val;
        }
        // Initialise les valeurs matériaux
        ENUMERATE_MAT(imat,(*ienv)){
          ENUMERATE_MATCELL(imatcell,(*imat)){
            var[imatcell] = init_val;
          }
        }
      }
    }

    {
      //![SampleDependencies]
      // Positionne la méthode de calcul.
      mat_density.setMaterialComputeFunction(this,&Sample::_computeDensity);
      // Ajoute dépendance sur une variable matériau
      mat_density.addMaterialDepend(mat_pressure);
      // Ajoute dépendance sur variables globales
      mat_density.addMaterialDepend(defaultMesh()->nodesCoordinates());
      mat_density.addMaterialDepend(m_global_time);

      ENUMERATE_MAT(imat,material_mng){
        IMeshMaterial* mat = *imat;
        // Met à jour la variable sur le matériau \a mat si besoin.
        mat_density.update(mat);
      }
      //![SampleDependencies]
    }

    {
      //![SampleConcurrency]
      // Boucle parallèle sur les mailles du milieu env1
      IMeshEnvironment* env = env1;
      Parallel::Foreach(env->envView(),[&](EnvItemVectorView view)
        {
          ENUMERATE_ENVCELL(ienvcell,view){
            mat_density[ienvcell] = 2.5;
          }
        });

      // Boucle parallèle sur les mailles du premier matériaux de env1
      IMeshMaterial* mat = env1->materials()[0];
      Parallel::Foreach(mat->matView(),[&](MatItemVectorView view)
        {
          ENUMERATE_MATCELL(imatcell,view){
            mat_density[imatcell] = 2.5;
          }
        });

      // Functor générique sur un matériau ou milieu.
      auto func = [&](ComponentItemVectorView view)
      {
        ENUMERATE_COMPONENTCELL(iccell,view){
          mat_density[iccell] = 2.5;
        }
      };
      // Application en parallèle de \a func sur le matériau
      Parallel::Foreach(mat->view(),func);
      // Application en parallèle de \a func sur le milieu
      Parallel::Foreach(env->view(),func);

      // Application en parallèle de \a func sur le milieu avec options
      ParallelLoopOptions options;
      Parallel::Foreach(env->view(),options,func);
      //![SampleConcurrency]
    }

    {
      //![SampleEnumerateVariableDeclaration]
      MaterialVariableCellReal mat_pressure(VariableBuildInfo(mesh,"Pressure"));
      MaterialVariableCellReal mat_volume(VariableBuildInfo(mesh,"Volume"));
      MaterialVariableCellReal mat_temperature(VariableBuildInfo(mesh,"Temperature"));
      //![SampleEnumerateVariableDeclaration]

      //![SampleEnumerateSimdComponentItem]
      Real nr = 1.0;
      // Température et volume en lecture seule
      auto in_volume = viewIn(mat_volume);
      auto in_temperature = viewIn(mat_temperature);
      // Pression en écriture
      auto out_pressure = viewOut(mat_pressure);

      ENUMERATE_COMPONENTITEM_LAMBDA(EnvPartSimdCell,scell,env){
        out_pressure[scell] = nr * in_temperature[scell] / in_volume[scell];
      };
      //![SampleEnumerateSimdComponentItem]

      //![SampleEnumerateComponentItemEnv]
      CellGroup test_env_group;
      IMeshEnvironment* env = env1;
      EnvCellVector env_vector(test_env_group,env);

      // Boucle sur les mailles du milieu \a env
      ENUMERATE_COMPONENTITEM(EnvCell,ienvcell,env){
        EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles du milieu \a env_vector
      ENUMERATE_COMPONENTITEM(EnvCell,ienvcell,env_vector){
        EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles pures du milieu \a env
      ENUMERATE_COMPONENTITEM(EnvPartCell,ienvcell,env->pureEnvItems()){
        EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles pures du milieu \a env
      ENUMERATE_COMPONENTITEM(EnvPartCell,ienvcell,env,eMatPart::Pure){
        EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles impures du milieu \a env
      ENUMERATE_COMPONENTITEM(EnvPartCell,ienvcell,env->impureEnvItems()){
        EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles impures du milieu \a env
      ENUMERATE_COMPONENTITEM(EnvPartCell,ienvcell,env,eMatPart::Impure){
        EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      //![SampleEnumerateComponentItemEnv]

      //![SampleEnumerateComponentItemMat]
      CellGroup test_mat_group;
      IMeshMaterial* mat = env1->materials()[0];
      MatCellVector mat_vector(test_mat_group,mat);

      // Boucle sur les mailles du matériau \a mat
      ENUMERATE_COMPONENTITEM(MatCell,imatcell,mat){
        MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles du matériau \a mat_vector
      ENUMERATE_COMPONENTITEM(MatCell,imatcell,mat_vector){
        MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM(MatPartCell,imatcell,mat->pureMatItems()){
        MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM(MatPartCell,imatcell,mat,eMatPart::Pure){
        MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM(MatPartCell,imatcell,mat->impureMatItems()){
        MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM(MatPartCell,imatcell,mat,eMatPart::Impure){
        MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }
      //![SampleEnumerateComponentItemMat]

      //![SampleEnumerateComponentItemComponent]
      // Boucle générique sur les mailles du matériau \a mat
      ENUMERATE_COMPONENTITEM(ComponentCell,iccell,mat){
        ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles du matériau \a mat_vector
      ENUMERATE_COMPONENTITEM(ComponentCell,iccell,mat_vector){
        ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM(ComponentPartCell,iccell,mat->pureItems()){
        ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM(ComponentPartCell,iccell,mat,eMatPart::Pure){
        ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM(ComponentPartCell,iccell,mat->impureItems()){
        ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM(ComponentPartCell,iccell,mat,eMatPart::Impure){
        ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }
      //![SampleEnumerateComponentItemComponent]

      
    }
  }
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MESHMATERIALTESTER(MeshMaterialTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
