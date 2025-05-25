// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialTesterModule_Init.cc                            (C) 2000-2025 */
/*                                                                           */
/* Module de test du gestionnaire des matériaux.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/MeshMaterialTesterModule.h"

#include "arcane/utils/ValueChecker.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/materials/IMeshBlock.h"

#include "arcane/materials/MeshBlockBuildInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/core/materials/CellToAllEnvCellConverter.h"


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

ARCANE_REGISTER_MODULE_MESHMATERIALTESTER(MeshMaterialTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
