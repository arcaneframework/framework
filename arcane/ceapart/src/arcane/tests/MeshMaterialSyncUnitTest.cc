// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSyncUnitTest.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Service de test de la synchronisation des matériaux.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/OStringStream.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMesh.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/RunCommandMaterialEnumerate.h"
#include "arcane/accelerator/MaterialVariableViews.h"
#include "arcane/accelerator/VariableViews.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MeshMaterialSyncUnitTest_axl.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;
using namespace Arcane::Materials;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test pour la gestion des matériaux et des milieux.
 */
class MeshMaterialSyncUnitTest
: public ArcaneMeshMaterialSyncUnitTestObject
{
 public:

  explicit MeshMaterialSyncUnitTest(const ServiceBuildInfo& mbi);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:
  
  IMeshMaterialMng* m_material_mng;
  VariableCellByte m_variable_is_own_cell;
  VariableCellInt64 m_cell_unique_ids;
  MaterialVariableCellInt64 m_material_uids;

  void _checkVariableSync1();
  void _doPhase1();
  void _doPhase2();

 public:

  void _checkVariableSync2(bool do_check,Int32 iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSyncUnitTest::
MeshMaterialSyncUnitTest(const ServiceBuildInfo& sbi)
: ArcaneMeshMaterialSyncUnitTestObject(sbi)
, m_material_mng(IMeshMaterialMng::getReference(sbi.mesh()))
, m_variable_is_own_cell(VariableBuildInfo(sbi.meshHandle(),"CellIsOwn"))
, m_cell_unique_ids(VariableBuildInfo(sbi.meshHandle(),"CellUniqueId"))
, m_material_uids(MaterialVariableBuildInfo(m_material_mng,"SyncMaterialsUid"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
initializeTest()
{
  IMeshMaterialMng* mm = m_material_mng;
  Integer nb_mat = options()->nbMaterial();
  info() << "Number of wanted materials: " << nb_mat;

    // Lit les infos des matériaux du JDD et les enregistre dans le gestionnaire
  for( Integer i=0; i<nb_mat; ++i ){
    String mat_name = String("MAT_") + String::fromNumber(i);
    mm->registerMaterialInfo(mat_name);
  }

  // Créé des milieux en fonction du nombre de matériaux du jeu de données.
  {
    Integer env_index = 1;
    Integer mat_index = 0;

    while(mat_index<nb_mat){
      String env_name = "ENV_" + String::fromNumber(env_index);
      Materials::MeshEnvironmentBuildInfo env_build(env_name);
      // Utilise un std::set pour être sur qu'on n'ajoute pas 2 fois le même matériau.
      std::set<String> mats_in_env;
      for( Integer z=0; z<=env_index; ++z ){
        String mat1_name = "MAT_" + String::fromNumber(mat_index);
        mats_in_env.insert(mat1_name);
        // Ajoute aussi des matériaux qui sont dans les milieux précédents
        // pour être sur d'avoir des matériaux qui appartiennent à plusieurs milieux.
        String mat2_name = "MAT_" + String::fromNumber(mat_index/2);
        mats_in_env.insert(mat2_name);

        ++mat_index;
        if (mat_index>=nb_mat)
          break;
      }
      for( String mat_name : mats_in_env ){
        info() << "Add material " << mat_name << " for environment " << env_name;
        env_build.addMaterial(mat_name);
      }
      mm->createEnvironment(env_build);
      ++env_index;
    }

    mm->endCreate(false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
executeTest()
{
  _doPhase1();

  {
    OStringStream ostr;
    m_material_mng->dumpInfos(ostr());
    info() << ostr.str();
  }

  _doPhase2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
_doPhase1()
{
  // Cette phase se fait juste après l'init donc on a déjà créé les
  // matériaux mais on n'a pas encore de mailles dedans.

  // Ajoute des mailles dans les matériaux.
  // Pour tester la synchronisation, on n'ajoute que des matériaux aux
  // mailles propres. Ainsi, il faudra que quelqu'un nous envoie les
  // informations pour les mailles fantômes.

  IMeshMaterialMng* mm = m_material_mng;
  Integer nb_mat = options()->nbMaterial();

  {
    CellGroup cells = ownCells();
    MeshMaterialModifier mmodifier(m_material_mng);
    Int32UniqueArray ids;
    // TODO: calculer en fonction du max des uid.
    for( Integer imat=0; imat<nb_mat; ++imat ){
      ids.clear();
      Int64 min_uid = imat*10;
      Int64 max_uid = min_uid + 10 + imat*10;
      ENUMERATE_CELL(icell,cells){
        Cell cell = *icell;
        Int64 uid = cell.uniqueId();
        if (uid<max_uid && uid>min_uid)
          ids.add(cell.localId());
      }
      info() << "Adding cells n=" << ids.size() << " to mat " << imat << " (min_uid="
             << min_uid << " max_uid=" << max_uid << ")";
      mmodifier.addCells(mm->materials()[imat],ids);
    }
  }

  m_material_mng->synchronizeMaterialsInCells();
  m_material_mng->checkMaterialsInCells();
  _checkVariableSync1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
_doPhase2()
{
  info() << "Begin phase2";
  IMeshMaterialMng* mm = m_material_mng;
  Integer nb_mat = options()->nbMaterial();
  // Cette phase doit se faire après la phase1 et supprime
  // des mailles qui ont été ajoutées dans la phase 1.

  // Ajoute et supprime des mailles dans les matériaux.
  // Pour tester la synchronisation, on ne travaille que sur
  // les mailles dont on est propriétaires. Ainsi, il faudra que
  // quelqu'un nous envoie les informations pour les mailles fantômes.
  {
    CellGroup cells = ownCells();
    MeshMaterialModifier mmodifier(m_material_mng);
    Int32UniqueArray add_ids;
    Int32UniqueArray remove_ids;
    // TODO: calculer en fonction du max des uid.
    for( Integer imat=0; imat<nb_mat; ++imat ){
      remove_ids.clear();
      add_ids.clear();
      Int64 phase1_min_uid = imat*10;
      Int64 phase1_max_uid = phase1_min_uid + 10 + imat*10;

      Int64 min_uid = phase1_max_uid  + 5;
      Int64 max_uid = min_uid + 20 + imat*5;
      ENUMERATE_CELL(icell,cells){
        Cell cell = *icell;
        Int64 uid = cell.uniqueId();
        if (uid<phase1_max_uid && uid>phase1_min_uid)
          remove_ids.add(cell.localId());
        else if (uid<max_uid && uid>min_uid)
          add_ids.add(cell.localId());
      }
      info() << "Adding cells n=" << add_ids.size() << " to mat " << imat << " (min_uid="
             << min_uid << " max_uid=" << max_uid << ")";
      info() << "Removing cells n=" << remove_ids.size() << " to mat " << imat << " (min_uid="
             << phase1_min_uid << " max_uid=" << phase1_max_uid << ")";
      IMeshMaterial* mat = mm->materials()[imat];
      mmodifier.removeCells(mat,remove_ids);
      mmodifier.addCells(mat,add_ids);
    }
  }

  m_material_mng->synchronizeMaterialsInCells();
  m_material_mng->checkMaterialsInCells();
  _checkVariableSync1();
  m_variable_is_own_cell.fill(0);
  ENUMERATE_(Cell,icell,ownCells()){
    m_variable_is_own_cell[icell] = 1;
  }
  ENUMERATE_(Cell,icell,allCells()){
    Cell cell = *icell;
    m_cell_unique_ids[cell] = cell.uniqueId();
  }

  for( int i=0; i<10; ++i )
    _checkVariableSync2(false,i);
  _checkVariableSync2(true,5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
_checkVariableSync1()
{
  // Vérifie que la synchronisation des variables marche bien
  MaterialVariableCellInt32 mat_indexes(MaterialVariableBuildInfo(m_material_mng,"SyncMatIndexes"));

  ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng,ownCells()){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
        MatCell mc = *imatcell;
        mat_indexes[mc] = mc.materialId() + 1;
      }
    }
  }

  mat_indexes.synchronize();

  Integer nb_error = 0;
  ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng,allCells()){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
        MatCell mc = *imatcell;
        if (mat_indexes[mc] != (mc.materialId() + 1)){
          ++nb_error;
          if (nb_error<10)
            error() << "VariableSync error mat=" << mc.materialId()
                    << " mat_index=" << mat_indexes[mc]
                    << " cell=" << ItemPrinter(mc.globalCell());
        }
      }
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad variable synchronization nb_error={0}",nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
_checkVariableSync2(bool do_check,Int32 iteration)
{
  // TODO: Faire aussi avec les matériaux lorsqu'ils seront disponibles
  // TODO: Ne pas mettre la même valeur dans chaque maille milieu/matériau (faire un offset du uid)
  Arcane::Accelerator::RunQueue* queue = subDomain()->acceleratorMng()->defaultQueue();

  // Vérifie que la synchronisation des variables marche bien

  ENUMERATE_ENV(ienv, m_material_mng) {
    IMeshEnvironment* env = *ienv;
    EnvCellVectorView envcellsv = env->envView();
    auto cmd = makeCommand(queue);
    auto out_mat_uids = viewOut(cmd, m_material_uids);
    auto in_is_own_cell = ax::viewIn(cmd, m_variable_is_own_cell);
    auto in_cell_uids = ax::viewIn(cmd, m_cell_unique_ids);
    cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv) {
      auto [mvi, cid] = evi();
      if (in_is_own_cell[cid]==1){
        out_mat_uids[mvi] = in_cell_uids[cid];
      }
    };
  }

  // Avec la version 7 des synchronisations, on teste une fois sur deux la version
  // non bloquante.
  if ((iteration%2)==0 && m_material_mng->synchronizeVariableVersion()==7){
    MeshMaterialVariableSynchronizerList vlist(m_material_mng);
    m_material_uids.synchronize(vlist);
    vlist.beginSynchronize();
    vlist.endSynchronize();
  }
  else
    m_material_uids.synchronize();

  if (!do_check)
    return;

  Integer nb_error = 0;
  ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng,allCells()){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      EnvCell mc = *ienvcell;
      Cell global_cell = mc.globalCell();
      if (m_material_uids[mc] != global_cell.uniqueId()){
        ++nb_error;
        if (nb_error<10)
          error() << "VariableSync error mat=" << mc
                  << " uid_value=" << m_material_uids[mc]
                  << " cell=" << ItemPrinter(mc.globalCell());
      }
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad variable synchronization nb_error={0}",nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHMATERIALSYNCUNITTEST(MeshMaterialSyncUnitTest,
                                                 MeshMaterialSyncUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
