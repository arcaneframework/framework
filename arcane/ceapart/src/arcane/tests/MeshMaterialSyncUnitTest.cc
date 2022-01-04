// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSyncUnitTest.cc                                 (C) 2000-2014 */
/*                                                                           */
/* Service de test de la synchronisation des matériaux.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/OStringStream.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ItemPrinter.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MeshMaterialSyncUnitTest_axl.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test pour la gestion des matériaux et des milieux.
 */
class MeshMaterialSyncUnitTest
: public ArcaneMeshMaterialSyncUnitTestObject
{
 public:


  MeshMaterialSyncUnitTest(const ServiceBuildInfo& mbi);
  ~MeshMaterialSyncUnitTest();

 public:

  virtual void initializeTest();
  virtual void executeTest();

 private:
  
  IMeshMaterialMng* m_material_mng;

  void _checkVariableSync();
  void _doPhase1();
  void _doPhase2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSyncUnitTest::
MeshMaterialSyncUnitTest(const ServiceBuildInfo& sbi)
: ArcaneMeshMaterialSyncUnitTestObject(sbi)
, m_material_mng(IMeshMaterialMng::getReference(sbi.mesh()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSyncUnitTest::
~MeshMaterialSyncUnitTest()
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
  _checkVariableSync();
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
  _checkVariableSync();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSyncUnitTest::
_checkVariableSync()
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
    fatal() << "Bad variable synchronization";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHMATERIALSYNCUNITTEST(MeshMaterialSyncUnitTest,
                                                 MeshMaterialSyncUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
