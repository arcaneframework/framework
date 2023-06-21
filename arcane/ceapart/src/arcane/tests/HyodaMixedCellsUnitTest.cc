// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HyodaMixedCellsUnitTest.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Service du test de l'affichage des mailles mixtes.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/BasicTimeLoopService.h"

#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeLoopService.h"
#include "arcane/ITimeLoop.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IParallelMng.h"


#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemVector.h"

#include <arcane/hyoda/Hyoda.h>
#include <arcane/hyoda/HyodaMix.h>
#include "arcane/utils/IOnlineDebuggerService.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/HyodaMixedCellsUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


class HyodaMixedCellsUnitTest: public ArcaneHyodaMixedCellsUnitTestObject{
public:
  HyodaMixedCellsUnitTest(const ServiceBuildInfo &sbi):
  ArcaneHyodaMixedCellsUnitTestObject(sbi),
  m_sub_domain(subDomain()),
  m_material_mng(IMeshMaterialMng::getReference(subDomain()->defaultMesh())),
  m_theta(0.0),
  //m_density(Materials::MaterialVariableBuildInfo(m_material_mng, "density")),
  //m_density(VariableBuildInfo(subDomain()->defaultMesh(),"density")),
  m_interface_normal(VariableBuildInfo(subDomain()->defaultMesh(), "InterfaceNormal")),
  m_interface_distance(VariableBuildInfo(subDomain()->defaultMesh(), "InterfaceDistance2")){
  }
  ~HyodaMixedCellsUnitTest(){}
public:
  // **************************************************************************
  // * build
  // **************************************************************************
  void build(){
    info() << "\33[7m[HyodaMixedCellsUnitTest::build]\33[m";
    // On demande à HyODA d'afficher les interfaces
    //if (platform::getOnlineDebuggerService()) platform::getOnlineDebuggerService()->doMixedCells();
  }

  
  // **************************************************************************
  // * onTimeLoopStartInit
  // **************************************************************************
  void onTimeLoopStartInit(){
    info() << "\33[7m[HyodaMixedCellsUnitTest::onTimeLoopStartInit]\33[m";
    // Initialisation du pas de temps
    m_global_deltat=1.0;
    
    ENUMERATE_FACE(face,allFaces()){
      m_qedge[face]=(Real)face->uniqueId().asInteger();
    }
    
    // Lecture des matériaux
    for(Integer i=0,n=options()->material().size(); i<n; ++i){
      String mat_name = options()->material[i].name;
      info() << "Registering material" << mat_name;
      m_material_mng->registerMaterialInfo(mat_name);
    }

    // Lecture des milieux
    for(Integer i=0,n=options()->environment().size(); i<n; ++i){
      String env_name = options()->environment[i].name;
      info() << "Creating environment name=" << env_name;
      Materials::MeshEnvironmentBuildInfo env_build(env_name);
      for( Integer k=0,kn=options()->environment[i].material.size(); k<kn; ++k ){
        String mat_name = options()->environment[i].material[k];
        info() << "\tAdding material " << mat_name << " for environment " << env_name;
        env_build.addMaterial(mat_name);
      }
      m_material_mng->createEnvironment(env_build);
    }
    // Signale au gestionnaire que tous les milieux ont été créés et qu'il peut allouer les variables.
    m_material_mng->endCreate();

    // Récupération des xMax et calcul des incréments en x: ix
    Real maxX=0.0;
    Real maxY=0.0;
    ENUMERATE_CELL(cell, allCells()){
      for (Node node : cell->nodes() ){
        maxX=math::max(maxX, nodesCoordinates()[node].x);
        maxY=math::max(maxY, nodesCoordinates()[node].y);
      }
    }
    m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax,maxX);
    m_sub_domain->parallelMng()->reduce(Parallel::ReduceMax,maxY);

    // On découpe en x les différents milieux
    Real ix=maxX/options()->environment().size();
    info() << "maxX="<<maxX<<", ix="<<ix;
    info() << "maxY="<<maxY;
    
    UniqueArray2< UniqueArray<Int32> > ids(options()->environment().size(),options()->material().size());
    ids.fill(UniqueArray<Int32>());
    //Int32Array ids[options()->environment().size()][options()->material().size()];
    ENUMERATE_CELL(cell, allCells()){
      //info() << "Cell #" << cell->uniqueId()<<", .x0="<<nodesCoordinates()[cell->node(0)].x;
      //info() << "Cell #" << cell->uniqueId()<<", .y2="<<nodesCoordinates()[cell->node(2)].y;
      //info() << "floor(x/ix)="<<math::floor(nodesCoordinates()[cell->node(0)].x/ix);
      Int32 iEnv = Convert::toInt32(math::floor(nodesCoordinates()[cell->node(0)].x/ix));
      Real iy=maxY/m_material_mng->environments()[iEnv]->nbMaterial();
      Int32 iMat0 = Convert::toInt32(math::floor(nodesCoordinates()[cell->node(0)].y/iy));
      //Int32 iMat2=math::floor(nodesCoordinates()[cell->node(2)].y/iy);
      //info()<<"iMat0="<<iMat0<<", iMat2="<<iMat2;
      ids[iEnv][iMat0].add(cell.itemLocalId());
      //if (iMat2!=iMat0) ids[iEnv][iMat2].add(cell.itemLocalId());
    }

    // On rajoute en dur les mailles 1 et 4
    ids[0][1].add(1);
    ids[0][1].add(4);
      
    // Une fois les matériaux et milieux créés, il est possible d'ajouter ou de
    // supprimer des mailles pour un matériau. Il n'est pas nécessaire de
    // modifier les mailles par milieu: Arcane se charge de recalculer automatiquement
    // la liste des mailles d'un milieu en fonction de celles de ses matériaux.
    for(Integer i=0,n=options()->environment().size(); i<n; ++i){
      IMeshEnvironment *env = m_material_mng->environments()[i];
      info() << "[EnvInit] "<<env->name()<<", nbMaterial="<<env->nbMaterial();
      for(Integer j=0,jMax=env->nbMaterial(); j<jMax; ++j){
        MeshMaterialModifier modifier(m_material_mng);
        modifier.addCells(env->materials()[j],ids[i][j]);
        info() << "\t[EnvInit] adding cell #"<<ids[i][j];
      }
    }


    // Itération sur tous les milieux, puis tous les matériaux et
    // toutes les mailles de ce matériau
     ENUMERATE_ENV(ienv, m_material_mng){
      IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT(imat,env){
        IMeshMaterial* mat = *imat;
        ENUMERATE_MATCELL(imatcell,mat){
          MatCell mc = *imatcell;
          info() << "Cell #"<<mc.globalCell().localId()<<" mat=" << mc.materialId();
          m_density[mc]=1.0+Real(mc.globalCell().localId()+mc.materialId());
          //if (mc.globalCell().localId()==1) 
        }
      }
      ENUMERATE_ENVCELL(ienvcell,env){
        EnvCell mmcell = *ienvcell;
        info() << "Cell env=" << mmcell.environmentId();
        m_density[mmcell]=10.0+ Real(mmcell.environmentId());
      }
    }

     ENUMERATE_CELL(cell,allCells()){
      m_density[*cell] = (double)cell.itemLocalId();
      info()<<"m_density[*cell]="<<m_density[*cell];
    }

    // Initialisation des variables
    m_interface_distance.resize(1);
    ENUMERATE_CELL(cell, allCells()){
      //info()<<"\t[onTimeLoopStartInit] cell #"<<cell->uniqueId();
      //m_interface_distance2[cell][0]=1.0/sqrt(2.0);
      m_interface_distance[cell][0]=((nodesCoordinates()[cell->node(0)]-(nodesCoordinates()[cell->node(0)]+
                                                                         nodesCoordinates()[cell->node(1)]+
                                                                         nodesCoordinates()[cell->node(2)]+
                                                                         nodesCoordinates()[cell->node(3)])/4.0).normL2())/2.0;
      //info()<<"\t[onTimeLoopStartInit] m_interface_distance2="<<m_interface_distance2[cell][0];
      //m_concentration[cell]=1.0/(1.0+((Real)cell->uniqueId().asInt32()));
    }
  }

  
  // **************************************************************************
  // * onTimeLoopBeginLoop
  // **************************************************************************
  void onTimeLoopBeginLoop(){
    const Real3 normal=Real3(cos(m_theta),sin(m_theta),0.0);
    info() << "\33[7m[HyodaMixedCellsUnitTest::onTimeLoopBeginLoop]\33[m m_theta="<<m_theta;
    info() << "[HyodaMixedCellsUnitTest::onTimeLoopBeginLoop] normal="<<normal;
    ARCANE_HYODA_SOFTBREAK(subDomain());
    if (m_global_iteration()>options()->iterations())
      subDomain()->timeLoopMng()->stopComputeLoop(true);
    ENUMERATE_CELL(cell, allCells()){
      m_interface_normal[cell]=normal; 
      //m_interface_distance2[cell][0]+=0.123;
      //if (m_interface_distance2[cell][0]>=+1.0) m_interface_distance2[cell][0]=0.0;
    }
    m_theta+=cgrPI/180.0;
    /*ENUMERATE_CELL(cell,allCells()){
      info()<<"m_density[*cell]="<<m_density[*cell];
      }*/
  }

  
  // **************************************************************************
  // * onTimeLoopRestore
  // **************************************************************************
  void onTimeLoopRestore(){
    info() << "\33[7m[HyodaMixedCellsUnitTest::onTimeLoopRestore]\33[m";
  }
  
private:
  ISubDomain *m_sub_domain;
  IMeshMaterialMng* m_material_mng;
  Real m_theta;
  //MaterialVariableCellReal m_density;
  VariableCellReal3 m_interface_normal;
  VariableCellArrayReal m_interface_distance;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HYODAMIXEDCELLSUNITTEST(HyodaMixedCellsUnitTest,
                                                HyodaMixedCellsUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
