// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultipleMeshUnitTest.cc                                     (C) 2000-2019 */
/*                                                                           */
/* Teste l'utilisation de maillage multiples.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/List.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/Properties.h"
#include "arcane/IMainFactory.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMeshModifier.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IVariableMng.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/IParallelMng.h"
#include "arcane/VariableCollection.h"
#include "arcane/IMeshMng.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MeshUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test l'utilisation de maillage multiples.
 */
class MultipleMeshUnitTest
: public ArcaneMeshUnitTestObject
{
 public:

  MultipleMeshUnitTest(const ServiceBuildInfo& cb);
  ~MultipleMeshUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  void _writePostProcessing(IMesh* new_mesh,String directory);
  IPrimaryMesh* _testMesh(const String& mesh_name,bool do_output);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHUNITTEST(MultipleMeshUnitTest,MultipleMeshUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultipleMeshUnitTest::
MultipleMeshUnitTest(const ServiceBuildInfo& mb)
: ArcaneMeshUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultipleMeshUnitTest::
~MultipleMeshUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultipleMeshUnitTest::
executeTest()
{
  IMeshMng* mm = subDomain()->meshMng();
  _testMesh("Mesh2",true);
  Integer nb_to_add = 8;
  for( Integer i=0; i<nb_to_add; ++i ){
    String name = String("MeshToAdd") + String::fromNumber(i);
    IPrimaryMesh* mesh = _testMesh(name,false);
    mm->destroyMesh(mesh->handle());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MultipleMeshUnitTest::
_testMesh(const String& mesh_name,bool do_output)
{
  ISubDomain* sd = subDomain();
  IMesh* current_mesh = mesh();
  IParallelMng* pm = mesh()->parallelMng();
  IMainFactory* mf = sd->application()->mainFactory();
  IPrimaryMesh* new_mesh = mf->createMesh(sd,pm->sequentialParallelMng(),mesh_name);
  VariableNodeReal3 mesh_coords(current_mesh->nodesCoordinates());

  // Création d'un nouveau maillage contenant uniquement la première moitié
  // des mailles du maillage initial.

  // Pour tester les performances de l'ajout incrémental de maille,
  // on désactive les vérifications, le compactage et le tri et on appelle endUpdate()
  // pour chaque maille ajoutée.
  new_mesh->setDimension(mesh()->dimension());
  new_mesh->properties()->setBool("compact",false);
  new_mesh->properties()->setBool("sort",false);
  new_mesh->allocateCells(0,Int64ConstArrayView(),false);
  new_mesh->endAllocate();
  IItemFamily* new_cell_family = new_mesh->cellFamily();
  VariableNodeReal3 new_mesh_coords(new_mesh->nodesCoordinates());
  VariableCellReal cell_coord(VariableBuildInfo(new_mesh,"CellCoord"));
  Int32UniqueArray new_cells_local_id(1);
  // Pour tester la destruction d'entités, fait 3 fois la
  // même opération
  Integer max_create = 3;
  for( Integer icreate=0; icreate<max_create; ++icreate ){
    CellGroup all_cells = mesh()->allCells();
    Integer max_index = all_cells.size() / (1 + max_create - icreate);
    info() << "Doing iteration i=" << icreate << " mesh=" << mesh_name << " max_index=" << max_index;
    new_mesh->modifier()->clearItems();
    // Désactive les vérifications au cours de la création pour
    // des raisons de performance.
    new_mesh->setCheckLevel(0);
    Int64UniqueArray cells_infos;
    IMeshModifier* modifier = new_mesh->modifier();
    Integer index = 0;
    ENUMERATE_CELL(icell,all_cells){
      if (index>(all_cells.size()/2))
        break;
      Cell cell = *icell;
      Integer nb_node = cell.nbNode();
      cells_infos.resize(nb_node+2);
      cells_infos[0] = cell.type();
      cells_infos[1] = cell.uniqueId();
      for( Integer i=0; i<nb_node; ++i )
        cells_infos[2+i] = cell.node(i).uniqueId();
      modifier->addCells(1,cells_infos,new_cells_local_id);
    }
    modifier->endUpdate();
    ENUMERATE_CELL(icell,all_cells){
      if (index>(all_cells.size()/2))
        break;
      Cell cell = *icell;
      Integer nb_node = cell.nbNode();
      //if ((index%30)==0)
      //info() << " new_cell_local_id=" << new_cells_local_id[0];
      Cell new_cell = new_cell_family->findOneItem(cell.uniqueId());
      // met à jour les coordonnées du nouveau maillage à partir de l'ancien
      for( Integer i=0; i<nb_node; ++i )
        new_mesh_coords[new_cell.node(i)] = mesh_coords[cell.node(i)];
      ++index;
    }

    // Vérifie que le maillage est valide.
    new_mesh->checkValidMesh();
    ENUMERATE_CELL(icell,new_mesh->allCells()){
      Cell cell = *icell;
      cell_coord[icell] = new_mesh_coords[cell.node(0)].abs2();
    }
  }

  if (do_output){
    // Pour test, fait une sortie de dépouillement sur ce nouveau maillage.
    StringBuilder outdir("test");
    outdir += pm->commRank();
    _writePostProcessing(new_mesh,outdir);
  }

  return new_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultipleMeshUnitTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultipleMeshUnitTest::
_writePostProcessing(IMesh* new_mesh,String directory)
{
  RealUniqueArray times(1);
  times[0] = 1.0;

  ServiceBuilder<IPostProcessorWriter> sbuilder(subDomain());
  String service_name = "Ensight7PostProcessor";
  auto post_processor(sbuilder.createReference(service_name,new_mesh,SB_AllowNull));
  //IServiceMng* sm = subDomain()->serviceMng();
  //post_processor = post_processor_factory.createInstance(service_name,new_mesh,true);
  if (!post_processor){
    info() << "The specified service for the output (" << service_name << ") is not"
           << " available";
    return;
  }

  VariableList variables;
  variables.add(new_mesh->toPrimaryMesh()->nodesCoordinates().variable());
  VariableCellReal cell_coord(VariableBuildInfo(new_mesh,"CellCoord"));
  variables.add(cell_coord.variable());

  ItemGroupList groups;
  groups.add(new_mesh->allCells());
  groups.add(new_mesh->allNodes());

  post_processor->setBaseDirectoryName(directory);
  post_processor->setTimes(times);
  post_processor->setVariables(variables);
  post_processor->setGroups(groups);
  info() << "** Output for test mesh nb_var=" << variables.count()
         << " directory=" << directory;
  
  subDomain()->variableMng()->writePostProcessing(post_processor.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
