// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshModificationTester.cc                                   (C) 2000-2023 */
/*                                                                           */
/* Service du test de la modification du maillage.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/BasicUnitTest.h"

#include "arcane/tests/MeshModificationTester_axl.h"

#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshModificationTester
: public ArcaneMeshModificationTesterObject
{

public:

  MeshModificationTester(const ServiceBuildInfo& sb)
    : ArcaneMeshModificationTesterObject(sb) {}
  
  ~MeshModificationTester() {}

 public:

  virtual void initializeTest();
  virtual void executeTest();
  
 private:
  void _refineCells();
  Int64 _searchMaxUniqueId(ItemGroup group);

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHMODIFICATIONTESTER(MeshModificationTester,MeshModificationTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshModificationTester::
initializeTest()
{ 
  info() << "init test";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshModificationTester::
executeTest()
{
  info() << "execute test";
  
  _refineCells(); // copied from MeshModifications.cs C# for easier debug

  Int32UniqueArray lids(allCells().size());
  ENUMERATE_CELL(icell,allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type=" 
	   << icell->type() << ", nb nodes=" << icell->nbNode();
    lids[icell.index()] = icell->localId();
  }

  // On vide le maillage
  IMeshModifier* modifier = mesh()->modifier();

  modifier->removeCells(lids);

  modifier->endUpdate();

  info() << "===================== THE MESH IS EMPTY";

  // On ajoute des noeuds
  Int64UniqueArray nodes_uid(5);
  for(Integer i = 0; i < 5; ++i)
    nodes_uid[i] = i+10;
  
  modifier->addNodes(nodes_uid);
  
  info() << "===================== THE NODES ARE ADDED";

  Int64UniqueArray cells_infos(4*4);
  Integer j = 0;
  for(Integer i = 0; i < 4; ++i) {
    cells_infos[j++] = IT_CellLine2;
    cells_infos[j++] = i+5;
    cells_infos[j++] = i+10;
    cells_infos[j++] = i+11;
  }
  
  IntegerUniqueArray cells_lid;
  modifier->addCells(4, cells_infos, cells_lid);

  info() << "===================== THE CELLS ARE ADDED";

  modifier->endUpdate();

  ENUMERATE_CELL(icell,allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type=" 
	   << icell->type() << ", nb nodes=" << icell->nbNode();
    //lids[icell.index()] = icell->localId();
  }
 
  info() << "===================== DELETING OF ONLY ONE CELL";

  lids.resize(1);
  lids[0] = 1;
  modifier->removeCells(lids);

  modifier->endUpdate();
 
  ENUMERATE_CELL(icell,allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type=" 
	   << icell->type() << ", nb nodes=" << icell->nbNode();
  }

  info() << "===================== ADDING ONLY ONE CELL";

  cells_infos.resize(4);
  cells_infos[0] = IT_CellLine2;
  cells_infos[1] = 16;
  cells_infos[2] = 11;
  cells_infos[3] = 12;

  cells_lid.resize(0);
  modifier->addCells(1, cells_infos, cells_lid);

  modifier->endUpdate();

  ENUMERATE_CELL(icell,allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type=" 
	   << icell->type() << ", nb nodes=" << icell->nbNode();
  }

  info() << "===================== DELETE THEN ADD ONE CELL ONLY";
 
  lids.resize(1);
  lids[0] = 1;

  CellGroup group = mesh()->cellFamily()->createGroup("Test");
  group.addItems(lids);
 
  ENUMERATE_CELL(icell,group) {
    info() << "before add/remove cell[" << icell->localId() << "," << icell->uniqueId() << "] in group " << group.name();
  }

  modifier->removeCells(lids);
  
  cells_infos.resize(4);
  cells_infos[0] = IT_CellLine2;
  cells_infos[1] = 17;
  cells_infos[2] = 12;
  cells_infos[3] = 13;

  cells_lid.resize(0);
  modifier->addCells(1, cells_infos, cells_lid);

  modifier->endUpdate();
  
  info() << "after add/remove cell[" << "group " << group.name() << " size = " << group.size();
  ENUMERATE_CELL(icell,group) {
    info() << "after add/remove cell[" << icell->localId() << "," << icell->uniqueId() << "] in group " << group.name();
  }

  if(group.size() != 0)
    fatal() << "Error after in mesh update, group is not empty...";

  ENUMERATE_CELL(icell,allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type=" 
	   << icell->type() << ", nb nodes=" << icell->nbNode();
  }

  // Detach cells
  Int32UniqueArray detached_cells;
  ENUMERATE_CELL(icell,allCells()) {
    detached_cells.add(icell->localId());
    }
  modifier->detachCells(detached_cells);
  modifier->removeDetachedCells(detached_cells);
  modifier->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Test cloned from MeshModification.cs for easier debug
void MeshModificationTester::
_refineCells()
{
    Int64 max_node_uid = _searchMaxUniqueId(mesh()->allNodes());
    Int64 max_cell_uid = _searchMaxUniqueId(mesh()->allCells());
    info() << String::format("MAX UID NODE={0} CELL={1}",max_node_uid,max_cell_uid);

    CellGroup all_cells = mesh()->allCells();
    int index = 0;
    Integer nb_cell_to_add = 0;
    UniqueArray<Int32> cells_to_detach;
    UniqueArray<Int64> to_add_cells;
    UniqueArray<Int64> nodes_to_add;
    UniqueArray<Real3> nodes_to_add_coords;
    VariableNodeReal3& nodes_coords = mesh()->nodesCoordinates();
    ENUMERATE_CELL(cell, all_cells) {
      //TODO: tester si la maille est un hexaedre
      //if (c.
      ++index;
      if ((index % 3) == 0){
        cells_to_detach.add(cell.localId());

        to_add_cells.add(8); // Pour une pyramide
        //to_add_cells.Add(max_cell_uid + index); // Pour le uid
        to_add_cells.add(cell->uniqueId().asInt64() + max_cell_uid); // Pour le uid, reutilise celui de la maille supprimÃ©e
        //to_add_cells.Add(c.UniqueId); // Pour le uid, reutilise celui de la maille supprimÃ©e
        to_add_cells.add(cell->node(0).uniqueId().asInt64());
        to_add_cells.add(cell->node(1).uniqueId().asInt64());
        to_add_cells.add(cell->node(2).uniqueId().asInt64());
        to_add_cells.add(cell->node(3).uniqueId().asInt64());
        Real3 center;
        for( NodeLocalId inode : cell->nodeIds()) {
          center += nodes_coords[inode];
          info() << String::format("ADD CENTER {0}",nodes_coords[inode]);
        }
        center /= (Real)cell->nbNode();
        Int64 node_uid = max_node_uid + index;
        Int64 cell_uid = max_cell_uid + index;
        nodes_to_add.add(node_uid);
        nodes_to_add_coords.add(center);
        to_add_cells.add(node_uid);
        info() << String::format("WANT ADD NODE UID={0} COORD={1} CELL_UID={2}",node_uid,center,cell_uid);
        ++nb_cell_to_add;
      }
    }

    IMeshModifier* modifier = mesh()->modifier();
    Integer nb_node_added = nodes_to_add.size();
    UniqueArray<Int32> new_nodes_local_id(nb_node_added);

    modifier->addNodes(nodes_to_add.constView(),new_nodes_local_id.view());
    mesh()->nodeFamily()->endUpdate();
    info() << String::format("NODES ADDED = {0}",nb_node_added);
    NodeInfoListView new_nodes(mesh()->nodeFamily());
    for(int i=0; i<nb_node_added; ++i){
      Int32 new_local_id = new_nodes_local_id[i];
      Item new_node = new_nodes[new_local_id];
      info() << String::format("NEW LOCAL ID={0} Coord={1} UID={2}",new_local_id, nodes_to_add_coords[i],new_node.uniqueId());
      nodes_coords[new_nodes[new_local_id]] = nodes_to_add_coords[i];
    }

    info() << String::format("NB CELL TO ADD = {0}",nb_cell_to_add);
    warning() << "TEST WARNING";
    Int64ConstArrayView uid_view = to_add_cells.constView();
    for( Integer i=0; i<uid_view.size(); ++i ){
      info() << " ";
      info() << uid_view[i];
    }
    info() << ".";

    // Avant d'ajouter les nouvelles mailles, il faut dÃ©tacher les anciennes
    modifier->detachCells(cells_to_detach.constView());
    modifier->addCells(nb_cell_to_add,to_add_cells.constView());
    modifier->removeDetachedCells(cells_to_detach.constView());
    modifier->endUpdate();
  }
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


Int64 MeshModificationTester::
_searchMaxUniqueId(ItemGroup group)
  {
  Int64 max_uid = 0;
  ENUMERATE_ITEM(item, group) {
    //Console.WriteLine("IT IS MY TEST! {0} {1} ",i.UniqueId,MyClass.NB);
    Int64 uid = item->uniqueId();
    //Console.WriteLine("IT IS MY TEST! {0} {1}",uid,MyClass.NB);
    max_uid = (uid>max_uid) ? uid : max_uid;
  }
  return max_uid;
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
