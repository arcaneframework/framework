// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMergeNodesUnitTest.cc                                   (C) 2000-2016 */
/*                                                                           */
/* Service de test de la fusion des noeuds.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IParallelMng.h"
#include "arcane/MeshStats.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MeshMergeNodesUnitTest_axl.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la fusion des noeuds.
 */
class MeshMergeNodesUnitTest
: public ArcaneMeshMergeNodesUnitTestObject
{
 public:
  class NodePair
  {
   public:
    NodePair() : left_uid(NULL_ITEM_UNIQUE_ID), right_uid(NULL_ITEM_UNIQUE_ID){}
    NodePair(Int64 left,Int64 right) : left_uid(left), right_uid(right){}
    Int64 left_uid;
    Int64 right_uid;
  };
 public:

  MeshMergeNodesUnitTest(const ServiceBuildInfo& cb);
  ~MeshMergeNodesUnitTest();

 public:

  virtual void initializeTest();
  virtual void executeTest();

 private:

  typedef std::map<Real,NodePair> NodePairMap;
  NodePairMap m_node_pair_map;

 private:

  void _fillNodePairMap();
  void _addNodePairDirect(Real y,Int64 left_uid,Int64 right_uid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHMERGENODESUNITTEST(MeshMergeNodesUnitTest,
                                               MeshMergeNodesUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMergeNodesUnitTest::
MeshMergeNodesUnitTest(const ServiceBuildInfo& mb)
: ArcaneMeshMergeNodesUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMergeNodesUnitTest::
~MeshMergeNodesUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
executeTest()
{
  Int32UniqueArray nodes_local_id;
  Int32UniqueArray nodes_to_merge_local_id;

  IItemFamily* family = mesh()->nodeFamily();

  for( auto iter : m_node_pair_map ){
    ItemInternal* left_item = family->findOneItem(iter.second.left_uid);
    ItemInternal* right_item = family->findOneItem(iter.second.right_uid);
    if (!left_item && !right_item)
      break;
    if (!left_item || !right_item)
      ARCANE_FATAL("Invalid NodePair (right or left item is null)");
    nodes_local_id.add(left_item->localId());
    nodes_to_merge_local_id.add(right_item->localId());
  }

  mesh()->utilities()->mergeNodes(nodes_local_id,nodes_to_merge_local_id);
  MeshStats ms(traceMng(),mesh(),mesh()->parallelMng());
  ms.dumpStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
_fillNodePairMap()
{
  // Pour détecter les noeuds à connecter, on suppose que le maillage
  // a les propriétés suivantes:
  // - les noeuds à fusionner sont sur deux verticales, l'une correspond à
  // un X=0.5 et l'autre à X=0.51. Les noeuds de la seconde seront fusionnés
  // avec ceux de la première.
  VariableNodeReal3& coords = mesh()->nodesCoordinates();
  ENUMERATE_NODE(inode,allNodes()){
    Node node = *inode;
    Real3 pos = coords[inode];
    Real ypos = pos.y;
    if (pos.x==0.5){
      info() << "Node Left =" << ItemPrinter(node) << " pos=" << pos.y;
      m_node_pair_map[ypos].left_uid = node.uniqueId();
    }
    if (pos.x==0.51){
      info() << "Node Right=" << ItemPrinter(node) << " pos=" << pos.y;
      m_node_pair_map[ypos].right_uid = node.uniqueId();
    }
  }
  for( auto iter : m_node_pair_map ){
    Real ypos = iter.first;
    Int64 left_uid = iter.second.left_uid;
    Int64 right_uid = iter.second.right_uid;
    if (left_uid==NULL_ITEM_UNIQUE_ID)
      ARCANE_FATAL("No 'right' item for ypos={0} left={1}",ypos,left_uid);
    if (right_uid==NULL_ITEM_UNIQUE_ID)
      ARCANE_FATAL("No 'left' item for ypos={0} right={1}",ypos,right_uid);
    info() << "UID y=" << ypos << " Left=" << left_uid
           << " Right=" << right_uid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
_addNodePairDirect(Real y,Int64 left_uid,Int64 right_uid)
{
  m_node_pair_map.insert(std::make_pair(y,NodePair(left_uid,right_uid)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMergeNodesUnitTest::
initializeTest()
{
  if (mesh()->parallelMng()->isParallel()){
    // En parallèle, comme on ne peut pas déterminer la liste des noeuds
    // connectés car ils ne sont pas forcément dans le même sous-domaine
    // on remplit directement les valeurs. Du coup ce test ne fonctionne
    // qu'avec un maillage spécifique.
    _addNodePairDirect(-0.5,6,3);
    _addNodePairDirect(-0.4,62,35);
    _addNodePairDirect(-0.3,63,36);
    _addNodePairDirect(-0.2,64,37);
    _addNodePairDirect(-0.1,65,38);
    _addNodePairDirect(0.0,66,39);
    _addNodePairDirect(0.1,67,40);
    _addNodePairDirect(0.2,68,41);
    _addNodePairDirect(0.3,69,42);
    _addNodePairDirect(0.4,70,43);
    _addNodePairDirect(0.5,7,0);

  }
  else
    _fillNodePairMap();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
