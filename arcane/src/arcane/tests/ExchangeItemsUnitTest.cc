﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshModificationTester.cc                                   (C) 2000-2018 */
/*                                                                           */
/* Service du test de l'échange d'items.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/BasicUnitTest.h"

#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParallelMng.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IVariableSynchronizer.h"



//enum class TestOperation {
enum TestOperation {
  GatherBroadcastCells, ExchangeCellOwners

};

#include "arcane/tests/ExchangeItemsUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ExchangeItemsUnitTest
: public ArcaneExchangeItemsUnitTestObject
{

public:

  ExchangeItemsUnitTest(const ServiceBuildInfo& sb)
    : ArcaneExchangeItemsUnitTestObject(sb) {}
  
  ~ExchangeItemsUnitTest() {}

 public:

  virtual void initializeTest();
  virtual void executeTest();
  
 private:
  void _refineCells();
  void _partitionCells();//Unitary test for exchangeItems
  void _exchangeCellOwner();
  Int64 _searchMaxUniqueId(ItemGroup group);
  void _computeGhostPPVariable();

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_EXCHANGEITEMSUNITTEST(ExchangeItemsUnitTest,ExchangeItemsUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExchangeItemsUnitTest::
initializeTest()
{ 
  info() << "init test";
  ENUMERATE_CELL(icell, allCells()) {
    m_cell_uids[icell] = icell->uniqueId();
  }
  _computeGhostPPVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExchangeItemsUnitTest::
executeTest()
{
  info() << "execute test";
  
  switch (options()->testOperation()) {
    case TestOperation::GatherBroadcastCells:
      _partitionCells();
      break;
    case TestOperation::ExchangeCellOwners:
      _exchangeCellOwner();
      break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExchangeItemsUnitTest::
_partitionCells()
{
  // Change cells owner
  VariableItemInt32& cell_new_owners = mesh()->cellFamily()->itemsNewOwner();
  Integer comm_size = mesh()->parallelMng()->commSize();
  ENUMERATE_CELL(icell, allCells()) {
    cell_new_owners[icell] = 0; // everybody on subdomain 0
    info() << "Cell uid " << icell->uniqueId() << " has owner "  << icell->owner();
  }
  mesh()->utilities()->changeOwnersFromCells();
  mesh()->modifier()->setDynamic(true);
  mesh()->toPrimaryMesh()->exchangeItems();// update ghost is done.

  // une fois tout sur un proc on redispatche pour mimer un partitionement initial
  ENUMERATE_CELL(icell, allCells()) {
    cell_new_owners[icell] = (icell.index()*subDomain()->parallelMng()->commSize())/mesh()->cellFamily()->nbItem();
  }

  mesh()->utilities()->changeOwnersFromCells();
  mesh()->modifier()->setDynamic(true);
  mesh()->toPrimaryMesh()->exchangeItems();// update ghost is done.


  ENUMERATE_CELL(icell, allCells()) {
    info() << "Cell uid " << icell->uniqueId() << " has owner "  << icell->owner();
  }
  _computeGhostPPVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExchangeItemsUnitTest::
_exchangeCellOwner()
{
  // Change cells owner
    VariableItemInt32& cell_new_owners = mesh()->cellFamily()->itemsNewOwner();
    Integer comm_size = mesh()->parallelMng()->commSize();
    ENUMERATE_CELL(icell, allCells()) {
      cell_new_owners[icell] = comm_size-(icell->owner()+1); // exchange owner
      info() << "Cell uid " << icell->uniqueId() << " has owner "  << icell->owner();
    }
    mesh()->utilities()->changeOwnersFromCells();
    mesh()->modifier()->setDynamic(true);
    mesh()->toPrimaryMesh()->exchangeItems();// update ghost is done.

    _computeGhostPPVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExchangeItemsUnitTest::
_computeGhostPPVariable()
{
  // Post-processing ghost debug
  const Integer isd = subDomain()->subDomainId();
  const Integer nsd = subDomain()->nbSubDomain();
  m_ghostpp.resize(nsd);
  m_node_ghostpp.resize(nsd);
  m_face_ghostpp.resize(nsd);

  // Cell
  for(Integer i=0;i<nsd;++i)
    ENUMERATE_CELL(icell,ownCells())
      m_ghostpp[icell][i] = (isd == i)?1:0;
  IVariableSynchronizer * synchronizer = mesh()->cellFamily()->allItemsSynchronizer();

  Int32ConstArrayView ranks = synchronizer->communicatingRanks();
  for(Integer i=0;i<ranks.size();++i)
    {
      const Integer rank = ranks[i];
      CellVectorView ghost_items(mesh()->cellFamily()->itemsInternal(), synchronizer->sharedItems(i));
      ENUMERATE_CELL(icell, ghost_items)
        {
          m_ghostpp[icell][rank] = 2; // ou bien mettre un identifiant du propriï¿½taire
        }
    }
  m_ghostpp.synchronize();

  {
  // Node
  for(Integer i=0;i<nsd;++i)
    ENUMERATE_NODE(inode,ownNodes())
      m_node_ghostpp[inode][i] = (isd == i)?1:0;
  IVariableSynchronizer * node_synchronizer = mesh()->nodeFamily()->allItemsSynchronizer();

  Int32ConstArrayView ranks = node_synchronizer->communicatingRanks();
  for(Integer i=0;i<ranks.size();++i)
    {
      const Integer rank = ranks[i];
      NodeVectorView ghost_items(mesh()->nodeFamily()->itemsInternal(), node_synchronizer->sharedItems(i));
      ENUMERATE_NODE(inode, ghost_items)
        {
          m_node_ghostpp[inode][rank] = 2; // ou bien mettre un identifiant du propriétaire
        }
    }
  m_node_ghostpp.synchronize();

  }


//    for(Integer i=0;i<nsd;++i)
//     ENUMERATE_NODE(inode,allNodes())
//       info() << " nsd " << i << " node_ghostpp["<< inode->uniqueId()<< "]= " << m_node_ghostpp[inode][i] ;


  { // Face
    for(Integer i=0;i<nsd;++i)
    ENUMERATE_FACE(iface,ownFaces())
      m_face_ghostpp[iface][i] = (isd == i)?1:0;
  IVariableSynchronizer * face_synchronizer = mesh()->faceFamily()->allItemsSynchronizer();

  Int32ConstArrayView ranks = face_synchronizer->communicatingRanks();
  for(Integer i=0;i<ranks.size();++i)
    {
      const Integer rank = ranks[i];
      FaceVectorView ghost_items(mesh()->faceFamily()->itemsInternal(), face_synchronizer->sharedItems(i));
      ENUMERATE_FACE(iface, ghost_items)
        {
          m_face_ghostpp[iface][rank] = 2; // ou bien mettre un identifiant du propriétaire
        }
    }
  m_face_ghostpp.synchronize();

  }



}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
