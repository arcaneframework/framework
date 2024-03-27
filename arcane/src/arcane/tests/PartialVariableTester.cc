// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PartialVariableTester.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Service de test des variables partielles.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicTimeLoopService.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemVector.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/ItemPrinter.h"
#include "arcane/tests/PartialVariableTester_axl.h"

#include "arcane/random/Uniform01.h"
#include "arcane/random/LinearCongruential.h"

#include "arcane/utils/SharedPtr.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test des variables partielles.
 */
class PartialVariableTester
: public ArcanePartialVariableTesterObject
{
 public:

  explicit PartialVariableTester(const ModuleBuildInfo& mbi);
  ~PartialVariableTester() {}

 public:

  void init();
  void compute();
  
 private:

  // test historique
  VariableCellReal m_cell_temperature;
  VariableNodeReal m_node_temperature;
  PartialVariableCellReal m_high_cell_temperature;

  // test pour l'équilibrage
  VariableCellInteger m_current_rank;
  VariableCellInteger m_initial_rank;
  PartialVariableCellInteger m_partial_initial_rank;
  PartialVariableCellArrayInteger m_partial_initial_and_current_rank;
  VariableCellInteger m_post_initial_rank;
  VariableCellInteger m_post_current_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PartialVariableTester::
PartialVariableTester(const ModuleBuildInfo& mbi)
: ArcanePartialVariableTesterObject(mbi)
, m_cell_temperature(VariableBuildInfo(mbi.meshHandle(),"CellTemperature"))
, m_node_temperature(VariableBuildInfo(mbi.meshHandle(),"NodeTemperature"))
, m_high_cell_temperature(VariableBuildInfo(mbi.meshHandle(),"CellHighTemperature",String(),"HIGH"))
, m_current_rank(VariableBuildInfo(mbi.meshHandle(),"Rank"))
, m_initial_rank(VariableBuildInfo(mbi.meshHandle(),"InitialRank"))
, m_partial_initial_rank(VariableBuildInfo(mbi.meshHandle(),"PartialInitialRank",String(),"RankGroup"))
, m_partial_initial_and_current_rank(VariableBuildInfo(mbi.meshHandle(),"PartialInitialAndCurrentRank",String(),"RankGroup"))
, m_post_initial_rank(VariableBuildInfo(mbi.meshHandle(),"PostInitialRank"))
, m_post_current_rank(VariableBuildInfo(mbi.meshHandle(),"PostCurrentRank"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PartialVariableTester::
init()
{
  m_global_deltat = 0.1;
  
  // test historique
  
  IItemFamily* cell_family = mesh()->cellFamily();
  //CellGroup high_group = cell_family->createGroup("HIGH");
  //m_high_cell_temperature.setItemGroup(high_group);

  CellGroup zd_group = cell_family->findGroup("ZD");
  if (zd_group.null())
    fatal() << "Can not find group 'ZD'";

  m_cell_temperature.fill(1.0);
  ENUMERATE_CELL(icell,zd_group){
    m_cell_temperature[icell] = 5.0;
  }
  m_cell_temperature.synchronize();
  
  
  ENUMERATE_CELL(icell,m_high_cell_temperature.itemGroup()){
    m_high_cell_temperature[icell] = m_cell_temperature[icell];
  }
  m_high_cell_temperature.synchronize();

  // test pour l'équilibrage

  const Integer rank = subDomain()->parallelMng()->commRank();
  
  m_current_rank.fill(rank);
  m_initial_rank.fill(rank);
  m_initial_rank.synchronize();

  ItemVectorT<Cell> new_cells(cell_family);
  ENUMERATE_CELL(icell,allCells()){
    if(Int64(icell->uniqueId()) % 2)
      new_cells.addItem(*icell);
  }
  {
    CellGroup group = cell_family->findGroup("RankGroup");
    if (m_partial_initial_rank.itemGroup()!=group)
      fatal() << "Bad group for m_partial_initial_rank="
              << m_partial_initial_rank.itemGroup().name();
    if (m_partial_initial_and_current_rank.itemGroup()!=group)
      fatal() << "Bad group for m_partial_initial_and_current_rank="
              << m_partial_initial_rank.itemGroup().name();
    
    group.addItems(new_cells.viewAsArray());
    // Mise à jour du synchronizer car modif manuelle
    group.synchronizer()->compute();
  }
  
  // Marche pas encore
  // {
  //   if (m_all_cells_uid.itemGroup() != allCells())
  //     fatal() << "Bad group for m_all_cells_rank="
  //             << m_all_cells_uid.itemGroup().name();
  // }
  // ENUMERATE_CELL(icell,allCells()){
  //   if(icell->isOwn())
  //     m_all_cells_uid[icell] = icell->uniqueId();
  //   else
  //     m_all_cells_uid[icell] = -1;
  // }
  // m_all_cells_uid.synchronize();
  // ENUMERATE_CELL(icell,allCells()) {
  //   if(m_all_cells_uid[icell] != icell->uniqueId())
  //     fatal() << "proc " << rank << " : error cell(" << icell->localId() 
  //             << "," << icell->uniqueId() << ") uid " << m_all_cells_uid[icell]
  //             << " from partial variable on all-cells group";
  // }
  
  ENUMERATE_CELL(icell,m_partial_initial_rank.itemGroup()){
    CellEnumeratorIndex iter_index(icell.index());
    Int32 wanted_value = -1;
    if(icell->isOwn()){
      wanted_value = rank;
      m_partial_initial_rank[*icell] = rank;
    }
    else{
      m_partial_initial_rank[icell] = -1;
    }
    if (m_partial_initial_rank[iter_index]!=wanted_value)
      ARCANE_FATAL("Bad value i={0}  v={1} expected={2}",icell.index(),m_partial_initial_rank[iter_index],wanted_value);
  }
  m_partial_initial_rank.synchronize();
 
  m_partial_initial_and_current_rank.resize(2);
  ENUMERATE_CELL(icell,m_partial_initial_and_current_rank.itemGroup()){
    CellEnumeratorIndex iter_index(icell.index());
    if(icell->isOwn()) {
      m_partial_initial_and_current_rank[*icell][0] = rank;
      m_partial_initial_and_current_rank[iter_index][1] = 0;
    } else {
      m_partial_initial_and_current_rank[icell][0] = -1;
    }
  }
  m_partial_initial_and_current_rank.synchronize();
  
  ENUMERATE_CELL(icell,m_partial_initial_rank.itemGroup()){
    if(m_initial_rank[icell] != m_partial_initial_rank[icell])
      fatal() << "proc " << rank << " : error cell(" << icell->localId() 
              << "," << icell->uniqueId() << ") rank " << m_initial_rank[icell] 
              << " from variable not equals rank " << m_partial_initial_rank[icell]
              << " from partial variable";
  }
  ENUMERATE_CELL(icell,m_partial_initial_and_current_rank.itemGroup()){
    if(m_initial_rank[icell] != m_partial_initial_and_current_rank[icell][0])
      fatal() << "proc " << rank << " : error cell(" << icell->localId() 
              << "," << icell->uniqueId() << ") initial rank " << m_initial_rank[icell] 
              << " from variable not equals rank " << m_partial_initial_and_current_rank[icell][0]
              << " from partial variable";
    if(m_partial_initial_and_current_rank[icell][1] != 0)
      fatal() << "proc " << rank << " : error cell(" << icell->localId() 
              << "," << icell->uniqueId() << ") default value 0"
              << " from variable not equals rank " << m_partial_initial_and_current_rank[icell][0]
              << " from partial variable";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PartialVariableTester::
compute()
{
  if (m_global_iteration()>options()->maxIteration())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
  
  // test historique
  
  IItemFamily* cell_family = mesh()->cellFamily();
  CellGroup high_group = cell_family->findGroup("HIGH");
  if (m_high_cell_temperature.itemGroup()!=high_group)
    ARCANE_FATAL("Bad group for m_high_cell_temperature group={0}",
                 m_high_cell_temperature.itemGroup().name());

  // Vérifie que les valeurs partielles sont les même que les valeurs globales.
  // Ce test permet de vérifier après un équilibrage que les valeurs sont correctes
  ENUMERATE_CELL(icell,high_group){
    Cell cell = *icell;
    Real partial = m_high_cell_temperature[icell];
    Real global =  m_cell_temperature[icell];
    if (partial!=global)
      ARCANE_FATAL("Bad partial temperature cell={0} partial={1} global={2}",
                   ItemPrinter(cell),partial,global);
  }

  {
    Int32UniqueArray cells_local_id;
    ENUMERATE_CELL(icell,allCells()){
      if (m_cell_temperature[icell]>1.0 && m_cell_temperature[icell]<3.0)
        cells_local_id.add(icell.itemLocalId());
    }
    high_group.setItems(cells_local_id);
    // Mise à jour du synchronizer car modif manuelle
    m_high_cell_temperature.itemGroup().synchronizer()->compute();
  }

  ENUMERATE_NODE(inode,allNodes()){
    Real t = 0.0;
    Node node = *inode;
    for( CellLocalId icell : node.cellIds() )
      t += m_cell_temperature[icell];
    m_node_temperature[inode] = t / (Real)node.nbCell();
  }
  m_node_temperature.synchronize();

  ENUMERATE_CELL(icell,allCells()){
    Real t = 0.0;
    const Cell& cell = *icell;
    for( NodeLocalId inode : cell.nodeIds() )
      t += m_node_temperature[inode];
    m_cell_temperature[icell] = t / (Real)cell.nbNode();
  }
  m_cell_temperature.synchronize();

  ENUMERATE_CELL(icell,m_high_cell_temperature.itemGroup()){
    m_high_cell_temperature[icell] = m_cell_temperature[icell];
  }
  m_high_cell_temperature.synchronize();

  if (m_high_cell_temperature.checkIfSync(10) != 0)
    ARCANE_FATAL("Not synchronized variable");

  // test pour l'équilibrage

  const Integer rank = subDomain()->parallelMng()->commRank();
  
  m_current_rank.fill(rank);
  m_current_rank.synchronize();
  
  ENUMERATE_CELL(icell,m_partial_initial_and_current_rank.itemGroup()){
    if(icell->isOwn())
      m_partial_initial_and_current_rank[*icell][1] = rank;
    else
      m_partial_initial_and_current_rank[icell][1] = -1;
  }
  m_partial_initial_and_current_rank.synchronize();
  
  ENUMERATE_CELL(icell,m_partial_initial_rank.itemGroup()){
    if(m_initial_rank[icell] != m_partial_initial_rank[icell])
      fatal() << "proc " << rank << " : error cell(" << icell->localId() 
              << "," << icell->uniqueId() << ") initial rank " << m_initial_rank[icell] 
              << " from variable not equals rank " << m_partial_initial_rank[icell]
              << " from partial variable";
  }
  m_post_current_rank.fill(-1);
  m_post_initial_rank.fill(-1);
  ENUMERATE_CELL(icell,m_partial_initial_and_current_rank.itemGroup()){
    m_post_initial_rank[icell] = m_partial_initial_and_current_rank[icell][0]; 
    if(m_initial_rank[icell] != m_partial_initial_and_current_rank[icell][0])
      fatal() << "proc " << rank << " : error cell(" << icell->localId() 
              << "," << icell->uniqueId() << ") initial rank " << m_initial_rank[icell] 
              << " from variable not equals rank " << m_partial_initial_and_current_rank[icell][0]
              << " from partial variable";
    m_post_current_rank[icell] = m_partial_initial_and_current_rank[icell][1]; 
    if(m_current_rank[icell] != m_partial_initial_and_current_rank[icell][1])
      fatal() << "proc " << rank << " : error cell(" << icell->localId() 
              << "," << icell->uniqueId() << ") current rank " << m_current_rank[icell] 
              << " from variable not equals rank " << m_partial_initial_and_current_rank[icell][1]
              << " from partial variable";
  }

  // Ne marche pas 
  // ENUMERATE_CELL(icell,allCells()) {
  //   if(m_all_cells_uid[icell] != icell->uniqueId())
  //     fatal() << "proc " << rank << " : error cell(" << icell->localId() 
  //             << "," << icell->uniqueId() << ") uid " << m_all_cells_uid[icell]
  //             << " from partial variable on all-cells group";
  // }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_PARTIALVARIABLETESTER(PartialVariableTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
