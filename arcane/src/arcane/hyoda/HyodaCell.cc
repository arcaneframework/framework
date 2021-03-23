// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * Hyoda.cc                                                    (C) 2000-2012 *
 *                                                                           *
 * Service de debugger hybrid.                                               *
 *****************************************************************************/
#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/AbstractService.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/ITransferValuesParallelOperation.h"

#include "arcane/hyoda/HyodaArc.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/******************************************************************************
 * Fonction de remplissage de la zone de donnée dumpée par le debugger
 *****************************************************************************/
void Hyoda::fetch_and_fill_data_to_be_dumped(ISubDomain* sd, UniqueIdType target_cell_uid){
  
  debug()<<"[fetch_and_fill_data_to_be_dumped] Fill data that will be broadcasted back in hook";
  m_data->global_iteration=(Int64)sd->commonVariables().globalIteration();
  m_data->global_time=sd->commonVariables().globalTime();
  m_data->global_deltat=sd->commonVariables().globalDeltaT();
  m_data->global_cpu_time=sd->commonVariables().globalCPUTime();
  
  SharedVariableNodeReal3 nodes_coords = sd->defaultMesh()->sharedNodesCoordinates();
  ItemInternalList cells = sd->defaultMesh()->cellFamily()->itemsInternal();
  LocalIdType target_cell_lid=targetCellIdToLocalId(sd, target_cell_uid);
  
  Cell focused_cell;
  if (target_cell_lid==NULL_ITEM_ID){
    debug()<<"[Hyoda::fetch_and_fill_data_to_be_dumped] NULL_ITEM_ID";
    focused_cell=cells[0]; // On met n'importe quoi, ce ne sera pas utilisé (faut que cela soit possible)
  }else{
    debug()<<"[Hyoda::fetch_and_fill_data_to_be_dumped] lid "<<target_cell_lid<<" (uid="<<target_cell_uid<<")";
    focused_cell=cells[target_cell_lid];
  }

  // En séquentiel, on affiche directement, on est sûr de l'avoir
  int i=0;
  ENUMERATE_NODE(inode,focused_cell.nodes()){
    Real3 coord = nodes_coords[inode];
    m_data->coords[i][0] = coord.x;
    m_data->coords[i][1] = coord.y;
    m_data->coords[i][2] = coord.z;
    debug()<<"[Hyoda::fetch_and_fill_data_to_be_dumped] node ("
          << m_data->coords[i][0] << ","
          << m_data->coords[i][1] << ","
          << m_data->coords[i][2] << ")"
      ;
    i+=1;
  }
  
  if (!sd->parallelMng()->isParallel()) return;

  // En parallele, on va envoyer à celui où est accroché gdbserver
  
  if ((target_cell_lid!=NULL_ITEM_ID)                         // J'ai la maille
      && (sd->parallelMng()->commRank()==m_gdbserver_rank)){  // et le serveur => rien à faire d'autre
    debug()<<"[Hyoda::fetch_and_fill_data_to_be_dumped] CELL and SERVER";
    return;
  }
    
  if (target_cell_lid!=NULL_ITEM_ID){                 // J'ai la maille, pas le serveur = je SEND
    debug()<<"[Hyoda::fetch_and_fill_data_to_be_dumped] SENDing CELL";;
    Real3UniqueArray cell_nodes_coords(focused_cell.nbNode());
    int i=0;
    ENUMERATE_NODE(inode,focused_cell.nodes()){
      Real3 coord = nodes_coords[inode];
      cell_nodes_coords[i].x=coord.x;
      cell_nodes_coords[i].y=coord.y;
      cell_nodes_coords[i].z=coord.z;
      //debug()<<"\t[Hyoda::fetch_and_fill_data_to_be_dumped] Filling coord #"<<i;
      i+=1;
    }
    sd->parallelMng()->send(cell_nodes_coords.constView(),m_gdbserver_rank);
    return;
  }

  
  if (sd->parallelMng()->commRank()==m_gdbserver_rank){ // Si c'est moi qui ai gdbserver, je RECV
    debug()<<"[Hyoda::fetch_and_fill_data_to_be_dumped] SERVEUR: recv from"<<m_data->target_cell_rank;
    Real3UniqueArray cell_nodes_coords(m_data->target_cell_nb_nodes);
    sd->parallelMng()->recv(cell_nodes_coords,m_data->target_cell_rank);
    for(int i=0;i<m_data->target_cell_nb_nodes;i++){
      m_data->coords[i][0] = cell_nodes_coords[i].x;
      m_data->coords[i][1] = cell_nodes_coords[i].y;
      m_data->coords[i][2] = cell_nodes_coords[i].z;
    }
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

