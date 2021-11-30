// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostCellsBuilder.cc                                   (C) 2011-2021 */
/*                                                                           */
/* Construction des mailles fantômes supplémentaires.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/IExtraGhostCellsBuilder.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/ParallelMngUtils.h"

#include "arcane/mesh/ExtraGhostCellsBuilder.h"
#include "arcane/mesh/DynamicMesh.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExtraGhostCellsBuilder::
ExtraGhostCellsBuilder(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostCellsBuilder::
computeExtraGhostCells()
{
  const Integer nb_builder = m_builders.size();
  
  if(nb_builder == 0) return;
  
  info() << "Compute extra ghost cells";
  
  for(Integer i=0; i<nb_builder; ++i) {
    // Calcul de mailles extraordinaires à envoyer
    m_builders[i]->computeExtraCellsToSend();
  }
  
  IParallelMng* pm = m_mesh->subDomain()->parallelMng();
  
  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };
  
  const Integer nsd = m_mesh->subDomain()->nbSubDomain();
  
  // Construction des items à envoyer
  UniqueArray<std::set<Integer> > to_sends(nsd);
  
  // Initialisation de l'échangeur de données
  for(Integer isd=0;isd<nsd;++isd) {
    std::set<Integer>& cell_set = to_sends[isd];
    for(Integer i=0; i<nb_builder; ++i) {
      Int32ConstArrayView extra_cells = m_builders[i]->extraCellsToSend(isd);
      // On trie les lids à envoyer pour éviter les doublons
      for(Integer j=0, size=extra_cells.size(); j<size; ++j)
        cell_set.insert(extra_cells[j]);
    }
    if (!cell_set.empty())
      exchanger->addSender(isd);
  }
  exchanger->initializeCommunicationsMessages();
  
  // Envoi des mailles
  for(Integer i=0, ns=exchanger->nbSender(); i<ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToSend(i);
    const Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    const std::set<Integer>& cell_set = to_sends[rank];
    Int32UniqueArray items_to_send(CheckedConvert::toInteger(cell_set.size()));
    std::copy(std::begin(cell_set),std::end(cell_set),std::begin(items_to_send));
    m_mesh->serializeCells(s, items_to_send);
  }
  exchanger->processExchange();
  
  // Réception des mailles
  for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i ) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    m_mesh->addCells(s);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
