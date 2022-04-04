// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostCellsBuilder.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Construction des mailles fantômes supplémentaires.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/IExtraGhostCellsBuilder.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/ParallelMngUtils.h"

#include "arcane/mesh/ExtraGhostCellsBuilder.h"
#include "arcane/mesh/DynamicMesh.h"

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
addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder)
{
  if (m_builders.find(builder)!=m_builders.end())
    ARCANE_FATAL("Instance {0} is already registered",builder);
  m_builders.insert(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostCellsBuilder::
removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder)
{
  auto x = m_builders.find(builder);
  if (x==m_builders.end())
    ARCANE_FATAL("Instance {0} is not registered",builder);
  m_builders.erase(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ExtraGhostCellsBuilder::
hasBuilder() const
{
  return !m_builders.empty();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostCellsBuilder::
computeExtraGhostCells()
{
  if (m_builders.empty())
    return;
  
  info() << "Compute extra ghost cells";
  
  for( IExtraGhostCellsBuilder* v : m_builders ){
    // Calcul de mailles extraordinaires à envoyer
    v->computeExtraCellsToSend();
  }
  
  IParallelMng* pm = m_mesh->parallelMng();
  
  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };
  
  const Int32 nsd = pm->commSize();
  
  // Construction des items à envoyer
  UniqueArray<std::set<Integer> > to_sends(nsd);
  
  // Initialisation de l'échangeur de données
  for(Integer isd=0;isd<nsd;++isd) {
    std::set<Integer>& cell_set = to_sends[isd];
    for( IExtraGhostCellsBuilder* builder : m_builders ){
      Int32ConstArrayView extra_cells = builder->extraCellsToSend(isd);
      // On trie les lids à envoyer pour éviter les doublons
      for(Integer j=0, size=extra_cells.size(); j<size; ++j)
        cell_set.insert(extra_cells[j]);
    }
    if (!cell_set.empty())
      exchanger->addSender(isd);
  }
  exchanger->initializeCommunicationsMessages();
  
  // Envoi des mailles
  for (Integer i=0, ns=exchanger->nbSender(); i<ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToSend(i);
    const Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    const std::set<Integer>& cell_set = to_sends[rank];
    Int32UniqueArray items_to_send(cell_set.size());
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
