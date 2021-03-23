// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostItemsManager.cc                                   (C) 2000-2015 */
/*                                                                           */
/* Construction des items fantômes supplémentaires.                          */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <set>

#include "ExtraGhostItemsManager.h"

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/ISerializeMessage.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
Arcane::mesh::ExtraGhostItemsManager::
computeExtraGhostItems()
{
  const Integer nb_builder = m_builders.size();


  if(nb_builder == 0) return;

  m_trace_mng->info() << "Compute extra ghost cells";

  for(Integer i=0; i<nb_builder; ++i) {
    // Calcul de mailles extraordinaires à envoyer
    m_builders[i]->computeExtraItemsToSend();
  }

  IParallelMng* pm = m_extra_ghost_items_adder->subDomain()->parallelMng();

  ScopedPtrT<IParallelExchanger> exchanger(pm->createExchanger());

  const Integer nsd = m_extra_ghost_items_adder->subDomain()->nbSubDomain();

  // Construction des items à envoyer // Voir comment rendre compatible avec le ExtraGhostBuilder
  UniqueArray< Arcane::SharedArray<Int32> >  item_to_send(nsd);

  // Initialisation de l'échangeur de données
  for(Integer isd=0;isd<nsd;++isd)
    {
      for(Integer i=0; i<nb_builder; ++i)
        {
          item_to_send[isd].addRange(m_builders[i]->extraItemsToSend(isd));
        }
    if (!item_to_send[isd].empty())
      exchanger->addSender(isd);
    }

  exchanger->initializeCommunicationsMessages();

  // Envoi des item
  for(Integer i=0, ns=exchanger->nbSender(); i<ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToSend(i);
    const Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    const Arcane::Int32Array& items_to_send_to_rank = item_to_send[rank];
    m_extra_ghost_items_adder->serializeGhostItems(s, items_to_send_to_rank);
  }
  exchanger->processExchange();

  // Réception des mailles
  for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i ) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    m_extra_ghost_items_adder->addExtraGhostItems(s);
  }
  m_extra_ghost_items_adder->updateSynchronizationInfo(); // Collective Operation

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

