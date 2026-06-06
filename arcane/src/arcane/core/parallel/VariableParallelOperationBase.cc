// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableParallelOperationBase.cc                            (C) 2000-2024 */
/*                                                                           */
/* Base class for parallel operations on variables.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/VariableParallelOperationBase.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ParallelMngUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

namespace
{
  const Int64 SERIALIZE_MAGIC_NUMBER = 0x4cf92789;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableParallelOperationBase::
VariableParallelOperationBase(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_item_family(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableParallelOperationBase::
setItemFamily(IItemFamily* family)
{
  if (m_item_family)
    ARCANE_FATAL("family already set");
  m_item_family = family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* VariableParallelOperationBase::
itemFamily()
{
  return m_item_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableParallelOperationBase::
addVariable(IVariable* variable)
{
  if (!m_item_family)
    ARCANE_FATAL("family not set. call setItemFamily()");
  if (variable->itemGroup().itemFamily()!=m_item_family)
    ARCANE_FATAL("variable->itemFamily() and itemFamily() differ");
  m_variables.add(variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableParallelOperationBase::
applyOperation(IDataOperation* operation)
{
  if (m_variables.empty())
    return;

#ifdef ARCANE_DEBUG
  const bool is_debug_print = true;
#else
  const bool is_debug_print = false;
#endif
  IParallelMng* pm = m_parallel_mng;
  Integer nb_rank = pm->commSize();
  m_items_to_send.clear();
  m_items_to_send.resize(nb_rank);
  _buildItemsToSend();

  UniqueArray<ISerializeMessage*> m_messages;
  m_messages.reserve(nb_rank);

  auto exchanger {ParallelMngUtils::createExchangerRef(pm)};
  
  for( Integer i=0; i<nb_rank; ++i )
    if (!m_items_to_send[i].empty())
      exchanger->addSender(i);
  
  bool no_exchange = exchanger->initializeCommunicationsMessages();
  if (no_exchange)
    return;  

  ItemInfoListView item_list(m_item_family);
  // Generates info for each processor to which we will send entities
  for( Integer i=0, is=exchanger->nbSender(); i<is; ++i ){
    ISerializeMessage* comm = exchanger->messageToSend(i);
    Int32 dest_sub_domain = comm->destination().value();
    ConstArrayView<ItemLocalId> dest_items_internal = m_items_to_send[dest_sub_domain];

    Integer nb_item = dest_items_internal.size();
    debug() << "Number of items to serialize: " << nb_item << " subdomain=" << dest_sub_domain;

    UniqueArray<Int32> dest_items_local_id(nb_item);
    UniqueArray<Int64> dest_items_unique_id(nb_item);
    for( Integer z=0; z<nb_item; ++z ){
      Item item = item_list[dest_items_internal[z]];
      dest_items_local_id[z] = item.localId();
      dest_items_unique_id[z] = item.uniqueId().asInt64();
    }
    ISerializer* sbuf = comm->serializer();

    // Reserves memory for serialization
    sbuf->setMode(ISerializer::ModeReserve);

    // Reserves for the magic number
    sbuf->reserveInt64(1);

    // Reserves for the list of uniqueId() of transferred entities
    sbuf->reserveInt64(1);
    sbuf->reserveSpan(dest_items_unique_id);

    // Reserves for each variable
    for( VariableList::Enumerator i_var(m_variables); ++i_var; ){
      IVariable* var = *i_var;
      debug(Trace::High) << "Serialize variable (reserve)" << var->name();
      var->serialize(sbuf,dest_items_local_id);
    }

    sbuf->allocateBuffer();

    // Serializes the info
    sbuf->setMode(ISerializer::ModePut);

    // Serializes the magic number
    sbuf->putInt64(SERIALIZE_MAGIC_NUMBER);

    // Serializes the list of uniqueId() of transferred entities
    sbuf->putInt64(nb_item);
    sbuf->putSpan(dest_items_unique_id);
    for( VariableList::Enumerator i_var(m_variables); ++i_var; ){
      IVariable* var = *i_var;
      debug(Trace::High) << "Serialise variable (put)" << var->name();
      var->serialize(sbuf,dest_items_local_id);
    }
  }

  exchanger->processExchange();

  {
    debug() << "VariableParallelOperationBase::readVariables()";
    
    UniqueArray<Int64> items_unique_id;
    UniqueArray<Int32> items_local_id;
    
    // Retrieves info for variables and fills them
    for( Integer i=0, n=exchanger->nbReceiver(); i<n; ++i ){
      ISerializeMessage* comm = exchanger->messageToReceive(i);
      ISerializer* sbuf = comm->serializer();

      // Deserializes the variables
      {
        // Serializes the magic number
        Int64 magic_number = sbuf->getInt64();
        if (magic_number!=SERIALIZE_MAGIC_NUMBER)
          ARCANE_FATAL("Bad magic number actual={0} expected={1}. This is probably due to incoherent messaging",
                       magic_number,SERIALIZE_MAGIC_NUMBER);

        // Retrieves the list of uniqueId() of transferred entities
        Int64 nb_item = sbuf->getInt64();
        items_unique_id.resize(nb_item);
        sbuf->getSpan(items_unique_id);
        items_local_id.resize(nb_item);
        debug(Trace::High) << "Receiving " << nb_item << " items from " << comm->destination().value();

        if (is_debug_print){
          for( Integer iz=0; iz<nb_item; ++iz )
            debug(Trace::Highest) << "Receiving uid=" << items_unique_id[iz];
        }

        itemFamily()->itemsUniqueIdToLocalId(items_local_id,items_unique_id);
        
        for( VariableList::Enumerator ivar(m_variables); ++ivar; ){
          IVariable* var = *ivar;
          var->serialize(sbuf,items_local_id,operation);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
