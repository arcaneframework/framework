// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyVariableSerializer.cc                             (C) 2000-2024 */
/*                                                                           */
/* Manages the serialization/deserialization of variables for a family.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/Collection.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/ItemFamilySerializeArgs.h"
#include "arcane/core/IParallelMng.h"
// TODO: a supprimer
#include "arcane/core/IMesh.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ISubDomain.h"

#include "arcane/mesh/ItemFamilyVariableSerializer.h"

#include <set>

/*
 * NOTE:
 * Instead of one instance of this class managing the serialization of variables
 * for the parent family and all child families, an instance per family would be necessary.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemFamilyExchange;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

namespace
{
  const Integer VARIABLE_MAGIC_NUMBER = 0x3a9e4324;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamilyVariableSerializer::
ItemFamilyVariableSerializer(IItemFamily* family)
: TraceAccessor(family->traceMng())
, m_item_family(family)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamilyVariableSerializer::
~ItemFamilyVariableSerializer()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyVariableSerializer::
initialize()
{
  // Determines the list of variables to exchange
  // It also includes variables from child families
  // TODO: it would be necessary to retrieve the variables of each
  // family via IItemFamily::usedVariables(), but this method does not guarantee
  // alphabetical order of transmission. By thinking with an std::map, we
  // should be able to manage it.
  IVariableMng* vm = m_item_family->mesh()->variableMng();
  VariableCollection used_vars(vm->usedVariables());

  UniqueArray<IItemFamily*> family_to_exchange;
  family_to_exchange.add(m_item_family); // The current family
  IItemFamilyCollection child_families = m_item_family->childFamilies();
  for (IItemFamily* child_family : child_families)
    family_to_exchange.add(child_family);

  for (Integer i = 0; i < family_to_exchange.size(); ++i) {
    IItemFamily* current_family = family_to_exchange[i];
    info(4) << " Serializing family " << current_family->fullName();

    // TODO: retrieve the variables used by the family
    // via ItemFamily::usedVariables
    for (VariableCollection::Enumerator i_var(used_vars); ++i_var;) {
      IVariable* var = *i_var;
      // TODO: check if all variables need to be exchanged.
      // Should PNoDump variables be sent?
      // TODO: call the writeObservable of IVariableMng or make
      // a specific one for balancing
      // TODO: check that all sub-domains have the same values
      // for m_variables_to_exchange with the variables in the same order.
      bool no_exchange = (var->property() & IVariable::PNoExchange);
      if (no_exchange)
        continue;
      IItemFamily* var_family = var->itemFamily();
      if (var_family == current_family) {
        debug(Trace::High) << "Add variable " << var->fullName() << " to serialize";
        m_variables_to_exchange.add(var);
      }
    }
  }
  debug() << "Number of variables to serialize: " << m_variables_to_exchange.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Serializes a magic number and the number of entities to verify that
 * deserialization is correct.
 */
void ItemFamilyVariableSerializer::
_checkSerialization(ISerializer* sbuf, Int32ConstArrayView local_ids)
{
  switch (sbuf->mode()) {
  case ISerializer::ModeReserve:
    // Reserves for the uniqueId() list of transferred entities
    sbuf->reserveInteger(1); // Magic number for readVariables();
    sbuf->reserveInteger(1); // Number of serialized entities.
    break;
  case ISerializer::ModePut:
    // Serializes the number of transferred entities
    sbuf->put(VARIABLE_MAGIC_NUMBER);
    sbuf->put(local_ids.size());
    break;
  case ISerializer::ModeGet:
    // Checks for serialization errors.
    Integer magic_number = sbuf->getInteger();
    if (magic_number != VARIABLE_MAGIC_NUMBER)
      ARCANE_FATAL("Internal error: bad magic number expected={0} found={1}",
                   VARIABLE_MAGIC_NUMBER, magic_number);

    // Retrieves the number of transferred entities for verification
    Integer nb_item = sbuf->getInteger();
    if (local_ids.size() != nb_item) {
      // Since Arcane 2.4.0, 'm_family_serializer' is responsible for
      // retrieving the localId() of the sent entities. The array
      // m_receive_local_ids[i] must therefore always be correct.
      ARCANE_FATAL("Bad value for received_items family={0} n={1} expected={2}",
                   m_item_family->name(), local_ids, nb_item);
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Serializes the variable name to verify that deserialization
 * is correct.
 */
void ItemFamilyVariableSerializer::
_checkSerializationVariable(ISerializer* sbuf, IVariable* var)
{
  String var_full_name = var->fullName();
  switch (sbuf->mode()) {
  case ISerializer::ModeReserve:
    // Reserves for the variable name
    sbuf->reserve(var_full_name);
    break;
  case ISerializer::ModePut:
    sbuf->put(var_full_name);
    break;
  case ISerializer::ModeGet: {
    String expected_name;
    sbuf->get(expected_name);
    if (expected_name != var_full_name)
      ARCANE_FATAL("Incoherent variable var={0} expected={1}",
                   var_full_name, expected_name);
  } break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 * During serialization, \a rank is the rank of the target.
 * During deserialization, \a rank is the rank of the sender.
 */
void ItemFamilyVariableSerializer::
serialize(const ItemFamilySerializeArgs& args)
{
  ISerializer* sbuf = args.serializer();
  Int32ConstArrayView local_ids = args.localIds();

  bool is_recv = (sbuf->mode() == ISerializer::ModeGet);
  String mode_str = (is_recv) ? "recv" : "send";
  const bool is_debug = arcaneIsDebug();

  Int32 owner_rank = (is_recv) ? m_item_family->parallelMng()->commRank() : args.rank();

  _checkSerialization(sbuf, local_ids);

  for (IVariable* var : m_variables_to_exchange) {
    info(4) << "-- Serializing variable " << var->fullName()
            << " group=" << var->itemGroup().name()
            << " (n=" << var->itemGroup().size() << ")"
            << " mode=" << (int)sbuf->mode()
            << " target_rank=" << args.rank();

    _checkSerializationVariable(sbuf, var);

    if (is_debug) {
      ENUMERATE_ITEM (iitem, m_item_family->view(local_ids)) {
        debug(Trace::Highest) << "To " << mode_str << " : " << ItemPrinter(*iitem);
      }
      debug(Trace::High) << "To " << mode_str << " count = " << local_ids.size();
    }

    if (var->itemFamily() != m_item_family) {
      IItemFamily* var_family = var->itemFamily();
      ItemVector dest_items = MeshToMeshTransposer::transpose(m_item_family,
                                                              var_family,
                                                              m_item_family->view(local_ids),
                                                              false);
      Int32UniqueArray dest_lids;
      dest_lids.reserve(dest_items.size());
      ENUMERATE_ITEM (iitem, dest_items) {
        Int32 lid = iitem.localId();
        if (lid != NULL_ITEM_LOCAL_ID && iitem->owner() == owner_rank) {
          dest_lids.add(lid);
        }
      }
      debug(Trace::High) << "Serializing " << dest_lids.size() << " sub-items";

      if (var->isPartial())
        _serializePartialVariable(var, sbuf, dest_lids);
      else
        var->serialize(sbuf, dest_lids);
    }
    else {
      if (var->isPartial())
        _serializePartialVariable(var, sbuf, local_ids);
      else
        var->serialize(sbuf, local_ids);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Serialise une variable partielle.
 *
 * This method is not very performant and must be optimized.
 * For now, the operation is as follows for sending
 * - determines the list of entities to send based on the
 * new owner of the entities.
 * - serializes the uniqueId() of these entities
 * - serializes the variable.
 * For receiving:
 * - builds a hash map uniqueId() -> index in the variable
 * - deserializes the uniqueId() and converts them to indices
 * - deserializes the variable.
 *
 * TODO: avoid sending all IDs for every variable
 * TODO: use the information from ItemGroupsSerializer2 if possible to
 * know the list of entities to send.
 */
void ItemFamilyVariableSerializer::
_serializePartialVariable(IVariable* var, ISerializer* sbuf, Int32ConstArrayView local_ids)
{
  String var_full_name = var->fullName();

  ItemGroup group = var->itemGroup();
  ISerializer::eMode mode = sbuf->mode();
  switch (mode) {
  case ISerializer::ModeReserve:
  case ISerializer::ModePut: {
    Int32UniqueArray indexes_to_send;
    Int64UniqueArray unique_ids_to_send;
    std::set<Int32> items_to_send;
    for (Integer i = 0, n = local_ids.size(); i < n; ++i) {
      items_to_send.insert(local_ids[i]);
    }
    ENUMERATE_ITEM (iitem, group) {
      if (items_to_send.find(iitem.itemLocalId()) != items_to_send.end()) {
        indexes_to_send.add(iitem.index());
        unique_ids_to_send.add((*iitem).uniqueId());
      }
    }
    Integer nb_item_to_send = indexes_to_send.size();
    if (mode == ISerializer::ModeReserve) {
      // Reserves for the number of elements and for each element
      sbuf->reserveInt64(1);
      sbuf->reserveSpan(eBasicDataType::Int64, nb_item_to_send);
    }
    else {
      sbuf->putInt64(nb_item_to_send);
      sbuf->putSpan(unique_ids_to_send);
    }
    var->serialize(sbuf, indexes_to_send);
  } break;
  case ISerializer::ModeGet: {
    Int32UniqueArray indexes;
    Int64UniqueArray unique_ids;
    String expected_name;
    //sbuf->get(expected_name);
    //if (expected_name!=var_full_name)
    //ARCANE_FATAL("Incoherent variable var={0} expected={1}",
    //var_full_name,expected_name);
    Int64 nb_item = sbuf->getInt64();
    unique_ids.resize(nb_item);
    sbuf->getSpan(unique_ids);
    Integer nb_item_in_variable = group.size();
    HashTableMapT<Int64, Int32> unique_ids_to_index(nb_item_in_variable, true);
    ENUMERATE_ITEM (iitem, group) {
      unique_ids_to_index.nocheckAdd((*iitem).uniqueId(), iitem.index());
    }
    indexes.resize(nb_item);
    for (Integer i = 0; i < nb_item; ++i) {
      // Checks that the entity is present. This is normally the case unless
      // \a group is not consistent between the PE (that is, for example an
      // entity \a x present on 2 PE but which is in \a group only for one
      // of the two PE).
      HashTableMapT<Int64, Int32>::Data* data = unique_ids_to_index.lookup(unique_ids[i]);
      if (!data)
        ARCANE_FATAL("Can not find item with unique_id={0} index={1}", unique_ids[i], i);
      indexes[i] = data->value();
    }

    var->serialize(sbuf, indexes);
  } break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
