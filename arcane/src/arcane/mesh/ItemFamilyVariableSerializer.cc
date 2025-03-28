// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyVariableSerializer.cc                             (C) 2000-2024 */
/*                                                                           */
/* Gère la sérialisation/désérialisation des variables d'une famille.        */
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
 * plutôt qu'une instance de cette classe gère la sérialisation des variables
 * de la famille mère et de toutes les familles filles, il faudrait une
 * instance par famille.
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
  // Détermine la liste des variables a échanger
  // On y intégre aussi les variables issues des familles enfants
  // TODO: il faudrait récupérer directement les variables de chaque
  // famille via IItemFamily::usedVariables() mais cette méthode ne garantit
  // par l'ordre alphabétique des envois. En pensant par un std::map, on
  // devrait pouvoir s'en sortir.
  IVariableMng* vm = m_item_family->mesh()->variableMng();
  VariableCollection used_vars(vm->usedVariables());

  UniqueArray<IItemFamily*> family_to_exchange;
  family_to_exchange.add(m_item_family); // La famille courante
  IItemFamilyCollection child_families = m_item_family->childFamilies();
  for( IItemFamily* child_family : child_families )
    family_to_exchange.add(child_family);
   
  for (Integer i=0; i<family_to_exchange.size(); ++i ){
    IItemFamily* current_family = family_to_exchange[i];
    info(4) << " Serializing family " << current_family->fullName();

    // TODO: récupérer directement les variables utilisées de la famille
    // via ItemFamily::usedVariables
    for( VariableCollection::Enumerator i_var(used_vars); ++i_var; ){
      IVariable* var = *i_var;
      // TODO: vérifier s'il faut échanger toutes les variables.
      // Faut-il envoyer les variables PNoDump ?
      // TODO: appeler le writeObservable de IVariableMng ou en faire
      // un spécifique pour l'équilibrage
      // TODO: vérifier que tout les sous-domaines ont les mêmes valeurs
      // pour m_variables_to_exchange avec les variables dans le même ordre.
      bool no_exchange = (var->property() & IVariable::PNoExchange);
      if (no_exchange)
        continue;
      IItemFamily* var_family = var->itemFamily();
      if (var_family==current_family) {
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
 * \brief Sérialize un nombre magique et le nombre d'entité afin de vérifier que
 * la désérialisation est correct.
 */
void ItemFamilyVariableSerializer::
_checkSerialization(ISerializer* sbuf,Int32ConstArrayView local_ids)
{
  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    // Réserve pour la liste uniqueId() des entités transférées
    sbuf->reserveInteger(1); // Nombre magique pour readVariables();
    sbuf->reserveInteger(1); // Nombre d'entités sérialisées.
    break;
  case ISerializer::ModePut:
    // Sérialise le nombre d'entités transférées
    sbuf->put(VARIABLE_MAGIC_NUMBER);
    sbuf->put(local_ids.size());
    break;
  case ISerializer::ModeGet:
    // Vérifie pas d'erreurs de sérialisation.
    Integer magic_number = sbuf->getInteger();
    if (magic_number!=VARIABLE_MAGIC_NUMBER)
      ARCANE_FATAL("Internal error: bad magic number expected={0} found={1}",
                   VARIABLE_MAGIC_NUMBER,magic_number);

    // Récupère le nombre d'entités transférées pour vérification
    Integer nb_item = sbuf->getInteger();
    if (local_ids.size()!=nb_item){
      // Depuis la 2.4.0 de Arcane 'm_family_serializer' se charge de
      // récupérer les localId() des entités envoyées. Le tableau
      // m_receive_local_ids[i] doit donc toujours être correct.
      ARCANE_FATAL("Bad value for received_items family={0} n={1} expected={2}",
                   m_item_family->name(),local_ids,nb_item);
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sérialise le nom de la variable pour vérifier que la désérialisation
 * est correcte.
 */
void ItemFamilyVariableSerializer::
_checkSerializationVariable(ISerializer* sbuf,IVariable* var)
{
  String var_full_name = var->fullName();
  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    // Réserve pour le nom de la variable
    sbuf->reserve(var_full_name);
    break;
  case ISerializer::ModePut:
    sbuf->put(var_full_name);
    break;
  case ISerializer::ModeGet:
    {
      String expected_name;
      sbuf->get(expected_name);
      if (expected_name!=var_full_name)
        ARCANE_FATAL("Incoherent variable var={0} expected={1}",
                     var_full_name,expected_name);
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 * En sérialisation, \a rank est le rang de la cible.
 * En désérialisation, \a rank est le rang de celui qui envoie le message.
 */
void ItemFamilyVariableSerializer::
serialize(const ItemFamilySerializeArgs& args)
{
  ISerializer* sbuf = args.serializer();
  Int32ConstArrayView local_ids  = args.localIds();

  bool is_recv = (sbuf->mode()==ISerializer::ModeGet);
  String mode_str = (is_recv) ? "recv" :"send";
  const bool is_debug = arcaneIsDebug();

  Int32 owner_rank = (is_recv) ? m_item_family->parallelMng()->commRank() : args.rank();

  _checkSerialization(sbuf,local_ids);

  for( IVariable* var : m_variables_to_exchange ){
    info(4) << "-- Serializing variable " << var->fullName()
            << " group=" << var->itemGroup().name()
            << " (n=" << var->itemGroup().size() << ")"
            << " mode=" << (int)sbuf->mode()
            << " target_rank=" << args.rank();

    _checkSerializationVariable(sbuf,var);

    if (is_debug){
      ENUMERATE_ITEM(iitem, m_item_family->view(local_ids)){
        debug(Trace::Highest) << "To " << mode_str << " : " << ItemPrinter(*iitem);
      }
      debug(Trace::High) << "To " << mode_str << " count = " << local_ids.size();
    }

    if (var->itemFamily() != m_item_family){
      IItemFamily * var_family = var->itemFamily();
      ItemVector dest_items = MeshToMeshTransposer::transpose(m_item_family,
                                                              var_family, 
                                                              m_item_family->view(local_ids),
                                                              false);
      Int32UniqueArray dest_lids; dest_lids.reserve(dest_items.size());
      ENUMERATE_ITEM(iitem, dest_items) {
        Int32 lid = iitem.localId();
        if (lid != NULL_ITEM_LOCAL_ID && iitem->owner() == owner_rank) {
          dest_lids.add(lid);
        }
      }
      debug(Trace::High) << "Serializing " << dest_lids.size() << " sub-items";

      if (var->isPartial())
        _serializePartialVariable(var,sbuf,dest_lids);
      else
        var->serialize(sbuf,dest_lids);
    }
    else{
      if (var->isPartial())
        _serializePartialVariable(var,sbuf,local_ids);
      else
        var->serialize(sbuf,local_ids);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Serialise une variable partielle.
 *
 * Cette méthode n'est pas très performante et doit être optimisée.
 * En attendant, le fonctionnement est le suivant pour l'envoie
 * - détermine la liste des entités a envoyer en se basant sur le
 * nouveau propriétaire des entités.
 * - sérialise les uniqueId() de ces entités
 * - sérialise la variable.
 * Pour la réception:
 * - construit une table de hashage uniqueId() -> index dans la variable
 * - désérialise les uniqueId() et les convertie en index
 * - désérialise la variable.
 *
 * TODO: eviter d'envoyer tous les ids pour chaque variable
 * TODO: utiliser les infos de ItemGroupsSerializer2 si possible pour
 * connaitre la liste des entités à envoyer.
 */
void ItemFamilyVariableSerializer::
_serializePartialVariable(IVariable* var,ISerializer* sbuf,Int32ConstArrayView local_ids)
{
  String var_full_name = var->fullName();

  ItemGroup group = var->itemGroup();
  ISerializer::eMode mode = sbuf->mode();
  switch(mode){
  case ISerializer::ModeReserve:
  case ISerializer::ModePut:
    {
      Int32UniqueArray indexes_to_send;
      Int64UniqueArray unique_ids_to_send;
      std::set<Int32> items_to_send;
      for( Integer i=0, n=local_ids.size(); i<n; ++i ){
        items_to_send.insert(local_ids[i]);
      }
      ENUMERATE_ITEM(iitem,group){
        if (items_to_send.find(iitem.itemLocalId())!=items_to_send.end()){
          indexes_to_send.add(iitem.index());
          unique_ids_to_send.add((*iitem).uniqueId());
        }
      }
      Integer nb_item_to_send = indexes_to_send.size();
      if (mode==ISerializer::ModeReserve){
        // Réserve pour le nombre d'élément et pour chaque élément
        sbuf->reserveInt64(1);
        sbuf->reserveSpan(eBasicDataType::Int64,nb_item_to_send);
      }
      else{
        sbuf->putInt64(nb_item_to_send);
        sbuf->putSpan(unique_ids_to_send);
      }
      var->serialize(sbuf,indexes_to_send);
    }
    break;
  case ISerializer::ModeGet:
    {
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
      HashTableMapT<Int64,Int32> unique_ids_to_index(nb_item_in_variable,true);
      ENUMERATE_ITEM(iitem,group){
        unique_ids_to_index.nocheckAdd((*iitem).uniqueId(),iitem.index());
      }
      indexes.resize(nb_item);
      for( Integer i=0; i<nb_item; ++i ){
        // Vérifie que l'entité est bien présente. C'est normalement le cas sauf
        // si \a group n'est pas cohérent entre les PE (c'est à dire par exemple une
        // entité \a x présente sur 2 PE mais qui est dans \a group que pour un seul
        // des deux PE).
        HashTableMapT<Int64,Int32>::Data* data = unique_ids_to_index.lookup(unique_ids[i]);
        if (!data)
          ARCANE_FATAL("Can not find item with unique_id={0} index={1}",unique_ids[i],i);
        indexes[i] = data->value();
      }

      var->serialize(sbuf,indexes);
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

