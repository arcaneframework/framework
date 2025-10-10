// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchanger.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Gestion d'un échange de maillage entre sous-domaines.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/IParallelMng.h"
#include "arcane/Timer.h"
#include "arcane/IItemFamilyPolicyMng.h"
#include "arcane/IItemFamilyExchanger.h"
#include "arcane/IParticleFamily.h"

#include "arcane/mesh/MeshExchanger.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/MeshExchange.h"
#include "arcane/core/internal/IMeshModifierInternal.h"
#include "arcane/core/internal/IItemFamilySerializerMngInternal.h"
#include "arcane/core/internal/IMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchanger::
MeshExchanger(IMesh* mesh,ITimeStats* stats)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_time_stats(stats)
, m_phase(ePhase::Init)
{
  // Temporairement utilise une variable d'environnement pour spécifier le
  // nombre maximum de messages en vol ou si on souhaite utiliser les collectives
  String max_pending_str = platform::getEnvironmentVariable("ARCANE_MESH_EXCHANGE_MAX_PENDING_MESSAGE");
  if (!max_pending_str.null()){
    Int32 max_pending = 0;
    if (!builtInGetValue(max_pending,max_pending_str))
      m_exchanger_option.setMaxPendingMessage(max_pending);
  }

  String use_collective_str = platform::getEnvironmentVariable("ARCANE_MESH_EXCHANGE_USE_COLLECTIVE");
  if (use_collective_str=="1" || use_collective_str=="TRUE")
    m_exchanger_option.setExchangeMode(ParallelExchangerOptions::EM_Collective);

  m_exchanger_option.setVerbosityLevel(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchanger::
~MeshExchanger()
{
  for( IItemFamilyExchanger* exchanger : m_family_exchangers )
    delete exchanger;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
build()
{
  if ( !m_mesh->itemFamilyNetwork() || !IItemFamilyNetwork::plug_serializer )
  { // handle family order by hand
    // Liste ordonnée des familles triée spécifiquement pour garantir un certain ordre
    // dans les échanges. Pour l'instant l'ordre est déterminé comme suit:
    // - d'abord Cell, puis Face, Edge et Node
    // - ensuite, les Particles doivent être gérées avant les familles de DualNode.
    UniqueArray<IItemFamily*> sorted_families;
    IItemFamilyCollection families(m_mesh->itemFamilies());
    sorted_families.reserve(families.count());
    sorted_families.add(m_mesh->cellFamily());
    sorted_families.add(m_mesh->faceFamily());
    sorted_families.add(m_mesh->edgeFamily());
    sorted_families.add(m_mesh->nodeFamily());
    for( IItemFamily* family : families )
    {
      IParticleFamily* particle_family = family->toParticleFamily();
      if (particle_family)
        sorted_families.add(family);
    }

    // Liste des instances gérant les échanges d'une famille.
    // ATTENTION: il faut garantir la libération des pointeurs associés.
    //m_family_exchangers.reserve(families.count());

    // Création de chaque échangeur associé à une famille.
    std::map<IItemFamily*,IItemFamilyExchanger*> family_exchanger_map;
    for( IItemFamily* family : sorted_families ){
      _addItemFamilyExchanger(family);
    }
  }
  else
  {
    if(m_mesh->useMeshItemFamilyDependencies())
    {
      _buildWithItemFamilyNetwork();
    }
    else
    {
      std::set<String> family_set ;
      UniqueArray<IItemFamily*> sorted_families;
      IItemFamilyCollection families(m_mesh->itemFamilies());
      sorted_families.reserve(families.count());
      sorted_families.add(m_mesh->cellFamily());
      family_set.insert(m_mesh->cellFamily()->name()) ;
      sorted_families.add(m_mesh->faceFamily());
      family_set.insert(m_mesh->faceFamily()->name()) ;
      sorted_families.add(m_mesh->edgeFamily());
      family_set.insert(m_mesh->edgeFamily()->name()) ;
      sorted_families.add(m_mesh->nodeFamily());
      family_set.insert(m_mesh->nodeFamily()->name()) ;
      for( IItemFamily* family : families )
      {
        IParticleFamily* particle_family = family->toParticleFamily();
        if (particle_family)
        {
          sorted_families.add(family);
          family_set.insert(family->name()) ;
        }
      }

      for( auto family : m_mesh->itemFamilyNetwork()->getFamilies(IItemFamilyNetwork::InverseTopologicalOrder) )
      {
        auto value = family_set.insert(family->name()) ;
        if(value.second)
        {
          sorted_families.add(family) ;
        }
      }

      // Liste des instances gérant les échanges d'une famille.
      // ATTENTION: il faut garantir la libération des pointeurs associés.
      //m_family_exchangers.reserve(families.count());

      // Création de chaque échangeur associé à une famille.
      std::map<IItemFamily*,IItemFamilyExchanger*> family_exchanger_map;
      for( IItemFamily* family : sorted_families ){
        _addItemFamilyExchanger(family);
      }
    }
  }
  m_phase = ePhase::ComputeInfos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_buildWithItemFamilyNetwork()
{
  m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family) {
    _addItemFamilyExchanger(family);
  }, IItemFamilyNetwork::InverseTopologicalOrder);
  // Particle should be handled soon
  for( IItemFamily* family : m_mesh->itemFamilies() ){
      IParticleFamily* particle_family = family->toParticleFamily();
      if (particle_family)
        _addItemFamilyExchanger(family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_addItemFamilyExchanger(IItemFamily* family)
{
  IItemFamilyExchanger* exchanger = family->policyMng()->createExchanger();
  m_family_exchangers.add(exchanger);
  m_family_exchanger_map.insert(std::make_pair(family,exchanger));
  exchanger->setParallelExchangerOption(m_exchanger_option);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_checkPhase(ePhase wanted_phase)
{
  if (m_phase!=wanted_phase)
    ARCANE_FATAL("Invalid exchange phase wanted={0} current={1}",
                 (int)wanted_phase,(int)m_phase);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshExchanger::
computeExchangeInfos()
{
  _checkPhase(ePhase::ComputeInfos);

  // TODO: faire en sorte de pouvoir customiser le calcul de l'échange
  // et faire cela par famille si possible.
  MeshExchange mesh_exchange(m_mesh);
  info() << "MeshExchange begin date=" << platform::getCurrentDateTime();
  {
    Timer::Action ts_action1(m_time_stats,"MeshExchangeComputeInfos",true);
    mesh_exchange.computeInfos();
    // MeshExchange a mise une marque NeedRemove sur les cellules partant complètement de ce proc
  }

  IItemFamily* cell_family = m_mesh->cellFamily();
  IItemFamilyExchanger* cell_exchanger = findExchanger(cell_family);

  // Détermine d'abord les infos à échanger sur les mailles car s'il n'y a aucune maille
  // à échanger le partitionnement s'arête.
  // NOTE GG: il faut voir si cela reste le cas avec les liens et les noeuds duaux.
  cell_exchanger->setExchangeItems(mesh_exchange.getItemsToSend(cell_family));
  if (cell_exchanger->computeExchangeInfos()){
    pwarning() << "No load balance is performed";
    return true;
  }

  // Détermine pour chaque famille la liste des informations à échanger.
  for( IItemFamilyExchanger* exchanger : m_family_exchangers ){
    // L'échange des mailles a déjà été fait.
    if (exchanger==cell_exchanger)
      continue;
    IItemFamily* family = exchanger->itemFamily();
    info() << "ComputeExchange family=" << family->name()
           << " date=" << platform::getCurrentDateTime();
    // Pour les familles de particules qui ne supportent pas la notion
    // de fantôme, il faut déterminer explicitement la liste des entités à échanger
    // via l'appel à computeExchangeItems().
    // Pour les autres familles où les familles de particules qui on la notion
    // de fantôme, cette liste à déjà été déterminée lors de l'appel à
    // mesh_exchange.computeInfos().
    IParticleFamily* particle_family = family->toParticleFamily() ;
    if (particle_family && particle_family->getEnableGhostItems()==false)
      exchanger->computeExchangeItems();
    else
      exchanger->setExchangeItems(mesh_exchange.getItemsToSend(family));
    exchanger->computeExchangeInfos();
  }

  // Recopie le champ owner() dans les ItemInternal pour qu'il
  // soit cohérent avec la variable correspondante

  // ATTENTION: Il faut absolument que les owner() des ItemInternal
  // soient corrects avant qu'on envoie les mailles qui nous appartenaient
  // aux sous-domaines à qui elles vont ensuite appartenir.

  // A noter qu'on ne peut pas fusionner cette boucle avec la précédente car
  // les familles ont besoin des infos des autres familles pour déterminer
  // la liste des entités à envoyer.
  Int32 rank = m_mesh->meshPartInfo().partRank();
  for( IItemFamilyExchanger* exchanger : m_family_exchangers ){
    IItemFamily* family = exchanger->itemFamily();
    VariableItemInt32& owners(family->itemsNewOwner());
    ENUMERATE_ITEM(i,family->allItems()){
      Item item = *i;
      Integer new_owner = owners[item];
      item.mutableItemBase().setOwner(new_owner,rank);
    }
    family->notifyItemsOwnerChanged();
  }

  m_phase = ePhase::ProcessExchange;

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
processExchange()
{
  _checkPhase(ePhase::ProcessExchange);

  info() << "ExchangeItems date=" << platform::getCurrentDateTime()
         << " MemUsed=" << platform::getMemoryUsed();

  Timer::Action ts_action1(m_time_stats,"MessagesExchange",true);
  for( IItemFamilyExchanger* e : m_family_exchangers ){
    // NOTE: Pour pouvoir envoyer tous les messages en même temps et les réceptions
    // aussi, il faudra peut être prévoir d'utiliser des tags MPI.
    e->prepareToSend();   // Préparation de toutes les données à envoyer puis sérialisation
    e->processExchange(); // Envoi effectif
    e->releaseBuffer();
  }
  m_phase = ePhase::RemoveItems;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
removeNeededItems()
{
  _checkPhase(ePhase::RemoveItems);

  // Maintenant que tous les messages avec le maillage avant modification
  // sont envoyés et réceptionnés, on peut modifier ce maillage en lui
  // supprimant les éléments qui ne lui appartiennent plus et en ajoutant
  // les nouveaux.

  // TODO: faire la méthode de supression par famille.

  // Pour les familles autres que les particules sans fantômes, cela
  // se fait dans le DynamicMeshIncrementalBuilder.
  // Pour les particules sans fantôme, cela se fait ici.
  info() << "RemoveItems date=" << platform::getCurrentDateTime();
  Timer::Action ts_action1(m_time_stats,"RemoveSendedItems",true);

  for( IItemFamilyExchanger* exchanger : m_family_exchangers ){
    IParticleFamily* particle_family = exchanger->itemFamily()->toParticleFamily() ;
    if (particle_family && particle_family->getEnableGhostItems()==false)
      exchanger->removeSentItems(); // integre le traitemaint des sous-maillages (pour les particules)
  }

  // Supprime les entités qui ne sont plus liées au sous-domaine
  m_mesh->modifier()->_modifierInternalApi()->removeNeedRemoveMarkedItems();

  m_phase = ePhase::AllocateItems;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
allocateReceivedItems()
{
  _checkPhase(ePhase::AllocateItems);
  {
    info() << "AllocItems date=" << platform::getCurrentDateTime();
    Timer::Action ts_action1(m_time_stats,"ReadAndAllocItems",true);
    // Il faut faire en premier l'échange de mailles
    // Cela est garanti par le fait que le premier élément de family_exchangers
    // est celui de la famille de maille.
    for( IItemFamilyExchanger* e : m_family_exchangers ){
      e->readAndAllocItems(); // Attention, ne procède plus sur les différents sous-maillages
    }
    // If needed, finalize item allocations (for polyhedral meshes)
    auto* family_serializer_mng = m_mesh->_internalApi()->familySerializerMng();
    if (family_serializer_mng) family_serializer_mng->finalizeItemAllocation();

    // Build item relations (only dependencies are build in readAndAllocItems)
    // only for families registered in the graph
    if (m_mesh->itemFamilyNetwork() && m_mesh->itemFamilyNetwork()->isActivated())
    {
      auto family_set = m_mesh->itemFamilyNetwork()->getFamilies();
      for (auto family : family_set) {
        m_family_exchanger_map[family]->readAndAllocItemRelations();
      }
    }

    // Separate mesh and submesh
    for( IItemFamilyExchanger* e : m_family_exchangers ){
      e->readAndAllocSubMeshItems(); // Procède sur les différents sous-maillages
    }
  }

  // Il est possible que les propriétaires des entités aient changés
  // suite a readAndAllocItems() même si aucune entité n'a été ajoutée.
  // Il faut donc l'indiquer aux familles.
  for( IItemFamilyExchanger* e : m_family_exchangers ){
    e->itemFamily()->notifyItemsOwnerChanged(); // appliqué jusqu'à un niveau de sous-maillage
  }

  m_phase = ePhase::UpdateItemGroups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
updateItemGroups()
{
  _checkPhase(ePhase::UpdateItemGroups);

  info() << "ReadGroups date=" << platform::getCurrentDateTime();
  // Maintenant que le nouveau maillage est créé on lit les groupes
  for( IItemFamilyExchanger* e : m_family_exchangers )
    e->readGroups();

  m_phase = ePhase::UpdateVariables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
updateVariables()
{
  _checkPhase(ePhase::UpdateVariables);

  info() << "ReadVariables date=" << platform::getCurrentDateTime();
  Timer::Action ts(m_time_stats,"ReadVariables",true);
  // Maintenant que les entités sont créées et les groupes mis à jour,
  // on peut mettre à jour les variables.
  for( IItemFamilyExchanger* e : m_family_exchangers )
    e->readVariables();

  m_phase = ePhase::Finalize;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
finalizeExchange()
{
  _checkPhase(ePhase::Finalize);

  // Finalize les échanges
  // Cela doit etre fait apres le compactage car dans le cas des interfaces liees,
  // il ne faut plus changer la numérotation des localId() une fois les
  // structures TiedInterface mises à jour.
  // TODO: il faudra supprimer cela en faisant ce traitement avant mais
  // pour cela il faut que TiedInterfaceMng soit notifié du compactage pour
  // mettre à jour les localId() de ses faces et de ses noeuds
  for( IItemFamilyExchanger* e : m_family_exchangers )
    e->finalizeExchange();

  m_phase = ePhase::Ended;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyExchanger* MeshExchanger::
findExchanger(IItemFamily* family)
{
  auto x = m_family_exchanger_map.find(family);
  if (x==m_family_exchanger_map.end())
    ARCANE_FATAL("No exchanger for family name={0}",family->name());
  return x->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshExchanger::
mesh() const
{
  return m_mesh->toPrimaryMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_setNextPhase(ePhase next_phase)
{
  m_phase = next_phase;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
