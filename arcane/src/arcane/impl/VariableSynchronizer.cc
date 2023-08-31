// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizer.cc                                     (C) 2000-2023 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableSynchronizer.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/Timer.h"
#include "arcane/core/parallel/IStat.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/impl/DataSynchronizeInfo.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCANE_DEBUG_SYNC
namespace
{
// Mettre à true pour afficher des informations supplémentaires pour le débug.
bool global_debug_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IDataSynchronizeImplementationFactory>
arcaneCreateSimpleVariableSynchronizerFactory(IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
VariableSynchronizer(IParallelMng* pm,const ItemGroup& group,
                     Ref<IDataSynchronizeImplementationFactory> implementation_factory)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_item_group(group)
{
  m_sync_list = DataSynchronizeInfo::create();
  if (!implementation_factory.get())
    implementation_factory = arcaneCreateSimpleVariableSynchronizerFactory(pm);
  m_implementation_factory = implementation_factory;

  GroupIndexTable* table = nullptr;
  if (!group.isAllItems())
    table = group.localIdToIndex().get();
  VariableSynchronizeDispatcherBuildInfo bi(pm,table,implementation_factory,m_sync_list);
  m_dispatcher = IVariableSynchronizerDispatcher::create(bi);
  if (!m_dispatcher)
    ARCANE_FATAL("No synchronizer created");
  m_multi_dispatcher = IVariableSynchronizerMultiDispatcher::create(bi);
  if (!m_multi_dispatcher)
    ARCANE_FATAL("No multi synchronizer created");

  {
    String s = platform::getEnvironmentVariable("ARCANE_ALLOW_MULTISYNC");
    if (s=="0" || s=="FALSE" || s=="false")
      m_allow_multi_sync = false;
  }
  {
    String s = platform::getEnvironmentVariable("ARCANE_TRACE_SYNCHRONIZE");
    if (s=="1" || s=="TRUE" || s=="true")
      m_trace_sync = true;
  }
  {
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_AUTO_COMPARE_SYNCHRONIZE", true))
      m_is_compare_sync = (v.value()!=0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
~VariableSynchronizer()
{
  delete m_sync_timer;
  delete m_multi_dispatcher;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Création de la liste des éléments de synchronisation.
 *
 Pour générer les infos de synchronisations, on suppose que le owner()
 de chaque entité est correcte

 A partir du fichier de communication, contruit les structures pour la
 synchronisation. Il s'agit d'une liste d'éléments, chaque élément étant
 composé du rang du processeur avec lequel il faut communiquer et les
 localId() des entités du maillage qu'on doit lui envoyer et qu'on doit
 réceptionner.

 Si le groupe associé à cette instance est allItems(), on vérifie que toutes
 les entités de la famille sont soit propres au domaine, soit fantomes.
 Si une entité n'est pas dans ce cas, alors elle ne sera pas synchronisée et 
 la cohérence du parallélisme ne sera pas assuré: il s'agit d'une erreur fatale.

 Le fonctionnement sur tout groupe (pluys que allItems) est principalement
 subordonné au fait que changeLocalIds() soit implementé sur tous les groupes.
*/
void VariableSynchronizer::
compute()
{
  IItemFamily* item_family = m_item_group.itemFamily();
  Int32 my_rank = m_parallel_mng->commRank();
  Int32 nb_rank = m_parallel_mng->commSize();

  m_is_verbose = traceMng()->verbosityLevel()>=4;

  UniqueArray< SharedArray<Int32> > boundary_items(nb_rank);

  info() << "Compute synchronize informations group=" << m_item_group.name()
         << " family=" << item_family->fullName()
         << " this=" << this
         << " group size=" << m_item_group.size()
         << " is_verbose=" << m_is_verbose;

  {
    Integer nb_error = 0;
    Int64UniqueArray bad_items_uid;
    ENUMERATE_ITEM(i,m_item_group){
      Item item = *i;
      impl::ItemBase item_internal = item.itemBase();
      Int32 owner = item_internal.owner();
      if (owner==my_rank)
        continue;
      Int64 uid = item_internal.uniqueId().asInt64();
      if (owner==A_NULL_RANK || owner>=nb_rank){
        ++nb_error;
        if (nb_error<10)
          bad_items_uid.add(uid);
        continue;
      }
      if (global_debug_sync){
        info() << "Add entity uid=" << uid
               << " lid=" << item_internal.localId() << " to the subdomain " << owner;
      }
      boundary_items[owner].add(item_internal.localId());
    }
    if (nb_error!=0){
      for( Integer i=0, is=bad_items_uid.size(); i<is; ++i ){
        info() << "ERROR: The entity uid=" << bad_items_uid[i]
               << " group=" << m_item_group.name() << " doesn't belong to "
               << "any subdomain or belong to an invalid subdomain";
      }
      ARCANE_FATAL("Error while creating synchronization information");
    }
  }

  _createList(boundary_items);

  //_printSyncList();
  
  if (m_is_verbose){
    info() << "Begin compute dispatcher Date=" << platform::getCurrentDateTime();
  }
  m_dispatcher->compute();
  if (m_is_verbose){
    info() << "End compute dispatcher Date=" << platform::getCurrentDateTime();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_createList(UniqueArray<SharedArray<Int32> >& boundary_items)
{
  m_sync_list->clear();

  IItemFamily* item_family = m_item_group.itemFamily();
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  info(4) << "VariableSynchronizer::createList() begin for group=" << m_item_group.name();

  // Table du voisinage connu par items fantomes.
  // Ceci n'est pas obligatoirement la liste finale pour m_communicating_ranks dans le cas 
  // de relation non symétrique ghost/shared entre processeurs (si l'un des deux vaut 0)
  // Le traitement complémentaire apparaît après la section "Réciprocité des communications"
  m_communicating_ranks.clear();
  Int32UniqueArray communicating_ghost_ranks;
  for( Integer i=0; i<nb_rank; ++i ){
    if (boundary_items[i].empty())
      continue;
    communicating_ghost_ranks.add(i);
    if (global_debug_sync){
      ItemInfoListView items_internal(item_family);
      for( Integer z=0, zs=boundary_items[i].size(); z<zs; ++z ){
        Item item = items_internal[boundary_items[i][z]];
        info() << "Item uid=" << item.uniqueId() << ",lid=" << item.localId();
      }
    }
  }

  Integer nb_comm_rank = communicating_ghost_ranks.size();

  Int32UniqueArray nb_ghost(nb_rank);
  Int32UniqueArray nb_share(nb_rank);
  nb_ghost.fill(0);

  // Nombre maximum de sous-domaines connectés. Sert pour dimensionner
  // les tableaux pour le allGather()
  Integer max_comm_rank = pm->reduce(Parallel::ReduceMax,nb_comm_rank);
  debug() << "communicating sub domains my=" << nb_comm_rank
          << " max=" << max_comm_rank;

  // Liste des groupes des mailles fantômes.
  UniqueArray<SharedArray<Int32> > ghost_group_list(boundary_items.size());

  // Récupère les listes des entités fantômes.
  for( Integer i=0; i<nb_comm_rank; ++i ){
    Int32 current_rank = communicating_ghost_ranks[i];
    SharedArray<Int32>& ghost_grp = boundary_items[current_rank];
    ghost_group_list[i] = ghost_grp;
    nb_ghost[current_rank] = ghost_grp.size();
  }

  UniqueArray<ShareRankInfo> share_rank_info;
  UniqueArray<GhostRankInfo> ghost_rank_info;
  {
    Integer gather_size = 1+(max_comm_rank*2);
    Int32UniqueArray global_ghost_info(gather_size*nb_rank);
    {
      // Chaque sous-domaine construit un tableau indiquand pour
      // chaque groupe d'éléments fantômes, le processeur concerné et
      // le nombre d'éléments de ce groupe.
      // Ce tableau sera ensuite regroupé sur l'ensemble des sous-domaines
      // (par un allGather()) afin que chaque sous-domaine puisse le parcourir
      // et ensuite savoir qui possède ses mailles partagés.
      Int32UniqueArray local_ghost_info(gather_size);
      Integer pos = 0;
      local_ghost_info[pos++] = nb_comm_rank; // Indique le nombre d'éléments du tableau
      debug() << "Send local info " << nb_comm_rank;
      for( Integer index=0, s=communicating_ghost_ranks.size(); index<s; ++index ){
        local_ghost_info[pos++] = communicating_ghost_ranks[index];
        local_ghost_info[pos++] = ghost_group_list[index].size();
        debug() << "Send local info i=" << index << " target=" << communicating_ghost_ranks[index]
                << " nb=" << ghost_group_list[index].size();
      }
      if (m_is_verbose){
        info() << "AllGather size() " << local_ghost_info.size()
               << ' ' << global_ghost_info.size()
               << " begin_date=" << platform::getCurrentDateTime();
      }
      pm->allGather(local_ghost_info,global_ghost_info);
      if (m_is_verbose){
        info() << "AllGather end_date=" << platform::getCurrentDateTime();
      }
    }
    {
      for( Integer index=0, s=nb_rank; index<s; ++index ){
        Integer pos = gather_size*index;
        Integer sub_size = global_ghost_info[pos++];
        for( Integer sub_index=0; sub_index<sub_size; ++sub_index ){
          Integer proc_id = global_ghost_info[pos++];
          Integer nb_elem = global_ghost_info[pos++];
          if (proc_id==my_rank){
            if (global_debug_sync){
              info() << "Get for share group " << index << ' ' << nb_elem;
            }
            share_rank_info.add(ShareRankInfo(index,nb_elem));
          }
        }
      }
    }
    
    {
      // Créé les infos concernant les mailles fantômes
      Integer nb_send = communicating_ghost_ranks.size();
      ghost_rank_info.resize(nb_send);
      for( Integer i=0; i<nb_send; ++i ){
        SharedArray<Int32> gr = ghost_group_list[i];
        ghost_rank_info[i].setInfos(communicating_ghost_ranks[i],gr);
      }
    }
  }
  //pm->barrier();
  ItemInfoListView items_internal(item_family);
  {
    {
      // Réciprocité des communications
      // Pour que les synchronisations se fassent bien, il faut que
      // 'share_rank_info' et 'ghost_rank_info' aient
      // le même nombre d'éléments. Si ce n'est pas le cas, c'est
      // qu'un processeur 'n' possède des mailles partagés avec le proc 'm'
      // sans que cela soit réciproque. Si c'est le cas, on rajoute
      // dans 'share_rank_info' une référence à ce sous-domaine
      // avec aucun élément à envoyer.
      Integer nb_recv = share_rank_info.size();
      Integer nb_send = ghost_rank_info.size();

      if (global_debug_sync){
        info() << "Infos before auto add: send " << nb_send << " recv " << nb_recv;
        for( Integer i=0; i<ghost_rank_info.size(); ++i ){
          const GhostRankInfo& asdi = ghost_rank_info[i];
          info() << "Ghost: " << i << asdi.nbItem() << ' ' << asdi.rank();
        }
        for( Integer i=0; i<share_rank_info.size(); ++i ){
          const ShareRankInfo& asdi = share_rank_info[i];
          info() << "Shared: " << i << ' ' << asdi.nbItem() << ' ' << asdi.rank();
        }
      }

      for( Integer i=0; i<nb_send; ++i ){
        Integer proc_id = ghost_rank_info[i].rank();
        Integer z = 0;
        for( ; z<nb_recv; ++z )
          if (share_rank_info[z].rank()==proc_id)
            break;
        debug(Trace::Highest) << "CHECKS " << proc_id << ' ' << z << ' ' << nb_recv;
        if (z==nb_recv){
          debug() << "Add communication with the subdomain " << proc_id;
          share_rank_info.add(ShareRankInfo(proc_id));
        }
      }

      for( Integer i=0; i<nb_recv; ++i ){
        Integer proc_id = share_rank_info[i].rank();
        Integer z = 0;
        for( ; z<nb_send; ++z )
          if (ghost_rank_info[z].rank()==proc_id)
            break;
        debug(Trace::Highest) << "CHECKR " << proc_id << ' ' << z << ' ' << nb_send;
        if (z==nb_send){
          debug() << "Add communication with subdomain " << proc_id;
          ghost_rank_info.add(GhostRankInfo(proc_id));
        }
      }

      if (ghost_rank_info.size()!=share_rank_info.size()){
        ARCANE_FATAL("Problem with the number of subdomain shared ({0}) and ghosts ({1})",
                     share_rank_info.size(),ghost_rank_info.size());
      }
      // Trie le tableau par numéro croissant de sous-domaine.
      std::sort(std::begin(share_rank_info),std::end(share_rank_info));
      std::sort(std::begin(ghost_rank_info),std::end(ghost_rank_info));
    }

    // Ok, maintenant on connait la liste des sous-domaines qui possèdent
    // les mailles partagées de ce sous-domaine et réciproquement, la liste des
    // sous-domaines intéressé par les mailles propres de ce sous-domaine.
    // Il ne reste qu'à envoyer et recevoir les informations correspondantes.
    // Pour cela, et afin d'éviter les blocages, on envoie d'abord les infos
    // pour les sous-domaines dont le numéro est inférieur à ce sous-domaine.
    Integer nb_comm_proc = ghost_rank_info.size();
    info(4) << "Number of communicating processors: " << nb_comm_proc;
    UniqueArray<Parallel::Request> requests;
    {
      //Integer nb_recv = share_rank_info.size();

      // Trie le tableau par numéro croissant de sous-domaine.
      for( Integer i=0; i<ghost_rank_info.size(); ++i ){
        const GhostRankInfo& asdi = ghost_rank_info[i];
        debug() << "Ghost: " << i << " " << asdi.nbItem() << ' ' << asdi.rank();
      }
      for( Integer i=0; i<share_rank_info.size(); ++i ){
        const ShareRankInfo& asdi = share_rank_info[i];
        debug() << "Shared: " << i << " " << asdi.nbItem() << ' ' << asdi.rank();
      }
      Integer current_send_index = 0;
      Integer current_recv_index = 0;
      for( Integer i=0; i<nb_comm_proc*2; ++i ){
        Integer send_proc = nb_rank;
        Integer recv_proc = nb_rank;
        if (current_send_index!=nb_comm_proc)
          send_proc = ghost_rank_info[current_send_index].rank();
        if (current_recv_index!=nb_comm_proc)
          recv_proc = ghost_rank_info[current_recv_index].rank();
        bool do_send = true;
        if (send_proc==recv_proc){
          if (send_proc<my_rank)
            do_send = true;
          else
            do_send = false;
        }
        else if (send_proc<recv_proc)
          do_send = true;
        else
          do_send = false;
        if (do_send){
          GhostRankInfo& asdi = ghost_rank_info[current_send_index];
          asdi.resize();
          Int64ArrayView uids = asdi.uniqueIds();
          Int32ConstArrayView asdi_local_ids = asdi.localIds();
          //Integer zindex = 0;
          Integer nb_local = asdi_local_ids.size();
          for( Integer z=0, zs=nb_local; z<zs; ++z ){
            //for( ItemGroup::const_iter z(asdi.group()); z.hasNext(); ++z, ++zindex ){
            const Item& elem = items_internal[asdi_local_ids[z]];
            uids[z] = elem.uniqueId().asInt64(); 
          }
          if (global_debug_sync){
            info() << "Number of elements that will be sent to the subdomain " << send_proc
                   << " " << nb_local << " éléments";
            for( Integer z=0; z<nb_local; ++z ){
              info() << "Unique id " << uids[z];
            }
          }
          debug() << "Send proc " << send_proc;
          if (!uids.empty())
            requests.add(pm->send(uids,send_proc,false));
          ++current_send_index;
        }
        else{
          ShareRankInfo& asdi = share_rank_info[current_recv_index];
          asdi.resize();
          Int64ArrayView items_unique_id = asdi.uniqueIds();
          debug() << "Recv proc " << recv_proc;
          //TODO utiliser non bloquant.
          if (!items_unique_id.empty())
            pm->recv(items_unique_id,recv_proc);
          //String group_name(share_name);
          //group_name += recv_proc;

          SharedArray<Int32> items_local_id(items_unique_id.size()); //! Ids des entités du groupe
          item_family->itemsUniqueIdToLocalId(items_local_id,items_unique_id);
          SharedArray<Int32> share_group = items_local_id;
          debug() << "Creating shared entities for the subdomain " << recv_proc
                  << " with " << items_local_id.size() << " entities";
          //ItemGroup share_group = mesh->itemFamily(item_kind)->createGroup(group_name,items_local_id,true);
          //share_group.setLocalToSubDomain(true);
          asdi.setLocalIds(share_group);
          if (global_debug_sync){
            for( Integer z=0, zs=share_group.size(); z<zs; ++z ){
              const Item& item = items_internal[share_group[z]];
              info() << "Item uid=" << item.uniqueId() << ",lid=" << item.localId();
            }
          }
          ++current_recv_index;
        }
      }
      if (m_is_verbose){
        info() << "Wait requests n=" << requests.size()
               << " begin_date=" << platform::getCurrentDateTime();
      }
      pm->waitAllRequests(requests);
      if (m_is_verbose){
        info() << "Wait requests end_date=" << platform::getCurrentDateTime();
      }
    }
  }
  _checkValid(ghost_rank_info,share_rank_info);
  m_sync_list->recompute();

  // Calcul de m_communicating_ranks qui synthétisent les processeurs communiquants
  for( Integer i=0, n=m_sync_list->size(); i<n; ++i ){
    Int32 target_rank = m_sync_list->targetRank(i);
    m_communicating_ranks.add(target_rank);
    if (m_sync_list->receiveInfo().nbItem(i) != boundary_items[target_rank].size())
      ARCANE_FATAL("Inconsistent ghost count");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_checkValid(ArrayView<GhostRankInfo> ghost_rank_info,
            ArrayView<ShareRankInfo> share_rank_info)
{
  Integer nb_comm_proc = ghost_rank_info.size();
  Integer nb_error  = 0;
  bool has_error = false;
  const Integer max_error = 10; // Nombre max d'erreurs affichées.
  Int32 my_rank = m_parallel_mng->commRank();
  IItemFamily* item_family = m_item_group.itemFamily();
  ItemInfoListView items_internal(item_family);

  // Tableau servant à marquer les éléments qui sont soit
  // propres au sous-domaine, soit fantômes.
  // Normalement, si les données sont cohérentes cela devrait marquer
  // tous les éléments.
  // NOTE: ceci n'est utile que si \a itemGroup() vaut allItems()
  UniqueArray<bool> marked_elem(item_family->maxLocalId());
  marked_elem.fill(false);
  // Marque les éléments propres au sous-domaine
  ENUMERATE_ITEM(i_item,m_item_group){
    Item item = *i_item;
    if (item.isOwn()) {
      marked_elem[item.localId()] = true;
      if (global_debug_sync){
        info() << "Own Item " << ItemPrinter(item);
      }
    }
  }

  for( Integer i_comm=0; i_comm<nb_comm_proc; ++i_comm ){
    GhostRankInfo& ghost_info = ghost_rank_info[i_comm];
    ShareRankInfo& share_info = share_rank_info[i_comm];
    if (ghost_info.rank()!=share_info.rank()){
      ARCANE_FATAL("Inconsistency between the subdomain numbers ghost_rank={0} share_rank={1}",
                   ghost_info.rank(),share_info.rank());
    }
    Integer current_proc = ghost_info.rank();
    Int32ConstArrayView ghost_grp = ghost_info.localIds();
    Int32ConstArrayView share_grp = share_info.localIds();

    if (share_grp.empty() && ghost_grp.empty()){
      error() << "Shared and ghosts groups null for the subdomain " << current_proc;
      has_error = true;
      continue;
    }
    if (current_proc==my_rank){
      error() << "Error in the communication pattern: "
              << "the processor can't communicate with itself";
      has_error = true;
      continue;
    }

    // Marque les éléments du groupe partagé
    for( Integer z=0, zs=ghost_grp.size(); z<zs; ++z ){
      const Item& elem = items_internal[ghost_grp[z]];
      bool is_marked = marked_elem[elem.localId()];
      if (is_marked){
        // L'élément ne doit pas déjà être marqué.
        if (nb_error<max_error)
          error() << "The entity " << ItemPrinter(elem) << " belongs to another ghost group "
                  << "or is owned by the subdomain.";
        ++nb_error;
        continue;
      }
      marked_elem[elem.localId()] = true;
    }

    m_sync_list->add( VariableSyncInfo (share_grp,ghost_grp,current_proc) );
  }

  // Vérifie que tous les éléments sont marqués
  ENUMERATE_ITEM(i,m_item_group){
    Item item = *i;
    if (!marked_elem[item.localId()]){
      if (nb_error<max_error){
        error() << "The entity " << ItemPrinter(item)
                << " doesn't belong to the subdomain or any ghost group.";
      }
      ++nb_error;
    }
  }

  // En cas d'erreur, on s'arrête.
  if (nb_error!=0){
    has_error = true;
    if (nb_error>=max_error)
      error() << nb_error << " total elements are incorrectly dealt with";
  }
  if (has_error)
    ARCANE_FATAL("Error while creating the exchange structures of the family={0}",
                 item_family->fullName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_printSyncList()
{
  Integer nb_comm = m_sync_list->size();
  info() << "SYNC LIST FOR GROUP : " << m_item_group.fullName() << " N=" << nb_comm;
  OStringStream ostr;
  IItemFamily* item_family = m_item_group.itemFamily();
  ItemInfoListView items_internal(item_family);
  for( Integer i=0; i<nb_comm; ++i ){
    Int32 target_rank = m_sync_list->targetRank(i);
    ostr() << " TARGET=" << target_rank << '\n';
    Int32ConstArrayView share_ids = m_sync_list->sendInfo().localIds(i);
    ostr() << "\t\tSHARE(lid,uid) n=" << share_ids.size() << " :";
    for( Integer z=0, zs=share_ids.size(); z<zs; ++z ){
      Item item = items_internal[share_ids[z]];
      ostr() << " (" << item.localId() << "," << item.uniqueId() << ")";
    }
    ostr() << "\n";
    Int32ConstArrayView ghost_ids = m_sync_list->receiveInfo().localIds(i);
    ostr() << "\t\tGHOST(lid,uid) n=" << ghost_ids.size() << " :";
    for( Integer z=0, zs=ghost_ids.size(); z<zs; ++z ){
      Item item = items_internal[ghost_ids[z]];
      ostr() << " (" << item.localId() << "," << item.uniqueId() << ")";
    }
    ostr() << "\n";
  }
  info() << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronize(IVariable* var)
{
  IParallelMng* pm = m_parallel_mng;
  ITimeStats* ts = pm->timeStats();
  Timer::Phase tphase(ts,TP_Communication);

  debug(Trace::High) << " Proc " << pm->commRank() << " Sync variable " << var->fullName();
  if (m_trace_sync){
    info() << " Synchronize variable " << var->fullName()
           << " stack=" << platform::getStackTrace();
  }
  // Debut de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(var,this);
    m_on_synchronized.notify(args);
  }
  if (!m_sync_timer)
    m_sync_timer = new Timer(pm->timerMng(),"SyncTimer",Timer::TimerReal);
  {
    Timer::Sentry ts2(m_sync_timer);
    _synchronize(var);
  }
  Real elapsed_time = m_sync_timer->lastActivationTime();
  pm->stat()->add("Synchronize",elapsed_time,1);
  // Fin de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(var,this,elapsed_time);
    m_on_synchronized.notify(args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronize(VariableCollection vars)
{
  if (vars.empty())
    return;

  const bool use_multi = m_allow_multi_sync;
  if (use_multi && _canSynchronizeMulti(vars)){
    _synchronizeMulti(vars);
  }
  else{
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      synchronize(*ivar);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataSynchronizeResult VariableSynchronizer::
_synchronize(INumericDataInternal* data)
{
  m_dispatcher->beginSynchronize(data, m_is_compare_sync);
  return m_dispatcher->endSynchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_synchronize(IVariable* var)
{
  ARCANE_CHECK_POINTER(var);
  INumericDataInternal* numapi = var->data()->_commonInternal()->numericData();
  if (!numapi)
    ARCANE_FATAL("Variable '{0}' can not be synchronized because it is not a numeric data",var->name());
  DataSynchronizeResult result = _synchronize(numapi);
  eDataSynchronizeCompareStatus s = result.compareStatus();
  if (m_is_compare_sync){
    if (s==eDataSynchronizeCompareStatus::Different)
      info() << "Different values name=" << var->name();
    else if (s==eDataSynchronizeCompareStatus::Same)
      info() << "Same values name=" << var->name();
    else
      info() << "Unknown values name=" << var->name();
  }
  var->setIsSynchronized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronizeData(IData* data)
{
  ARCANE_CHECK_POINTER(data);
  INumericDataInternal* numapi = data->_commonInternal()->numericData();
  if (!numapi)
    ARCANE_FATAL("Data can not be synchronized because it is not a numeric data");
  _synchronize(numapi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  info(4) << "** VariableSynchronizer::changeLocalIds() group=" << m_item_group.name();
  m_sync_list->changeLocalIds(old_to_new_ids);
  m_dispatcher->compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indique si les variables de la liste \a vars peuvent être synchronisées
 * en une seule fois.
 *
 * Pour que cela soit possible, il faut que ces variables ne soient pas
 * partielles et reposent sur le même ItemGroup (donc soient de la même famille)
 */
bool VariableSynchronizer::
_canSynchronizeMulti(const VariableCollection& vars)
{
  if (vars.count()==1)
    return false;
  ItemGroup group;
  bool is_set = false;
  for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
    IVariable* var = *ivar;
    if (var->isPartial())
      return false;
    ItemGroup var_group = var->itemGroup();
    if (!is_set){
      group = var_group;
      is_set = true;
    }
    if (group!=var_group)
      return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_synchronizeMulti(VariableCollection vars)
{
  IParallelMng* pm = m_parallel_mng;
  ITimeStats* ts = pm->timeStats();
  Timer::Phase tphase(ts,TP_Communication);

  debug(Trace::High) << " Proc " << pm->commRank() << " MultiSync variable";
  if (m_trace_sync){
    info() << " MultiSynchronize"
           << " stack=" << platform::getStackTrace();
  }
  // Debut de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(vars,this);
    m_on_synchronized.notify(args);
  }
  if (!m_sync_timer)
    m_sync_timer = new Timer(pm->timerMng(),"SyncTimer",Timer::TimerReal);
  {
    Timer::Sentry ts2(m_sync_timer);
    m_multi_dispatcher->synchronize(vars);
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      (*ivar)->setIsSynchronized();
    }
  }
  Real elapsed_time = m_sync_timer->lastActivationTime();
  pm->stat()->add("MultiSynchronize",elapsed_time,1);
  // Fin de la synchro
  if (m_on_synchronized.hasObservers()){
    VariableSynchronizerEventArgs args(vars,this,elapsed_time);
    m_on_synchronized.notify(args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
communicatingRanks()
{
  return m_communicating_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
sharedItems(Int32 index)
{
  return m_sync_list->sendInfo().localIds(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
ghostItems(Int32 index)
{
  return m_sync_list->receiveInfo().localIds(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
