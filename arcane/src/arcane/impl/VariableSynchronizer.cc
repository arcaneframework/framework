// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizer.cc                                     (C) 2000-2018 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/NotImplementedException.h"
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
#include "arcane/utils/ScopedPtr.h"

#include "arcane/impl/VariableSynchronizer.h"
#include "arcane/VariableSynchronizerEventArgs.h"
#include "arcane/IParallelMng.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IVariable.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/IData.h"
#include "arcane/VariableCollection.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/ISerializer.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/Timer.h"
#include "arcane/parallel/IStat.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
applyDispatch(IArrayDataT<SimpleType>* data)
{
  this->beginSynchronize(data->value(),m_1d_buffer);
  this->endSynchronize(data->value(),m_1d_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
  
template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
_copyFromBuffer(Int32ConstArrayView indexes,ConstArrayView<SimpleType> buffer,
                ArrayView<SimpleType> var_value,Integer dim2_size)
{
  if (dim2_size==1)
    m_buffer_copier->copyFromBufferOne(indexes,buffer,var_value);
  else
    m_buffer_copier->copyFromBufferMultiple(indexes,buffer,var_value,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
_copyToBuffer(Int32ConstArrayView indexes,ArrayView<SimpleType> buffer,
              ConstArrayView<SimpleType> var_value,Integer dim2_size)
{
  if (dim2_size==1)
    m_buffer_copier->copyToBufferOne(indexes,buffer,var_value);
  else
    m_buffer_copier->copyToBufferMultiple(indexes,buffer,var_value,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
applyDispatch(IArray2DataT<SimpleType>* data)
{
  Array2<SimpleType>& value = data->value();
  SimpleType* value_ptr = value.view().unguardedBasePointer();
  // Cette valeur doit être la même sur tous les procs
  Integer dim2_size = value.dim2Size();
  if (dim2_size==0)
    return;
  Integer dim1_size = value.dim1Size();
  m_2d_buffer.compute(m_sync_list,dim2_size);
  ArrayView<SimpleType> buf(dim1_size*dim2_size,value_ptr);
  this->beginSynchronize(buf,m_2d_buffer);
  this->endSynchronize(buf,m_2d_buffer);
  //TODO: liberer la memoire si besoin ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 * \todo: ne pas allouer les tampons car leur conservation est couteuse en
 * terme de memoire.
 */
template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::SyncBuffer::
compute(ConstArrayView<VariableSyncInfo> sync_list,Integer dim2_size)
{
  m_dim2_size = dim2_size;
  Integer nb_message = sync_list.size();

  m_ghost_locals_buffer.resize(nb_message);
  m_share_locals_buffer.resize(nb_message);

  Integer total_ghost_buffer = 0;
  Integer total_share_buffer = 0;
  for( Integer i=0; i<nb_message; ++i ){
    total_ghost_buffer += sync_list[i].m_ghost_ids.size();
    total_share_buffer += sync_list[i].m_share_ids.size();
  }
  m_ghost_buffer.resize(total_ghost_buffer*dim2_size);
  m_share_buffer.resize(total_share_buffer*dim2_size);
    
  {
    Integer array_index = 0;
    for( Integer i=0, is=sync_list.size(); i<is; ++i ){
      const VariableSyncInfo& vsi = sync_list[i];
      Int32ConstArrayView ghost_grp = vsi.m_ghost_ids;
      Integer local_size = ghost_grp.size();
      m_ghost_locals_buffer[i] = ArrayView<SimpleType>();
      if (local_size!=0)
        m_ghost_locals_buffer[i] = ArrayView<SimpleType>(local_size*dim2_size,
                                                         &m_ghost_buffer[array_index*dim2_size]);
      array_index += local_size;
    }
  }
  {
    Integer array_index = 0;
    for( Integer i=0, is=sync_list.size(); i<is; ++i ){
      const VariableSyncInfo& vsi = sync_list[i];
      Int32ConstArrayView share_grp = vsi.m_share_ids;
      Integer local_size = share_grp.size();
      m_share_locals_buffer[i] = ArrayView<SimpleType>();
      if (local_size!=0)
        m_share_locals_buffer[i] = ArrayView<SimpleType>(local_size*dim2_size,
                                                         &m_share_buffer[array_index*dim2_size]);
      array_index += local_size;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul et alloue les tampons nécessaire aux envois et réceptions
 * pour les synchronisations des variables 1D.
 */
template<typename SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
compute(ConstArrayView<VariableSyncInfo> sync_list)
{
  //IParallelMng* pm = m_parallel_mng;
  m_sync_list = sync_list;
  //Integer nb_message = sync_list.size();
  //pm->traceMng()->info() << "** RECOMPUTE SYNC LIST!!! N=" << nb_message
  //                       << " this=" << (IVariableSynchronizeDispatcher*)this
  //                       << " m_sync_list=" << &m_sync_list;

  m_1d_buffer.compute(sync_list,1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
beginSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (m_is_in_sync)
    ARCANE_FATAL("Only one pending serialisation is supported");
  
  IParallelMng* pm = m_parallel_mng;
  
  //ITraceMng* trace = pm->traceMng();
  //Integer nb_elem = var_values.size();
  bool use_blocking_send = false;
  Integer nb_message = m_sync_list.size();
  Integer dim2_size = sync_buffer.m_dim2_size;

  /*pm->traceMng()->info() << " ** ** COMMON BEGIN SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/

  //SyncBuffer& sync_buffer = m_1d_buffer;
  // Envoie les messages de réception non bloquant
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[i];
    if (!ghost_local_buffer.empty()){
      Parallel::Request rval = pm->recv(ghost_local_buffer,vsi.m_target_rank,false);
      m_all_requests.add(rval);
    }
  }

  // Envoie les messages d'envoie en mode non bloquant.
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    Int32ConstArrayView share_grp = vsi.m_share_ids;
    ArrayView<SimpleType> share_local_buffer = sync_buffer.m_share_locals_buffer[i];
      
    _copyToBuffer(share_grp,share_local_buffer,var_values,dim2_size);

    ConstArrayView<SimpleType> const_share = share_local_buffer;
    if (!share_local_buffer.empty()){
      //for( Integer i=0, is=share_local_buffer.size(); i<is; ++i )
      //trace->info() << "TO rank=" << vsi.m_target_rank << " I=" << i << " V=" << share_local_buffer[i]
      //                << " lid=" << share_grp[i] << " v2=" << var_values[share_grp[i]];
      Parallel::Request rval = pm->send(const_share,vsi.m_target_rank,use_blocking_send);
      if (!use_blocking_send)
        m_all_requests.add(rval);
    }
  }
  m_is_in_sync = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType> void VariableSynchronizeDispatcher<SimpleType>::
endSynchronize(ArrayView<SimpleType> var_values,SyncBuffer& sync_buffer)
{
  if (!m_is_in_sync)
    ARCANE_FATAL("endSynchronize() called but no beginSynchronize() was called before");

  IParallelMng* pm = m_parallel_mng;
  
  //ITraceMng* trace = pm->traceMng();
  //Integer nb_elem = var_values.size();
  //bool use_blocking_send = false;
  Integer nb_message = m_sync_list.size();
  Integer dim2_size = sync_buffer.m_dim2_size;

  /*pm->traceMng()->info() << " ** ** COMMON END SYNC n=" << nb_message
                         << " this=" << (IVariableSynchronizeDispatcher*)this
                         << " m_sync_list=" << &this->m_sync_list;*/


  // Attend que les receptions se terminent
  pm->waitAllRequests(m_all_requests);
  m_all_requests.clear();

  // Recopie dans la variable le message de retour.
  for( Integer i=0; i<nb_message; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    Int32ConstArrayView ghost_grp = vsi.m_ghost_ids;
    ArrayView<SimpleType> ghost_local_buffer = sync_buffer.m_ghost_locals_buffer[i];
    _copyFromBuffer(ghost_grp,ghost_local_buffer,var_values,dim2_size);
    //for( Integer i=0, is=ghost_local_buffer.size(); i<is; ++i )
    //trace->info() << "RECV rank=" << vsi.m_target_rank << " I=" << i << " V=" << ghost_local_buffer[i]
    //                << " lid=" << ghost_grp[i] << " v2=" << var_values[ghost_grp[i]];
  }

  m_is_in_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerMultiDispatcher
{
 public:
  VariableSynchronizerMultiDispatcher(IParallelMng* pm)
  : m_parallel_mng(pm)
  {
  }

  void synchronize(VariableCollection vars,ConstArrayView<VariableSyncInfo> sync_infos)
  {
    ScopedPtrT<IParallelExchanger> exchanger(m_parallel_mng->createExchanger());
    Integer nb_rank = sync_infos.size();
    Int32UniqueArray recv_ranks(nb_rank);
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = sync_infos[i].m_target_rank;
      exchanger->addSender(rank);
      recv_ranks[i] = rank;
    }
    exchanger->initializeCommunicationsMessages(recv_ranks);
    for( Integer i=0; i<nb_rank; ++i ){
      ISerializeMessage* msg = exchanger->messageToSend(i);
      ISerializer* sbuf = msg->serializer();
      Int32ConstArrayView share_ids = sync_infos[i].m_share_ids;
      sbuf->setMode(ISerializer::ModeReserve);
      for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
        (*ivar)->serialize(sbuf,share_ids,0);
      }
      sbuf->allocateBuffer();
      sbuf->setMode(ISerializer::ModePut);
      for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
        (*ivar)->serialize(sbuf,share_ids,0);
      }
    }
    exchanger->processExchange();
    for( Integer i=0; i<nb_rank; ++i ){
      ISerializeMessage* msg = exchanger->messageToReceive(i);
      ISerializer* sbuf = msg->serializer();
      Int32ConstArrayView ghost_ids = sync_infos[i].m_ghost_ids;
      sbuf->setMode(ISerializer::ModeGet);
      for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
        (*ivar)->serialize(sbuf,ghost_ids,0);
      }
    }
  }

 private:
  IParallelMng* m_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
VariableSynchronizer(IParallelMng* pm,const ItemGroup& group,
                     VariableSynchronizerDispatcher* dispatcher)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_item_group(group)
, m_dispatcher(dispatcher)
, m_multi_dispatcher()
, m_sync_timer(0)
, m_is_verbose(false)
, m_allow_multi_sync(true)
, m_trace_sync(false)
{
  typedef DataTypeDispatchingDataVisitor<IVariableSynchronizeDispatcher> DispatcherType;
  if (!m_dispatcher){
    VariableSynchronizeDispatcherBuildInfo bi(pm,nullptr);
    DispatcherType* dt = DispatcherType::create<VariableSynchronizeDispatcher>(bi);
    m_dispatcher = new VariableSynchronizerDispatcher(pm,dt);
  }
  m_multi_dispatcher = new VariableSynchronizerMultiDispatcher(pm);

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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizer::
~VariableSynchronizer()
{
  delete m_sync_timer;
  delete m_multi_dispatcher;
  delete m_dispatcher;
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
      const Item& item = *i;
      ItemInternal* item_internal = item.internal();
      Integer owner = item_internal->owner();
      if (owner==my_rank)
        continue;
      Int64 uid = item_internal->uniqueId().asInt64();
      if (owner==A_NULL_RANK || owner>=nb_rank){
        if (nb_error<10)
          bad_items_uid.add(uid);
        continue;
      }
      if (global_debug_sync){
        info() << "Add entity uid=" << uid
               << " lid=" << item_internal->localId() << " to the subdomain " << owner;
      }
      boundary_items[owner].add(item_internal->localId());
    }
    if (nb_error!=0){
      for( Integer i=0, is=bad_items_uid.size(); i<is; ++i ){
        info() << "ERROR: The entity uid=" << bad_items_uid[i]
               << " group=" << m_item_group.name() << " doesn't belong to "
               << "any subdomain or belong to an invalid subdomain";
      }
      fatal() << "Error while creating synchronization information";
    }
  }

  _createList(boundary_items);

  //_printSyncList();
  
  if (m_is_verbose){
    info() << "Begin compute dispatcher Date=" << platform::getCurrentDateTime();
  }
  m_dispatcher->compute(m_sync_list);
  if (m_is_verbose){
    info() << "End compute dispatcher Date=" << platform::getCurrentDateTime();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
_createList(UniqueArray<SharedArray<Int32> >& boundary_items)
{
  m_sync_list.clear();

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
      ItemInternalList items_internal = item_family->itemsInternal();
      for( Integer z=0, zs=boundary_items[i].size(); z<zs; ++z ){
        const Item& item = items_internal[boundary_items[i][z]];
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
  ItemInternalList items_internal = item_family->itemsInternal();
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
        fatal() << "Problem with the number of subdomain shared ("
                     << share_rank_info.size() << ") and ghosts ("
                     << ghost_rank_info.size() << ")";
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

  // Calcul de m_communicating_ranks qui synthétisent les processeurs communiquants
  for( Integer i=0; i<m_sync_list.size(); ++i ){
    const VariableSyncInfo & sync_info = m_sync_list[i];
    const Integer target_rank = sync_info.targetRank();
    m_communicating_ranks.add(target_rank);
    ARCANE_ASSERT((sync_info.m_ghost_ids.size() == boundary_items[target_rank].size()),("Inconsistent ghost count"));
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
  ItemInternalList items_internal = item_family->itemsInternal();

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

    // La synchronisation se fait de telle manière que le processeur
    // de numéro le plus faible envoie d'abord ses informations et les
    // recoient ensuite.
    bool is_send_first = current_proc < my_rank;
    m_sync_list.add( VariableSyncInfo (share_grp,ghost_grp,current_proc,is_send_first) );
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
  Integer nb_comm = m_sync_list.size();
  info() << "SYNC LIST FOR GROUP : " << m_item_group.fullName() << " N=" << nb_comm;
  OStringStream ostr;
  IItemFamily* item_family = m_item_group.itemFamily();
  ItemInternalList items_internal = item_family->itemsInternal();
  for( Integer i=0; i<nb_comm; ++i ){
    const VariableSyncInfo& vsi = m_sync_list[i];
    ostr() << " TARGET=" << vsi.m_target_rank << '\n';
    ostr() << "\t\tSHARE(lid,uid) n=" << vsi.m_share_ids.size() << " :";
    for( Integer z=0, zs=vsi.m_share_ids.size(); z<zs; ++z ){
      ItemInternal* item = items_internal[vsi.m_share_ids[z]];
      ostr() << " (" << item->localId() << "," << item->uniqueId() << ")";
    }
    ostr() << "\n";
    ostr() << "\t\tGHOST(lid,uid) n=" << vsi.m_ghost_ids.size() << " :";
    for( Integer z=0, zs=vsi.m_ghost_ids.size(); z<zs; ++z ){
      ItemInternal* item = items_internal[vsi.m_ghost_ids[z]];
      ostr() << " (" << item->localId() << "," << item->uniqueId() << ")";
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
    Timer::Sentry ts(m_sync_timer);
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
    Timer::Sentry ts(m_sync_timer);
    _synchronize(vars);
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

void VariableSynchronizer::
_synchronize(IVariable* var)
{
  var->data()->visit(m_dispatcher->visitor());
  var->setIsSynchronized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
synchronizeData(IData* data)
{
  data->visit(m_dispatcher->visitor());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizer::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  info(4) << "** VariableSynchronizer::changeLocalIds() group=" << m_item_group.name();
  Integer nb_comm = m_sync_list.size();
  for( Integer i=0; i<nb_comm; ++i ){
    VariableSyncInfo& vsi = m_sync_list[i];

    UniqueArray<Int32> orig_share_ids(vsi.m_share_ids);
    Int32Array& share_ids = vsi.m_share_ids;
    share_ids.clear();

    for( Integer z=0, zs=orig_share_ids.size(); z<zs; ++z ){
      Int32 old_id = orig_share_ids[z];
      Int32 new_id = old_to_new_ids[old_id];
      //info() << "SHARE ID=" << old_id << " NEW=" << new_id;
      if (new_id!=NULL_ITEM_LOCAL_ID)
        share_ids.add(new_id);
    }
    info(4) << "NEW_SHARE_SIZE=" << share_ids.size() << " old=" << orig_share_ids.size();

    UniqueArray<Int32> orig_ghost_ids(vsi.m_ghost_ids);
    Int32Array& ghost_ids = vsi.m_ghost_ids;
    ghost_ids.clear();

    for( Integer z=0, zs=orig_ghost_ids.size(); z<zs; ++z ){
      Int32 old_id = orig_ghost_ids[z];
      Int32 new_id = old_to_new_ids[old_id];
      //info() << "GHOST ID=" << old_id << " NEW=" << new_id;
      if (new_id!=NULL_ITEM_LOCAL_ID)
        ghost_ids.add(new_id);
    }
    info(4) << "NEW_GHOST_SIZE=" << ghost_ids.size() << " old=" << orig_ghost_ids.size();
  }
  m_dispatcher->compute(m_sync_list);
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
_synchronize(VariableCollection vars)
{
  if (vars.empty())
    return;

  const bool use_multi = m_allow_multi_sync;
  if (use_multi && _canSynchronizeMulti(vars)){
    m_multi_dispatcher->synchronize(vars,m_sync_list);
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      (*ivar)->setIsSynchronized();
    }
  }
  else{
    for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
      _synchronize(*ivar);
    }
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
  return m_sync_list[index].m_share_ids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VariableSynchronizer::
ghostItems(Int32 index)
{
  return m_sync_list[index].m_ghost_ids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class VariableSynchronizeDispatcher<Byte>;
template class VariableSynchronizeDispatcher<Real>;
template class VariableSynchronizeDispatcher<Int16>;
template class VariableSynchronizeDispatcher<Int32>;
template class VariableSynchronizeDispatcher<Int64>;
template class VariableSynchronizeDispatcher<Real2>;
template class VariableSynchronizeDispatcher<Real3>;
template class VariableSynchronizeDispatcher<Real2x2>;
template class VariableSynchronizeDispatcher<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
