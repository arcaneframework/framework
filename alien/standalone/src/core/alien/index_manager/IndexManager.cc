/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <list>
#include <map>
#include <utility>
#include <vector>

#include <arccore/message_passing/BasicSerializeMessage.h>
#include <arccore/message_passing/ISerializeMessageList.h>
#include <arccore/message_passing/Messages.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <arccore/message_passing_mpi/MpiSerializeMessageList.h>

#include <alien/index_manager/IAbstractFamily.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/Trace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct IndexManager::EntryLocalId
{
  explicit EntryLocalId(Alien::Integer size)
  : m_is_defined(size, false)
  {}

  void reserveLid(const Integer count)
  {
    m_defined_lids.reserve(m_defined_lids.size() + count);
  }

  [[nodiscard]] bool isDefinedLid(const Integer localId) const
  {
    return m_is_defined[localId];
  }

  void defineLid(const Integer localId, const Integer pos)
  {
    m_is_defined[localId] = true;
    Alien::add(m_defined_lids, std::make_pair(localId, pos));
  }

  void undefineLid(const Integer localId)
  {
    m_is_defined[localId] = false;
    for (Integer i = 0; i < m_defined_lids.size(); ++i) {
      if (m_defined_lids[i].first == localId) {
        m_defined_lids[i] = m_defined_lids.back();
        m_defined_lids.resize(m_defined_lids.size() - 1);
        return;
      }
    }
    throw FatalErrorException(
    A_FUNCINFO, "Inconsistent state : cannot find id to remove");
  }

  [[nodiscard]] const UniqueArray<std::pair<Integer, Integer>>& definedLids() const
  {
    return m_defined_lids;
  }

  void freeDefinedLids()
  {
    Alien::freeData(m_defined_lids);
    std::vector<bool>().swap(m_is_defined);
  }

  std::vector<bool> m_is_defined;
  UniqueArray<std::pair<Integer, Integer>> m_defined_lids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct IndexManager::EntrySendRequest
{
  EntrySendRequest() = default;

  ~EntrySendRequest() = default;

  EntrySendRequest(const EntrySendRequest& esr)
  : count(esr.count)
  {}

  Arccore::Ref<Arccore::MessagePassing::ISerializeMessage> comm;
  Integer count = 0;

  void operator=(const EntrySendRequest&) = delete;
};

/*---------------------------------------------------------------------------*/

struct IndexManager::EntryRecvRequest
{
  EntryRecvRequest() = default;

  ~EntryRecvRequest() = default;

  explicit EntryRecvRequest(const EntrySendRequest& err) {}

  Arccore::Ref<Arccore::MessagePassing::ISerializeMessage> comm;
  UniqueArray<Int64> ids;

  void operator=(const EntryRecvRequest&) = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexManager::IndexManager(
Alien::IMessagePassingMng* parallelMng, Alien::ITraceMng* traceMng)
: m_parallel_mng(parallelMng)
, m_trace_mng(traceMng)
, m_local_owner(0)
, m_state(Undef)
, m_verbose(false)
, m_local_entry_count(0)
, m_global_entry_count(0)
, m_global_entry_offset(0)
, m_local_removed_entry_count(0)
, m_global_removed_entry_count(0)
, m_max_null_index_opt(false)
{
  this->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexManager::~IndexManager()
{
  this->init();
}

/*---------------------------------------------------------------------------*/

void IndexManager::init()
{
  m_local_owner = m_parallel_mng->commRank();

  m_state = Initialized;

  m_local_entry_count = 0;
  m_global_entry_count = 0;
  m_global_entry_offset = 0;
  m_local_removed_entry_count = 0;
  m_global_removed_entry_count = 0;

  // Destruction des structure de type entry
  for (auto& m_entrie : m_entries) {
    delete m_entrie;
  }
  m_entries.clear();

  m_abstract_families.clear();
}

/*---------------------------------------------------------------------------*/

void IndexManager::setVerboseMode(bool verbose)
{
  m_verbose = verbose;
}

/*---------------------------------------------------------------------------*/

ScalarIndexSet
IndexManager::buildEntry(
const String& name, const IAbstractFamily* family, const Integer kind)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  for (auto* e : m_entries) {
    if (name == e->getName())
      throw FatalErrorException(A_FUNCINFO, "Already defined entry");
  }

  const Integer uid = m_entries.size();

  m_entry_families[uid] = family;

  auto* entry = new ScalarIndexSet(name, uid, this, kind);

  m_entries.push_back(entry);

  return *entry;
}

/*---------------------------------------------------------------------------*/

void IndexManager::defineIndex(const ScalarIndexSet& entry, ConstArrayView<Integer> localIds)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));

  const Integer uid = entry.getUid();

  const auto* family = m_entry_families[uid];

  auto entry_local_ids = std::make_shared<EntryLocalId>(family->maxLocalId());

  m_entry_local_ids[uid] = entry_local_ids;

  auto owners = family->owners(localIds);

  entry_local_ids->reserveLid(localIds.size());
  for (Integer i = 0, is = localIds.size(); i < is; ++i) {
    const Integer localId = localIds[i];
    if (not entry_local_ids->isDefinedLid(localId)) { // nouvelle entrée
      if (owners[i] == m_local_owner) {
        entry_local_ids->defineLid(
        localId, +(m_local_removed_entry_count + m_local_entry_count++));
      }
      else {
        entry_local_ids->defineLid(
        localId, -(m_global_removed_entry_count + (++m_global_entry_count)));
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void IndexManager::removeIndex(const ScalarIndexSet& entry, ConstArrayView<Integer> localIds)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));

  const Integer uid = entry.getUid();

  const auto* family = m_entry_families[uid];

  auto entry_local_ids = m_entry_local_ids[uid];

  const auto owners = family->owners(localIds);

  for (Integer localId : localIds) {
    if (entry_local_ids->isDefinedLid(localId)) {
      entry_local_ids->undefineLid(localId);
      if (owners[localId] == m_local_owner) {
        --m_local_entry_count;
        ++m_local_removed_entry_count;
      }
      else {
        --m_global_entry_count;
        ++m_global_removed_entry_count;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void IndexManager::prepare()
{
  EntryIndexMap entry_index;

  begin_prepare(entry_index);

  if (m_parallel_mng->commSize() > 1) {
    begin_parallel_prepare(entry_index);
    end_parallel_prepare(entry_index);
  }
  else {
    sequential_prepare(entry_index);
  }

  end_prepare(entry_index);
}

/*---------------------------------------------------------------------------*/

void IndexManager::begin_prepare(EntryIndexMap& entry_index)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  Integer total_size = 0;
  for (auto* entry : m_entries) {
    auto entry_local_ids = m_entry_local_ids[entry->getUid()];
    total_size += entry_local_ids->definedLids().size();
  }

  entry_index.reserve(total_size);

  for (auto* entry : m_entries) {
    const Integer entry_uid = entry->getUid();
    const auto* family = m_entry_families[entry_uid];
    const Integer entry_kind = entry->getKind();
    auto entry_local_ids = m_entry_local_ids[entry_uid];
    const auto& lids = entry_local_ids->definedLids();
    for (Integer i = 0, is = lids.size(); i < is; ++i) {
      const Integer item_localid = lids[i].first;
      auto item = family->item(item_localid);
      entry_index.push_back(InternalEntryIndex{ entry_uid, entry_kind, item.uniqueId(),
                                                item_localid, lids[i].second, item.owner() });
    }
    entry_local_ids->freeDefinedLids();
  }

  m_entry_local_ids.clear();

  // Tri par défaut
  std::sort(entry_index.begin(), entry_index.end(),
            [](const InternalEntryIndex& a, const InternalEntryIndex& b) {
              if (a.m_entry_kind != b.m_entry_kind)
                return a.m_entry_kind < b.m_entry_kind;
              else if (a.m_item_uid != b.m_item_uid)
                return a.m_item_uid < b.m_item_uid;
              else
                return a.m_entry_uid < b.m_entry_uid;
            });
  ALIEN_ASSERT(
  ((Integer)entry_index.size() == m_local_entry_count + m_global_entry_count),
  ("Inconsistent global size"));
}

/*---------------------------------------------------------------------------*/

void IndexManager::end_prepare(EntryIndexMap& entryIndex)
{
  // Calcul de la taille des indices par entrée
  std::map<Integer, Integer> count_table;
  for (auto& i : entryIndex) {
    count_table[i.m_entry_uid]++;
  }

  auto isOwn = [&](const InternalEntryIndex& i) -> bool {
    return i.m_item_owner == m_local_owner;
  };

  // Dimensionnement des buffers de chaque entrée
  for (auto* entry : m_entries) {
    const Integer uid = entry->getUid();
    const Integer size = count_table[uid];
    auto& all_items = m_entry_all_items[uid];
    auto& all_indices = m_entry_all_indices[uid];

    all_items.resize(size);
    all_indices.resize(size);

    Integer own_i = 0;
    Integer ghost_i = size;
    for (auto& i : entryIndex) {
      if (i.m_entry_uid == uid) {
        const Integer local_id = i.m_item_localid;
        const Integer index = i.m_item_index;
        const bool is_own = isOwn(i);
        if (is_own) {
          all_items[own_i] = local_id;
          all_indices[own_i] = index;
          ++own_i;
        }
        else {
          --ghost_i;
          all_items[ghost_i] = local_id;
          all_indices[ghost_i] = index;
        }
      }
    }

    const Integer own_size = own_i;
    ALIEN_ASSERT((own_i == ghost_i), ("Not merged insertion"));

    m_entry_own_items[uid] = ConstArrayView<Integer>(own_size, &all_items[0]);
    m_entry_own_indices[uid] = ConstArrayView<Integer>(own_size, &all_indices[0]);
  }

  m_state = Prepared;

  if (m_verbose) {
    alien_info([&] {
      cout() << "Entry ordering :";
      for (auto* entry : m_entries) {
        cout() << "\tEntry '" << entry->getName() << "' placed at rank "
               << entry->getUid() << " with " << getOwnLocalIds(*entry).size()
               << " local / " << getAllLocalIds(*entry).size() << " global indexes ";
      }
      cout() << "Total local Entry indexes = " << m_local_entry_count;
    });
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct IndexManager::ParallelRequests
{
  // Structure pour accumuler et structurer la collecte de l'information
  typedef std::map<Integer, EntrySendRequest> SendRequestByEntry;
  typedef std::map<Integer, SendRequestByEntry> SendRequests;
  SendRequests sendRequests;

  // Table des requetes exterieures (reçoit les uid et renverra les EntryIndex finaux)
  typedef std::list<EntryRecvRequest> RecvRequests;
  RecvRequests recvRequests;

  Alien::Ref<ISerializeMessageList> messageList;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IndexManager::begin_parallel_prepare(EntryIndexMap& entry_index)
{

  ALIEN_ASSERT((m_parallel_mng->commSize() > 1), ("Parallel mode expected"));

  /* Algorithme:
   * 1 - listing des couples Entry-Item non locaux
   * 2 - Envoi vers les propriétaires des items non locaux
   * 3 - Prise en compte éventuelle de nouvelles entrées
   * 4 - Nommage locales
   * 5 - Retour vers demandeurs des EntryIndex non locaux
   * 6 - Finalisation de la numérotation (table reindex)
   */

  parallel = std::make_shared<ParallelRequests>();

  // 1 - Comptage des Items non locaux
  for (auto& entryIndex : entry_index) {
    const Integer item_owner = entryIndex.m_item_owner;
    if (item_owner != m_local_owner) {
      parallel->sendRequests[item_owner][entryIndex.m_entry_uid].count++;
    }
  }

  // Liste de synthèse des messages (emissions / réceptions)
  parallel->messageList =
  Arccore::MessagePassing::mpCreateSerializeMessageListRef(m_parallel_mng);

  // Contruction de la table de communications + préparation des messages d'envoi
  UniqueArray<Integer> sendToDomains(2 * m_parallel_mng->commSize(), 0);

  for (auto i = parallel->sendRequests.begin(); i != parallel->sendRequests.end(); ++i) {
    const Integer destDomainId = i->first;
    auto& requests = i->second;
    for (auto& j : requests) {
      EntrySendRequest& request = j.second;
      const Integer entryImpl = j.first;
      const String nameString = m_entries[entryImpl]->getName();

      // Données pour receveur
      sendToDomains[2 * destDomainId + 0] += 1;
      sendToDomains[2 * destDomainId + 1] += request.count;

      // Construction du message du EntrySendRequest
      request.comm = Arccore::MessagePassing::Mpi::BasicSerializeMessage::create(
      MessageRank(m_parallel_mng->commRank()), MessageRank(destDomainId),
      Arccore::MessagePassing::ePointToPointMessageType::MsgSend);

      parallel->messageList->addMessage(request.comm.get());
      auto sbuf = request.comm->serializer();
      sbuf->setMode(Alien::ISerializer::ModeReserve); // phase préparatoire
      sbuf->reserve(nameString); // Chaine de caractère du nom de l'entrée
      sbuf->reserveInteger(1); // Nb d'item
      sbuf->reserve(Alien::ISerializer::DT_Int64, request.count); // Les uid
      sbuf->allocateBuffer(); // allocation mémoire
      sbuf->setMode(Alien::ISerializer::ModePut);
      sbuf->put(nameString);
      sbuf->put(request.count);
    }
  }

  // 2 - Accumulation des valeurs à demander

  for (auto& entryIndex : entry_index) {
    const Integer entryImpl = entryIndex.m_entry_uid;
    const Integer item_owner = entryIndex.m_item_owner;
    const Int64 item_uid = entryIndex.m_item_uid;
    if (item_owner != m_local_owner)
      parallel->sendRequests[item_owner][entryImpl].comm->serializer()->put(item_uid);
  }

  // Réception des annonces de demandes (les nombres d'entrée + taille)

  UniqueArray<Integer> recvFromDomains(2 * m_parallel_mng->commSize());
  Arccore::MessagePassing::mpAllToAll(m_parallel_mng, sendToDomains, recvFromDomains, 2);

  for (Integer isd = 0, nsd = m_parallel_mng->commSize(); isd < nsd; ++isd) {
    Integer recvCount = recvFromDomains[2 * isd + 0];
    while (recvCount-- > 0) {
      auto recvMsg = Arccore::MessagePassing::Mpi::BasicSerializeMessage::create(
      MessageRank(m_parallel_mng->commRank()), MessageRank(isd),
      Arccore::MessagePassing::ePointToPointMessageType::MsgReceive);

      parallel->recvRequests.push_back(EntryRecvRequest());
      EntryRecvRequest& recvRequest = parallel->recvRequests.back();
      recvRequest.comm = recvMsg;
      parallel->messageList->addMessage(recvMsg.get());
    }
  }

  // Traitement des communications
  parallel->messageList->processPendingMessages();
  parallel->messageList->waitMessages(Arccore::MessagePassing::WaitAll);
  parallel->messageList.reset();
  // delete parallel->messageList;
  // parallel->messageList = NULL; // Destruction propre

  // Pour les réponses vers les demandeurs
  parallel->messageList =
  Arccore::MessagePassing::mpCreateSerializeMessageListRef(m_parallel_mng);

  // 3 - Réception et mise en base local des demandes
  for (auto i = parallel->recvRequests.begin(); i != parallel->recvRequests.end(); ++i) {
    auto& recvRequest = *i;
    String nameString;
    Integer uidCount;

    { // Traitement des arrivées
      auto sbuf = recvRequest.comm->serializer();
      sbuf->setMode(Alien::ISerializer::ModeGet);

      sbuf->get(nameString);
      uidCount = sbuf->getInteger();
      recvRequest.ids.resize(uidCount);
      sbuf->getSpan(recvRequest.ids);
      ALIEN_ASSERT((uidCount == recvRequest.ids.size()), ("Inconsistency detected"));

#ifndef NO_USER_WARNING
#ifdef _MSC_VER
#pragma message("CHECK: optimisable ?")
#else
#warning "CHECK: optimisable ?"
#endif
#endif
      /* Si on est sûr que les entrées et l'item demandées doivent
       * toujours exister (même les pires cas), on peut faire
       * l'indexation locale avant et envoyer immédiatement (via un
       * buffer; dans la présente boucle) la réponse.
       */

      // Reconstruction de l'entrée à partir du nom
      auto lookup = std::find_if(m_entries.begin(), m_entries.end(),
                                 [&](ScalarIndexSet* s) { return s->getName() == nameString; });

      // Si pas d'entrée de ce côté => système défectueux ?
      if (lookup == m_entries.end())
        throw FatalErrorException("Non local Entry Requested : degenerated system ?");

      auto* currentEntry = *lookup;
      const Integer entry_uid = currentEntry->getUid();

      // Passage de l'uid à l'item associé (travaille sur place : pas de recopie)
      ArrayView<Int64> ids = recvRequest.ids;

      const auto* family = m_entry_families[entry_uid];
      const Integer entry_kind = currentEntry->getKind();
      UniqueArray<Int32> lids(ids.size());
      family->uniqueIdToLocalId(lids, ids);
      // Vérification d'intégrité : toutes les entrées demandées sont définies localement
      auto owners = family->owners(lids);
      for (Integer j = 0; j < uidCount; ++j) {
        const Integer current_item_lid = lids[j];
        const Int64 current_item_uid = ids[j];
        const Integer current_item_owner = owners[j];
        if (current_item_owner != m_local_owner) {
          throw FatalErrorException("Non local EntryIndex requested");
        }
        InternalEntryIndex lookup_entry{ entry_uid, entry_kind, current_item_uid,
                                         current_item_lid, 0, current_item_owner };

        // Recherche de la liste triée par défaut
        auto lookup2 = std::lower_bound(entry_index.begin(), entry_index.end(),
                                        lookup_entry, [](const InternalEntryIndex& a, const InternalEntryIndex& b) {
                                          if (a.m_entry_kind != b.m_entry_kind)
                                            return a.m_entry_kind < b.m_entry_kind;
                                          else if (a.m_item_uid != b.m_item_uid)
                                            return a.m_item_uid < b.m_item_uid;
                                          else
                                            return a.m_entry_uid < b.m_entry_uid;
                                        });

        if ((lookup2 == entry_index.end()) || !(*lookup2 == lookup_entry))
          throw FatalErrorException("Not locally defined entry requested");

        // Mise en place de la pre-valeur retour [avant renumérotation locale] (EntryIndex
        // écrit sur un Int64)
        ids[j] = lookup2->m_item_index;
      }
    }

    { // Préparation des retours
      auto dest = recvRequest.comm->destination(); // Attention à l'ordre bizarre
      auto orig = recvRequest.comm->source(); //       de SerializeMessage
      recvRequest.comm.reset();
      recvRequest.comm = Arccore::MessagePassing::Mpi::BasicSerializeMessage::create(
      orig, dest, Arccore::MessagePassing::ePointToPointMessageType::MsgSend);

      parallel->messageList->addMessage(recvRequest.comm.get());

      auto sbuf = recvRequest.comm->serializer();
      sbuf->setMode(Alien::ISerializer::ModeReserve); // phase préparatoire
      sbuf->reserve(nameString); // Chaine de caractère du nom de l'entrée
      sbuf->reserveInteger(1); // Nb d'item
      sbuf->reserveInteger(uidCount); // Les index
      sbuf->allocateBuffer(); // allocation mémoire
      sbuf->setMode(Alien::ISerializer::ModePut);
      sbuf->put(nameString);
      sbuf->put(uidCount);
    }
  }

  // 4 - Indexation locale
  /* La politique naive ici appliquée est de numéroter tous les
   * (Entry,Item) locaux d'abord.
   */
  // Calcul de des offsets globaux sur Entry (via les tailles locales)
  UniqueArray<Integer> allLocalSizes(m_parallel_mng->commSize());
  UniqueArray<Integer> myLocalSize(1);
  myLocalSize[0] = m_local_entry_count;
  Arccore::MessagePassing::mpAllGather(m_parallel_mng, myLocalSize, allLocalSizes);

  // Mise à jour du contenu des entrées
  m_global_entry_offset = 0;
  for (Integer i = 0; i < m_parallel_mng->commRank(); ++i) {
    m_global_entry_offset += allLocalSizes[i];
  }

  // Calcul de la taille global d'indexation (donc du système associé)
  m_global_entry_count = 0;
  for (Integer i = 0; i < m_parallel_mng->commSize(); ++i) {
    m_global_entry_count += allLocalSizes[i];
  }
}

/*---------------------------------------------------------------------------*/

void IndexManager::end_parallel_prepare(EntryIndexMap& entry_index)
{
  ALIEN_ASSERT((m_parallel_mng->commSize() > 1), ("Parallel mode expected"));

  {
    // Table de ré-indexation (EntryIndex->Integer)
    Alien::UniqueArray<Integer> entry_reindex(m_local_entry_count);
    Alien::fill(entry_reindex, -1); // valeur de type Erreur par défaut

    // C'est ici et uniquement ici qu'est matérialisé l'ordre des entrées
    Integer currentEntryIndex = m_global_entry_offset; // commence par l'offset local
    for (auto& i : entry_index) {
      if (i.m_item_owner == m_local_owner) { // Numérotation locale !
        const Integer newIndex = currentEntryIndex++;
        ALIEN_ASSERT(newIndex >= 0, "Invalid local id");
        entry_reindex[i.m_item_index] = newIndex; // Table de translation
        i.m_item_index = newIndex;
      }
    }

    // 5 - Envoie des retours (EntryIndex globaux)
    for (auto& recvRequest : parallel->recvRequests) {
      auto sbuf = recvRequest.comm->serializer();
      for (auto id : recvRequest.ids) {
        sbuf->putInteger(entry_reindex[id]); // Via la table de réindexation
      }
    }
  }
  // Table des buffers de retour
  typedef std::list<Alien::Ref<Alien::ISerializeMessage>> ReturnedRequests;
  ReturnedRequests returnedRequests;

  // Acces rapide aux buffers connaissant le proc emetteur et le nom d'une entrée
  /* Car on ne peut tager les buffers donc l'entrée reçue dans un buffer est non
   * déterminée
   * surtout si 2 domaines se communiquent plus d'une entrée
   */
  typedef std::map<Integer, EntrySendRequest*> SubFastReturnMap;
  typedef std::map<String, SubFastReturnMap> FastReturnMap;
  FastReturnMap fastReturnMap;

  // Préparation des réceptions [sens inverse]
  for (auto i = parallel->sendRequests.begin(); i != parallel->sendRequests.end(); ++i) {
    const Integer destDomainId = i->first;
    auto& requests = i->second;
    for (auto& j : requests) {
      auto& request = j.second;
      const Integer entryImpl = j.first;
      const String nameString = m_entries[entryImpl]->getName();

      // On ne peut pas associer directement le message à cette entrée
      // : dans le cas d'échange multiple il n'y pas de garantie d'arrivée
      // à la bonne place
      // delete request.comm;
      // request.comm = NULL;
      request.comm.reset();

      auto msg = Arccore::MessagePassing::Mpi::BasicSerializeMessage::create(
      MessageRank(m_parallel_mng->commRank()), MessageRank(destDomainId),
      Arccore::MessagePassing::ePointToPointMessageType::MsgReceive);

      returnedRequests.push_back(msg);
      parallel->messageList->addMessage(msg.get());

      fastReturnMap[nameString][destDomainId] = &request;
    }
  }

  // Traitement des communications
  parallel->messageList->processPendingMessages();

  parallel->messageList->waitMessages(Arccore::MessagePassing::WaitAll);
  parallel->messageList.reset();

  // 6 - Traitement des réponses
  // Association aux EntrySendRequest du buffer correspondant
  for (auto& returnedRequest : returnedRequests) {
    auto& message = returnedRequest;
    auto origDomainId = message->destination().value();
    auto sbuf = message->serializer();
    sbuf->setMode(Alien::ISerializer::ModeGet);
    String nameString;
    sbuf->get(nameString);
    ALIEN_ASSERT(
    (fastReturnMap[nameString][origDomainId] != nullptr), ("Inconsistency detected"));
    auto& request = *fastReturnMap[nameString][origDomainId];
    request.comm =
    returnedRequest; // Reconnection pour accès rapide depuis l'EntrySendRequest
#ifdef ALIEN_DEBUG_ASSERT
    const Integer idCount = sbuf.getInteger();
    ALIEN_ASSERT((request.count == idCount), ("Inconsistency detected"));
#else
    const Integer idCount = sbuf->getInteger();
    ALIEN_ASSERT((request.count == idCount), ("Inconsistency detected"));
#endif
  }

  // Distribution des reponses
  // Par parcours dans ordre initial (celui de la demande)
  for (auto& entry : entry_index) {
    const Integer item_owner = entry.m_item_owner;
    if (item_owner != m_local_owner) {
      const Integer entryImpl = entry.m_entry_uid;
      auto& request = parallel->sendRequests[item_owner][entryImpl];
      ALIEN_ASSERT((request.count > 0), ("Unexpected empty request"));
      --request.count;
      auto sbuf = request.comm->serializer();
      const Integer newIndex = sbuf->getInteger();
      entry.m_item_index = newIndex;
    }
  }
}

/*---------------------------------------------------------------------------*/

void IndexManager::sequential_prepare(EntryIndexMap& entry_index)
{
  ALIEN_ASSERT((m_parallel_mng->commSize() <= 1), ("Sequential mode expected"));
  ALIEN_ASSERT((m_global_entry_count == 0),
               ("Unexpected global entries (%d)", m_global_entry_count));

  // Très similaire à la section parallèle :
  // 4 - Indexation locale
  /* La politique naive ici appliquée est de numéroter tous les
   * (Entry,Item) locaux d'abord.
   */

  // Mise à jour du contenu des entrées

  // C'est ici et uniquement ici qu'est matérialisé l'ordre des entrées
  Integer currentEntryIndex = 0; // commence par l'offset local
  for (auto i = entry_index.begin(); i != entry_index.end(); ++i) {
    ALIEN_ASSERT((i->m_item_owner == m_local_owner),
                 ("Item cannot be non-local for sequential mode"));
    // Numérotation locale only !
    const Integer newIndex = currentEntryIndex++;
    i->m_item_index = newIndex;
  }

  m_global_entry_count = m_local_entry_count;
}

/*---------------------------------------------------------------------------*/

UniqueArray<Integer>
IndexManager::getIndexes(const ScalarIndexSet& entry) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));
  const IAbstractFamily& family = entry.getFamily();
  UniqueArray<Integer> allIds(family.maxLocalId(), nullIndex());
  const ConstArrayView<Integer> allIndices = getAllIndexes(entry);
  const ConstArrayView<Integer> allLocalIds = getAllLocalIds(entry);
  const Integer size = allIndices.size();
  for (Integer i = 0; i < size; ++i)
    allIds[allLocalIds[i]] = allIndices[i];
  return allIds;
}

/*---------------------------------------------------------------------------*/

UniqueArray2<Integer>
IndexManager::getIndexes(const VectorIndexSet& entries) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  Integer max_family_size = 0;
  for (Integer i = 0; i < entries.size(); ++i) {
    // controles uniquement en première passe
    ALIEN_ASSERT(
    (entries[i].manager() == this), ("Incompatible entry from another manager"));
    const auto& entry = entries[i];
    const IAbstractFamily& family = entry.getFamily();
    max_family_size = std::max(max_family_size, family.maxLocalId());
  }

  UniqueArray2<Integer> allIds;
  Alien::allocateData(allIds, max_family_size, entries.size());
  Alien::fill(allIds, nullIndex());

  for (Integer i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    const ConstArrayView<Integer> allIndices = getAllIndexes(entry);
    const ConstArrayView<Integer> allLocalIds = getAllLocalIds(entry);
    const Integer size = allIndices.size();
    for (Integer j = 0; j < size; ++j)
      allIds[allLocalIds[j]][i] = allIndices[i];
  }
  return allIds;
}

/*---------------------------------------------------------------------------*/

void IndexManager::stats(Integer& globalSize, Integer& minLocalIndex, Integer& localSize) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  globalSize = m_global_entry_count;
  minLocalIndex = m_global_entry_offset;
  localSize = m_local_entry_count;
}

/*---------------------------------------------------------------------------*/

Integer
IndexManager::globalSize() const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  return m_global_entry_count;
}

/*---------------------------------------------------------------------------*/

Integer
IndexManager::minLocalIndex() const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  return m_global_entry_offset;
}

/*---------------------------------------------------------------------------*/

Integer
IndexManager::localSize() const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");

  return m_local_entry_count;
}

/*---------------------------------------------------------------------------*/

ScalarIndexSet
IndexManager::buildScalarIndexSet(const String& name,
                                  ConstArrayView<Integer> localIds,
                                  const IAbstractFamily& family,
                                  Integer kind,
                                  eKeepAlive alive)
{
  alien_debug([&] {
    cout() << "IndexManager: build scalar index set '" << name << "', kind=" << kind;
  });
  ScalarIndexSet en = buildEntry(name, addNewAbstractFamily(&family, alive), kind);
  defineIndex(en, localIds);
  return en;
}

/*---------------------------------------------------------------------------*/

ScalarIndexSet
IndexManager::buildScalarIndexSet(
const String& name, const IAbstractFamily& family, Integer kind, eKeepAlive alive)
{
  alien_debug([&] {
    cout() << "IndexManager: build scalar index set '" << name << "', kind=" << kind;
  });
  auto localIds = family.allLocalIds();
  ScalarIndexSet en = buildEntry(name, addNewAbstractFamily(&family, alive), kind);
  defineIndex(en, localIds.view());
  return en;
}

/*---------------------------------------------------------------------------*/

IndexManager::VectorIndexSet
IndexManager::buildVectorIndexSet(const String& name,
                                  ConstArrayView<Integer> localIds,
                                  const IAbstractFamily& family,
                                  const UniqueArray<Integer>& kind,
                                  eKeepAlive alive)
{
  alien_debug([&] {
    cout() << "IndexManager: build vector index set '" << name
           << "', size=" << kind.size();
  });
  const Integer size = kind.size();
  VectorIndexSet ens(size);
  const auto* f = addNewAbstractFamily(&family, alive);
  for (Integer i = 0; i < size; ++i) {
    ens[i] = buildEntry(Alien::format("{0}[{1}]", name, i), f, kind[i]);
    defineIndex(ens[i], localIds);
  }
  return ens;
}

/*---------------------------------------------------------------------------*/

IndexManager::VectorIndexSet
IndexManager::buildVectorIndexSet(const String& name,
                                  const IAbstractFamily& family,
                                  const UniqueArray<Integer>& kind,
                                  eKeepAlive alive)
{
  alien_debug([&] {
    cout() << "IndexManager: build vector index set '" << name
           << "', size=" << kind.size();
  });
  auto localIds = family.allLocalIds();
  const Integer size = kind.size();
  VectorIndexSet ens(size);
  const auto* f = addNewAbstractFamily(&family, alive);
  for (Integer i = 0; i < size; ++i) {
    ens[i] = buildEntry(Alien::format("{0}[{1}]", name, i), f, kind[i]);
    defineIndex(ens[i], localIds.view());
  }
  return ens;
}

/*---------------------------------------------------------------------------*/

const IAbstractFamily*
IndexManager::addNewAbstractFamily(const IAbstractFamily* family, eKeepAlive alive)
{
  auto finder = m_abstract_families.find(family);
  if (finder == m_abstract_families.end()) // La famille n'est pas stockée, nouvelle famille
  {
    if (alive == eKeepAlive::DontClone) {
      m_abstract_families[family] = std::shared_ptr<IAbstractFamily>();
      return family;
    }
    else {
      auto clone = std::shared_ptr<IAbstractFamily>(family->clone());
      m_abstract_families[family] = clone;
      // On remplace les familles des entrées
      for (auto& f : m_entry_families) {
        if (f.second == family)
          f.second = clone.get();
      }
      return clone.get();
    }
  }
  else // La famille est connue
  {
    if (finder->second) // Si clone, on le renvoit
      return finder->second.get();
    else { // Sinon, on crée éventuellement le clone
      if (alive == eKeepAlive::DontClone) {
        return family;
      }
      else {
        auto clone = std::shared_ptr<IAbstractFamily>(family->clone());
        m_abstract_families[family] = clone;
        // On remplace les familles des entrées
        for (auto& f : m_entry_families) {
          if (f.second == family)
            f.second = clone.get();
        }
        return clone.get();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
IndexManager::getOwnIndexes(const ScalarIndexSet& entry) const
{
  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");
  auto it = m_entry_own_indices.find(entry.getUid());
  return it->second;
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
IndexManager::getOwnLocalIds(const ScalarIndexSet& entry) const
{
  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");
  auto it = m_entry_own_items.find(entry.getUid());
  return it->second;
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
IndexManager::getAllIndexes(const ScalarIndexSet& entry) const
{
  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");
  auto it = m_entry_all_indices.find(entry.getUid());
  return it->second;
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
IndexManager::getAllLocalIds(const ScalarIndexSet& entry) const
{
  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");
  auto it = m_entry_all_items.find(entry.getUid());
  return it->second;
}

/*---------------------------------------------------------------------------*/

const IAbstractFamily&
IndexManager::getFamily(const ScalarIndexSet& entry) const
{
  ALIEN_ASSERT((entry.manager() == this), ("Incompatible entry from another manager"));
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO, "Inconsistent state");
  auto it = m_entry_families.find(entry.getUid());
  return *(it->second);
}

/*---------------------------------------------------------------------------*/

void IndexManager::setMaxNullIndexOpt(bool flag)
{
  m_max_null_index_opt = flag;

  alien_debug(flag, [&] { cout() << "IndexManager: null index optimized enabled"; });
}

/*---------------------------------------------------------------------------*/

Integer
IndexManager::nullIndex() const
{
  ALIEN_ASSERT((m_state == Prepared), ("nullIndex is valid only in Prepared state"));
  if (m_max_null_index_opt)
    return m_global_entry_offset + m_local_entry_count;
  else
    return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
