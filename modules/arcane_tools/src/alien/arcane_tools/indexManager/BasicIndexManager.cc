#include "alien/arcane_tools/indexManager/BasicIndexManager.h"

#include <list>
#include <vector>

#include <arcane/ArcaneVersion.h>
#include <arcane/IItemFamily.h>
#include <arcane/IParallelMng.h>
#include <arcane/ISerializeMessageList.h>
#include <arcane/ItemGroup.h>
#include <arcane/ItemInternal.h>
#include <arcane/SerializeMessage.h>
#include <arcane/utils/Collection.h>
#include <arcane/utils/Enumerator.h>
#include <arcane/utils/Math.h>

#include <arccore/collections/Array2.h>
#include <arccore/trace/ITraceMng.h>

using namespace Arcane;

// #define SPLIT_CONTAINER
/* La version avec SPLIT_CONTAINER fait moins d'appel virtuel mais consomme un peu plus de
 * mémoire (tableau de vectorisation). Factuellement est sans SPLIT_CONTAINER est plus
 * rapide de environ 5% NB: Si l'on supprime définitivement SPLIT_CONTAINER, l'API de
 * IAbstractFamily peut être simplifier des méthodes uids et owners vectorisées (ainsi que
 * ses implémentations).
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  // Utilities
  namespace { // Unnamed namespace to avoid conflicts at linking time.

    const Arccore::Integer KIND_SHIFT = 32;

  }

  /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicIndexManager::ItemAbstractFamily::
ItemAbstractFamily(const Arcane::IItemFamily * family) 
  : m_family(family)
  , m_item_internals(const_cast<Arcane::IItemFamily *>(family)->itemsInternal())
{ 
#ifndef NO_USER_WARNING
#ifdef _MSC_VER
#pragma message ("TODO: const_cast due to ill formed view/itemsInternal methods : const requested")
#else
#warning "TODO: const_cast due to ill formed view/itemsInternal methods : const requested"
#endif
#endif /* NO_USER_WARNING */
}

IIndexManager::IAbstractFamily *
BasicIndexManager::ItemAbstractFamily::
clone() const
{
  return new ItemAbstractFamily(*this);
}

Int32
BasicIndexManager::ItemAbstractFamily::
maxLocalId() const
{
  return m_family->maxLocalId();
}

void
BasicIndexManager::ItemAbstractFamily::
uniqueIdToLocalId(Int32ArrayView localIds, Int64ConstArrayView uniqueIds) const
{
  m_family->itemsUniqueIdToLocalId(localIds, uniqueIds, true); 
}

IIndexManager::IAbstractFamily::Item
BasicIndexManager::ItemAbstractFamily::
item(Int32 localId) const
{
  Arcane::ItemInternal * internal = m_item_internals[localId];
  return Item(internal->uniqueId(), internal->owner());
}

IntegerSharedArray
BasicIndexManager::ItemAbstractFamily::
owners(Int32ConstArrayView localIds) const
{
  IntegerSharedArray result(localIds.size());
  for(Integer i=0;i<localIds.size();++i)
  {
    Arcane::ItemInternal * internal = m_item_internals[localIds[i]];
    result[i] = internal->owner();
  }
  return result;
}

Int64SharedArray
BasicIndexManager::ItemAbstractFamily::
uids(Int32ConstArrayView localIds) const
{
  Int64SharedArray result(localIds.size());
  for(Integer i=0;i<localIds.size();++i)
  {
    Arcane::ItemInternal * internal = m_item_internals[localIds[i]];
    result[i] = internal->uniqueId();
  }
  return result;
}

Int32SharedArray
BasicIndexManager::ItemAbstractFamily::
allLocalIds() const {
  return m_family->allItems().view().localIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/** \brief Squelette de l'implementation locale de Entry
 *  La vraie implémentation est MyAbstractEntryImpl.
 *  Cette classe permet de la factorisation de code
 */
class BasicIndexManager::MyEntryImpl :
  public IIndexManager::EntryImpl
{
  friend class BasicIndexManager;
public:
  //! Constructeur par défaut
  MyEntryImpl(const String name, const IAbstractFamily * family, const Integer creationIndex, BasicIndexManager * manager, Integer kind)
    : m_creation_index(creationIndex)
    , m_manager(manager)
    , m_name(name)
    , m_family(family)
    , m_kind(kind)
    , m_is_defined(family->maxLocalId(), false)
    , m_own_size(0)
    , m_size(0)
  {
    ;
  }

  virtual ~MyEntryImpl()
  {
    ;
  }

  Arccore::ConstArrayView<Arccore::Integer> getOwnIndexes() const
  {
    return Arccore::ConstArrayView<Arccore::Integer>(m_own_size, m_all_indices.data());
  }

  Arccore::ConstArrayView<Arccore::Integer> getOwnLocalIds() const
  {
    return Arccore::ConstArrayView<Arccore::Integer>(m_own_size, m_all_items.data());
  }

  Arccore::ConstArrayView<Arccore::Integer> getAllIndexes() const
  {
    return Arccore::ConstArrayView<Arccore::Integer>(m_all_indices);
  }

  Arccore::ConstArrayView<Integer> getAllLocalIds() const
  {
    return Arccore::ConstArrayView<Integer>(m_all_items);
  }

  void addTag(const String &tagname, const String &tagvalue)
  {
    m_tags[tagname] = tagvalue;
  }

  void removeTag(const String &tagname)
  {
    m_tags.erase(tagname);
  }

  bool hasTag(const String &tagname)
  {
    return m_tags.find(tagname) != m_tags.end();
  }

  String tagValue(const String & tagname)
  {
    std::map<String,String>::const_iterator i = m_tags.find(tagname);
    if (i==m_tags.end())
      return String();
    return i->second;
  }

  String getName() const
  {
    return m_name;
  }

  Integer getKind() const
  {
    ARCANE_ASSERT((m_kind>=0),("Unexpected negative kind"));
    if (m_kind < KIND_SHIFT)
      return -1; // this is an abstract entity
    else
      return m_kind % KIND_SHIFT;
  }

  const IAbstractFamily & getFamily() const
  {
    return *m_family;
  }

  IIndexManager * manager() const
  {
    return m_manager;
  }

protected:
  //! Préparation des buffers d'indices et d'items
  /*! Utilisation réservée au 'friend' BasicIndexManager */
  void reserve(const Integer n)
  {
    m_size = n;
  }

  //! Fige les données de l'entry (fin de phase prepare)
  /*! Utilisation réservée au 'friend' BasicIndexManager */
  void finalize(const EntryIndexMap & entryIndex)
  {
    m_all_items.resize(m_size);
    m_all_indices.resize(m_size);

    Integer own_i = 0;
    Integer ghost_i = m_size;
    for(EntryIndexMap::const_iterator i = entryIndex.begin(); i != entryIndex.end(); ++i)
      if (i->m_entry == this)
      {
	const Integer local_id = i->m_localid;
	const Integer index = i->m_index;
	const bool is_own = m_manager->isOwn(*i);
	if (is_own)
	{
	  m_all_items[own_i] = local_id;
	  m_all_indices[own_i] = index;
	  ++own_i;
	}
	else
	{
	  --ghost_i;
	  m_all_items[ghost_i] = local_id;
	  m_all_indices[ghost_i] = index;
	}
      }
    m_own_size = own_i;
    ARCANE_ASSERT((own_i == ghost_i),("Not merged insertion"));

    //     // Tri  de la partie own des indices
    //     typedef UniqueArray<Integer>::iterator iterator;
    //     DualRandomIterator<iterator,iterator> begin(m_all_indices.begin(), m_all_items.begin());
    //     DualRandomIterator<iterator,iterator> end = begin + m_own_size;
    //     Comparator<DualRandomIterator<iterator,iterator> > comparator;
    //     std::sort(begin, end, comparator);
  }

  void resetFamily(const IAbstractFamily * family)
  {
    m_family = family;
  }

protected:

  void reserveLid(const Integer count) {
    m_defined_lids.reserve(m_defined_lids.size() + count);
#ifdef SPLIT_CONTAINER
    m_defined_indexes.reserve(m_defined_indexes.size() + count);
#endif /* SPLIT_CONTAINER */
  }
  bool isDefinedLid(const Integer localId) const { return m_is_defined[localId]; }

  void defineLid(const Integer localId, const Integer pos) {
    m_is_defined[localId] = true;
#ifdef SPLIT_CONTAINER
    m_defined_lids.add(localId);
    m_defined_indexes.add(pos);
#else /* SPLIT_CONTAINER */
    m_defined_lids.add(std::make_pair(localId,pos));
#endif /* SPLIT_CONTAINER */
  }

  void undefineLid(const Integer localId) {
    m_is_defined[localId] = false;
    for(Integer i=0;i<m_defined_lids.size();++i)
    {
#ifdef SPLIT_CONTAINER
      if (m_defined_lids[i] == localId)
#else /* SPLIT_CONTAINER */
        if (m_defined_lids[i].first == localId)
#endif /* SPLIT_CONTAINER */
	{
	  m_defined_lids[i] = m_defined_lids.back();
	  m_defined_lids.resize(m_defined_lids.size()-1);
#ifdef SPLIT_CONTAINER
	  m_defined_indexes[i] = m_defined_indexes.back();
	  m_defined_indexes.resize(m_defined_lids.size()-1);
#endif /* SPLIT_CONTAINER */
	}
    }
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state : cannot find id to remove");
  }

#ifdef SPLIT_CONTAINER
  const UniqueArray<Integer> & definedLids() const { return m_defined_lids; }
  const UniqueArray<Integer> & definedIndexes() const { return m_defined_indexes; }
#else /* SPLIT_CONTAINER */
  const UniqueArray<std::pair<Integer, Integer> > & definedLids() const { return m_defined_lids; }
#endif /* SPLIT_CONTAINER */


  void freeDefinedLids() {
#ifdef SPLIT_CONTAINER
    m_defined_lids.dispose();
    m_defined_indexes.dispose();
#else /* SPLIT_CONTAINER */
    m_defined_lids.dispose();
#endif /* SPLIT_CONTAINER */
    std::vector<bool>().swap(m_is_defined);
  }

private:
  std::map<String,String> m_tags;
  Integer m_creation_index;
  BasicIndexManager * m_manager;
  const String m_name;
  const IAbstractFamily * m_family;
  const Integer m_kind;

  std::vector<bool> m_is_defined;
#ifdef SPLIT_CONTAINER
  UniqueArray<Integer> m_defined_lids;
  UniqueArray<Integer> m_defined_indexes;
#else /* SPLIT_CONTAINER */
  UniqueArray<std::pair<Integer, Integer> > m_defined_lids;
#endif /* SPLIT_CONTAINER */

private:
  UniqueArray<Integer> m_all_items; //!< LocalIds des items gérés par cette entrée rangés own puis ghost
  UniqueArray<Integer> m_all_indices; //!< Indices 'own' dans la numérotation globale de l'index-manager par cette entrée rangés own puis ghost
  Integer m_own_size; //!< Nombre d'items own dans les tableaux m_all_*
  Integer m_size;

public:
  Integer getCreationIndex() const { return m_creation_index; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct BasicIndexManager::EntrySendRequest
{
  EntrySendRequest() : comm(NULL), count(0)
  {
    ;
  }

  ~EntrySendRequest()
  {
    // Valide même si comm vaut NULL
    delete comm;
  }

  EntrySendRequest(const EntrySendRequest & esr)
    : comm(NULL)
    , count(esr.count)
  {
    ARCANE_ASSERT((esr.comm == NULL),("Bad initialization"));
  }

  SerializeMessage * comm;
  Integer count;

private:
  void operator=(const EntrySendRequest &);
};

/*---------------------------------------------------------------------------*/

struct BasicIndexManager::EntryRecvRequest
{
  EntryRecvRequest() : comm(NULL)
  {
    ;
  }

  ~EntryRecvRequest()
  {
    // Valide même si comm vaut NULL
    delete comm;
  }

  // err is unused if ARCANE_ASSERT is empty.
  EntryRecvRequest(const EntrySendRequest & err ALIEN_UNUSED_PARAM)
    : comm(NULL)
  {
    ARCANE_ASSERT((err.comm == NULL),("Bad initialization"));
  }

  SerializeMessage * comm;
  Int64UniqueArray ids;

private:
  void operator=(const EntryRecvRequest &);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicIndexManager::MyEntryEnumeratorImpl
  : public IIndexManager::EntryEnumeratorImpl
{
protected:
  EntrySet::const_iterator m_iter, m_end;
public:
  MyEntryEnumeratorImpl(const EntrySet & entries)
    : m_iter(entries.begin()),
      m_end(entries.end())
  {
    ;
  }

  void moveNext() { ++m_iter; }

  bool hasNext() const { return m_iter != m_end; }

  EntryImpl * get() const { return m_iter->second; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicIndexManager::
BasicIndexManager(IParallelMng * parallelMng) 
  : m_parallel_mng(parallelMng)
  , m_local_owner(0)
  , m_state(Undef)
  , m_trace(nullptr)
  , m_local_entry_count(0)
  , m_global_entry_count()
  , m_global_entry_offset(0)
  , m_local_removed_entry_count(0)
  , m_global_removed_entry_count(0)
  , m_max_null_index_opt(false)
  , m_creation_index(0)
  , m_abstract_family_base_kind(0)
{
  this->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicIndexManager::
~BasicIndexManager()
{
  this->init();
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
init()
{
  m_local_owner = m_parallel_mng->commRank();

  m_state = Initialized;

  m_local_entry_count = 0;
  m_global_entry_count = 0;
  m_global_entry_offset = 0;
  m_local_removed_entry_count = 0;
  m_global_removed_entry_count = 0;

  // Destruction des structure de type entry
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
  {
    delete i->second;
  }
  m_entry_set.clear();

  m_abstract_family_base_kind = 0;
  m_item_family_meshes.clear();
  m_abstract_families.clear();
  m_abstract_family_to_kind_map.clear();
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
setTraceMng(ITraceMng * traceMng)
{
  m_trace = traceMng;
}

/*---------------------------------------------------------------------------*/

IIndexManager::Entry
BasicIndexManager::
buildEntry(const String name, const IAbstractFamily * family, const Integer kind)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  // Recherche de l'entrée d'un nom
  std::pair<EntrySet::iterator,bool> lookup
    = m_entry_set.insert(EntrySet::value_type(name,(MyEntryImpl *)NULL));
  if (lookup.second)
  {
    MyEntryImpl * entry = new MyEntryImpl(name,family, m_creation_index++, this, kind);
    lookup.first->second = entry;
    return entry;
  }
  else
  {
    throw FatalErrorException(A_FUNCINFO,"Already defined entry");
    return NULL;
  }
}

/*---------------------------------------------------------------------------*/

IIndexManager::Entry
BasicIndexManager::
getEntry(const String name) const
{
  EntrySet::const_iterator lookup = m_entry_set.find(name);
  if (lookup != m_entry_set.end())
  {
    return lookup->second;
  }
  else
  {
    throw FatalErrorException(A_FUNCINFO,"Undefined entry requested");
    return NULL;
  }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
defineIndex(const Entry & entry, const IntegerConstArrayView localIds)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ARCANE_ASSERT((entry.manager() == this),("Incompatible entry from another manager"));
  MyEntryImpl * myEntry = static_cast<MyEntryImpl*>(entry.internal());

  const IAbstractFamily & family = myEntry->getFamily();
  IntegerSharedArray owners = family.owners(localIds);
  myEntry->reserveLid(localIds.size());
  for(Integer i=0,is=localIds.size();i<is; ++i)
  {
    const Integer localId = localIds[i];

    if (not myEntry->isDefinedLid(localId)) { // nouvelle entrée
      if (owners[i] == m_local_owner)
      {
	myEntry->defineLid(localId, +(m_local_removed_entry_count+m_local_entry_count++));
      }
      else
      {
	myEntry->defineLid(localId, -(m_global_removed_entry_count+(++m_global_entry_count)));
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
removeIndex(const ScalarIndexSet & entry, const Arcane::ItemGroup & itemGroup)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ARCANE_ASSERT((entry.manager() == this),("Incompatible entry from another manager"));
  MyEntryImpl * myEntry = static_cast<MyEntryImpl*>(entry.internal());

  ARCANE_ASSERT((itemGroup.itemKind() == myEntry->getKind()),("Bad kind item"));
  ENUMERATE_ITEM(i,itemGroup)
  {
    const Item & item = *i;
    const Integer localId = item.localId();
    if (myEntry->isDefinedLid(localId)) {
      myEntry->undefineLid(localId);
      if (item.isOwn())
      {
	--m_local_entry_count ;
	++m_local_removed_entry_count ;
      }
      else
      {
	--m_global_entry_count ;
	++m_global_removed_entry_count ;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
prepare()
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  Integer total_size = 0;
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
  {
    MyEntryImpl * entry = i->second;
    total_size += entry->definedLids().size();
  }

  EntryIndexMap entry_index; //!< Table des index d'entrées (>=0:local, <0:global) en phase1
  entry_index.reserve(total_size);
  
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
  {
    MyEntryImpl * entry = i->second;

    const Integer creation_index = entry->getCreationIndex();
    const IAbstractFamily & family = entry->getFamily();
    const Integer entry_kind = entry->getKind();
#ifdef SPLIT_CONTAINER
    const UniqueArray<Integer> & lids = entry->definedLids();
    const UniqueArray<Integer> & indexes = entry->definedIndexes();
    ConstArrayView<Integer> owners = family.owners(lids);
    ConstArrayView<Int64> uids = family.uids(lids);
#else /* SPLIT_CONTAINER */
    const UniqueArray<std::pair<Integer, Integer> > & lids = entry->definedLids();
#endif /* SPLIT_CONTAINER */

    for(Integer i=0, is=lids.size();i<is;++i)
    {
#ifdef SPLIT_CONTAINER
      const Integer item_lid = lids[i];
      const Integer item_index = indexes[i];
      const Integer item_owner = owners[i];
      const Int64   item_uid = uids[i];
      entry_index.push_back(InternalEntryIndex(entry, item_lid, entry_kind, item_uid, item_index, creation_index, item_owner));
#else /* SPLIT_CONTAINER */
      const Integer localid = lids[i].first;
      IAbstractFamily::Item item = family.item(localid);
      entry_index.push_back(InternalEntryIndex(entry, localid, entry_kind, item.uniqueId(), lids[i].second, creation_index, item.owner()));
#endif /* SPLIT_CONTAINER */
    }
    entry->freeDefinedLids();
  }

  std::sort(entry_index.begin(), entry_index.end(), EntryIndexComparator());

  ARCANE_ASSERT(((Integer)entry_index.size() == m_local_entry_count + m_global_entry_count),("Inconsistent global size"));

  if (m_parallel_mng->isParallel() and m_parallel_mng->commSize() > 1)
    parallel_prepare(entry_index);
  else
    sequential_prepare(entry_index);

  // Finalize : fige les données dans les entries
  for(EntrySet::iterator i=m_entry_set.begin(); i != m_entry_set.end(); ++i)
  {
    i->second->finalize(entry_index);
  }

  if (m_trace) {
    m_trace->info() << "Entry ordering :";
    for(EntrySet::iterator i=m_entry_set.begin(); i != m_entry_set.end(); ++i)
    {
      m_trace->info() << "\tEntry '" << i->first << "' placed at rank " << i->second->getCreationIndex() << " with " << i->second->getOwnLocalIds().size() << " local / " << i->second->getAllLocalIds().size() << " global indexes ";
    }
    m_trace->info() << "Total local Entry indexes = " << m_local_entry_count;
  }

  m_state = Prepared;
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
parallel_prepare(EntryIndexMap & entry_index)
{
  ARCANE_ASSERT((m_parallel_mng->isParallel()),("Parallel mode expected"));

  /* Algorithme:
   * 1 - listing des couples Entry-Item non locaux
   * 2 - Envoi vers les propriétaires des items non locaux
   * 3 - Prise en compte éventuelle de nouvelles entrées
   * 4 - Nommage locales
   * 5 - Retour vers demandeurs des EntryIndex non locaux
   * 6 - Finalisation de la numérotation (table reindex)
   */

  // Infos utiles
  ISerializeMessageList * messageList;

  // Structure pour accumuler et structurer la collecte de l'information
  typedef std::map<EntryImpl*,EntrySendRequest> SendRequestByEntry;
  typedef std::map<Integer, SendRequestByEntry> SendRequests;
  SendRequests sendRequests;

  // 1 - Comptage des Items non locaux
  for(EntryIndexMap::const_iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  {
    const InternalEntryIndex & entryIndex = *i;
    MyEntryImpl * entryImpl = entryIndex.m_entry;
    const Integer item_owner = entryIndex.m_owner;
    if (item_owner != m_local_owner)
    {
      // 	  if (m_trace) m_trace->pinfo() << item.localId() << " : " << item.uniqueId() << " is owned by " << item.owner() << " with localIndex=" << i->second;
      sendRequests[item_owner][entryImpl].count++;
    }
    else
    {
      // 	  if (m_trace) m_trace->pinfo() << item.localId() << " : " << item.uniqueId() << " is local with localIndex=" << i->second;
    }
  }

  // Liste de synthèse des messages (emissions / réceptions)
  messageList = m_parallel_mng->createSerializeMessageList();

  // Contruction de la table de communications + préparation des messages d'envoi
  UniqueArray<Integer> sendToDomains(2*m_parallel_mng->commSize(),0);

  for(SendRequests::iterator i = sendRequests.begin(); i != sendRequests.end(); ++i)
  {
    const Integer destDomainId = i->first;
    SendRequestByEntry & requests = i->second;
    for(SendRequestByEntry::iterator j = requests.begin(); j != requests.end(); ++j)
    {
      EntrySendRequest & request = j->second;
      EntryImpl * entryImpl = j->first;
      const String nameString = entryImpl->getName();

      //          if (m_trace) m_trace->pinfo() << "Entry [" << nameString << "] to " << destDomainId << " : " << request.count;

      // Données pour receveur
      sendToDomains[2*destDomainId + 0] += 1;
      sendToDomains[2*destDomainId + 1] += request.count;

      // Construction du message du EntrySendRequest
      request.comm = new SerializeMessage(m_parallel_mng->commRank(),destDomainId,ISerializeMessage::MT_Send);

      messageList->addMessage(request.comm);
      SerializeBuffer & sbuf = request.comm->buffer();
      sbuf.setMode(ISerializer::ModeReserve); // phase préparatoire
      sbuf.reserve(nameString); // Chaine de caractère du nom de l'entrée
      sbuf.reserveInteger(1); // Nb d'item
      sbuf.reserve(DT_Int64,request.count); // Les uid
      sbuf.allocateBuffer(); // allocation mémoire
      sbuf.setMode(ISerializer::ModePut);
      sbuf.put(nameString);
      sbuf.put(request.count);
    }
  }

  // 2 - Accumulation des valeurs à demander
  for(EntryIndexMap::const_iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  {
    const InternalEntryIndex & entryIndex = *i;
    MyEntryImpl * entryImpl = entryIndex.m_entry;
    const Integer item_owner = entryIndex.m_owner;
    const Int64 item_uid = entryIndex.m_uid;
    if (item_owner != m_local_owner)
      sendRequests[item_owner][entryImpl].comm->buffer().put(item_uid);
  }

  // Réception des annonces de demandes (les nombres d'entrée + taille)
  UniqueArray<Integer> recvFromDomains(2*m_parallel_mng->commSize());
  m_parallel_mng->allToAll(sendToDomains,recvFromDomains,2);

  // Table des requetes exterieures (reçoit les uid et renverra les EntryIndex finaux)
  typedef std::list<EntryRecvRequest> RecvRequests;
  RecvRequests recvRequests;

  for(Integer isd=0, nsd=m_parallel_mng->commSize();isd<nsd;++isd)
  {
    Integer recvCount = recvFromDomains[2*isd+0];
    while ( recvCount-- > 0 )
    {
      // 	  if (m_trace) m_trace->pinfo() << "will receive an entry with " << recvFromDomains[2*isd+1] << " uid from " << isd;
      SerializeMessage * recvMsg = new SerializeMessage(m_parallel_mng->commRank(),isd,ISerializeMessage::MT_Recv);
      recvRequests.push_back(EntryRecvRequest());
      EntryRecvRequest & recvRequest = recvRequests.back();
      recvRequest.comm = recvMsg;
      messageList->addMessage(recvMsg);
    }
  }

  // Traitement des communications
  messageList->processPendingMessages();
  messageList->waitMessages(Parallel::WaitAll);
  delete messageList; messageList = NULL; // Destruction propre


  // Pour les réponses vers les demandeurs
  messageList = m_parallel_mng->createSerializeMessageList();

  // 3 - Réception et mise en base local des demandes
  for(RecvRequests::iterator i = recvRequests.begin(); i != recvRequests.end(); ++i)
  {
    EntryRecvRequest & recvRequest = *i;
    String nameString;
    Integer uidCount;

    { // Traitement des arrivées
      SerializeBuffer& sbuf = recvRequest.comm->buffer();
      sbuf.setMode(ISerializer::ModeGet);

      sbuf.get(nameString);
      uidCount = sbuf.getInteger();
      // 	if (m_trace) m_trace->pinfo() << nameString << " received with " << uidCount << " ids";
      recvRequest.ids.resize(uidCount);
      sbuf.get(recvRequest.ids);
      ARCANE_ASSERT((uidCount == recvRequest.ids.size()),("Inconsistency detected"));

#ifndef NO_USER_WARNING
#ifdef _MSC_VER
#pragma message ("CHECK: optimisable ?")
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
      EntrySet::iterator lookup = m_entry_set.find(nameString);
      // Si pas d'entrée de ce côté => système défectueux ?
      if (lookup == m_entry_set.end())
	throw FatalErrorException("Non local Entry Requested : degenerated system ?");

      MyEntryImpl * currentEntry = lookup->second;
      const Integer current_creation_index = currentEntry->getCreationIndex();

      // Passage de l'uid à l'item associé (travaille sur place : pas de recopie)
      Int64ArrayView ids = recvRequest.ids;

      const IAbstractFamily & family = currentEntry->getFamily();
      const Integer entry_kind = currentEntry->getKind();
      Int32UniqueArray lids(ids.size());
      family.uniqueIdToLocalId(lids, ids);
      // Vérification d'intégrité : toutes les entrées demandées sont définies localement
      IntegerSharedArray owners = family.owners(lids).clone();
      for(Integer j=0; j < uidCount; ++j)
      {
	const Integer current_item_lid   = lids[j];
	const Int64   current_item_uid   = ids[j];
	const Integer current_item_owner = owners[j];
	if (current_item_owner != m_local_owner)
	  throw FatalErrorException("Non local EntryIndex requested");

	InternalEntryIndex lookup_entry(currentEntry, current_item_lid, entry_kind,current_item_uid,0, current_creation_index, current_item_owner);

	EntryIndexMap::const_iterator lookup = std::lower_bound(entry_index.begin(), entry_index.end(), 
								lookup_entry,
								EntryIndexComparator());

	if ((lookup == entry_index.end()) || !(*lookup == lookup_entry))
	  throw FatalErrorException("Not locally defined entry requested");

	// Mise en place de la pre-valeur retour [avant renumérotation locale] (EntryIndex écrit sur un Int64)
	ids[j] = lookup->m_index;
      }
    }

    { // Préparation des retours
      Integer dest = recvRequest.comm->destRank(); // Attention à l'ordre bizarre
      Integer orig = recvRequest.comm->origRank(); //       de SerializeMessage
      delete recvRequest.comm;
      recvRequest.comm = new SerializeMessage(orig, dest, ISerializeMessage::MT_Send);
      messageList->addMessage(recvRequest.comm);

      SerializeBuffer& sbuf = recvRequest.comm->buffer();
      sbuf.setMode(ISerializer::ModeReserve); // phase préparatoire
      sbuf.reserve(nameString); // Chaine de caractère du nom de l'entrée
      sbuf.reserveInteger(1); // Nb d'item
      sbuf.reserveInteger(uidCount); // Les index
      sbuf.allocateBuffer(); // allocation mémoire
      sbuf.setMode(ISerializer::ModePut);
      sbuf.put(nameString);
      sbuf.put(uidCount);
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
  m_parallel_mng->allGather(myLocalSize,allLocalSizes);

  // Table de ré-indexation (EntryIndex->Integer)
  Arcane::IntegerUniqueArray entry_reindex(m_local_entry_count+m_global_entry_count);
  entry_reindex.fill(-1); // valeur de type Erreur par défaut

  // Calcul de la taille des indices par entrée
  reserveEntries(entry_index);

  // Mise à jour du contenu des entrées
  m_global_entry_offset = 0;
  for(Integer i=0; i<m_parallel_mng->commRank(); ++i)
  {
    m_global_entry_offset += allLocalSizes[i];
  }
  // Utiliser MPI_Scan ? (equivalent Arcane)

  // C'est ici et uniquement ici qu'est matérialisé l'ordre des entrées
  Integer currentEntryIndex = m_global_entry_offset; // commence par l'offset local
  for(EntryIndexMap::iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  {
    const InternalEntryIndex & entryIndex = *i;
    ARCANE_ASSERT((entryIndex.m_entry != NULL),("Unexpected null entry"));
    const Integer item_owner = entryIndex.m_owner;
    if (item_owner == m_local_owner)
    { // Numérotation locale !
      const Integer newIndex = currentEntryIndex++;
      entry_reindex[i->m_index+m_global_entry_count] = newIndex; // Table de translation
      i->m_index = newIndex;
    }
  }

  // 5 - Envoie des retours (EntryIndex globaux)
  for(RecvRequests::iterator i = recvRequests.begin(); i != recvRequests.end(); ++i)
  {
    EntryRecvRequest & recvRequest = *i;
    SerializeBuffer& sbuf = recvRequest.comm->buffer();
    Int64UniqueArray & ids = recvRequest.ids;
    for(Integer j = 0; j<ids.size(); ++j)
    {
      sbuf.putInteger(entry_reindex[ids[j]+m_global_entry_count]); // Via la table de réindexation
    }
  }

  // Table des buffers de retour
  typedef std::list<SerializeMessage *> ReturnedRequests;
  ReturnedRequests returnedRequests;

  // Acces rapide aux buffers connaissant le proc emetteur et le nom d'une entrée
  /* Car on ne peut tager les buffers donc l'entrée reçue dans un buffer est non déterminée
   * surtout si 2 domaines se communiquent plus d'une entrée
   */
  typedef std::map<Integer, EntrySendRequest*> SubFastReturnMap;
  typedef std::map<String, SubFastReturnMap> FastReturnMap;
  FastReturnMap fastReturnMap;

  // Préparation des réceptions [sens inverse]
  for(SendRequests::iterator i = sendRequests.begin(); i != sendRequests.end(); ++i)
  {
    const Integer destDomainId = i->first;
    SendRequestByEntry & requests = i->second;
    for(SendRequestByEntry::iterator j = requests.begin(); j != requests.end(); ++j)
    {
      EntrySendRequest & request = j->second;
      EntryImpl * entryImpl = j->first;
      const String nameString = entryImpl->getName();

      // On ne peut pas associer directement le message à cette entrée
      // : dans le cas d'échange multiple il n'y pas de garantie d'arrivée
      // à la bonne place
      delete request.comm; request.comm = NULL;

      SerializeMessage * msg = new SerializeMessage(m_parallel_mng->commRank(),destDomainId,ISerializeMessage::MT_Recv);
      returnedRequests.push_back(msg);
      messageList->addMessage(msg);

      fastReturnMap[nameString][destDomainId] = &request;
    }
  }

  // Traitement des communications
  messageList->processPendingMessages();
  messageList->waitMessages(Parallel::WaitAll);
  delete messageList; messageList = NULL; // Destruction propre de l'ancienne liste

  // 6 - Traitement des réponses
  // Association aux EntrySendRequest du buffer correspondant
  for(ReturnedRequests::iterator i = returnedRequests.begin(); i != returnedRequests.end(); ++i)
  {
    SerializeMessage* message = *i;
    const Integer origDomainId = message->destRank();
    SerializeBuffer& sbuf = message->buffer();
    sbuf.setMode(ISerializer::ModeGet);
    String nameString;
    sbuf.get(nameString);
    ARCANE_ASSERT((fastReturnMap[nameString][origDomainId] != NULL),("Inconsistency detected"));
    EntrySendRequest & request = *fastReturnMap[nameString][origDomainId];
    request.comm = *i; // Reconnection pour accès rapide depuis l'EntrySendRequest
#ifdef ARCANE_DEBUG_ASSERT
    const Integer idCount = sbuf.getInteger();
#else 
    sbuf.getInteger();
#endif
    ARCANE_ASSERT((request.count == idCount),("Inconsistency detected"));
  }

  // Distribution des reponses
  // Par parcours dans ordre initial (celui de la demande)
  for(EntryIndexMap::iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  {
    const InternalEntryIndex & entryIndex = *i;
    const Integer item_owner = entryIndex.m_owner;
    if (item_owner != m_local_owner)
    {
      EntryImpl * entryImpl = entryIndex.m_entry;
      EntrySendRequest & request = sendRequests[item_owner][entryImpl];
      ARCANE_ASSERT((request.count > 0),("Unexpected empty request"));
      --request.count;
      SerializeBuffer& sbuf = request.comm->buffer();
      const Integer newIndex = sbuf.getInteger();
      entry_reindex[i->m_index+m_global_entry_count] = newIndex;
      i->m_index = newIndex;
    }
  }

  // Calcul de la taille global d'indexation (donc du système associé)
  m_global_entry_count = 0;
  for(Integer i=0; i<m_parallel_mng->commSize(); ++i)
  {
    m_global_entry_count    += allLocalSizes[i];
  }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
sequential_prepare(EntryIndexMap & entry_index)
{
  ARCANE_ASSERT((not m_parallel_mng->isParallel() || m_parallel_mng->commSize() <= 1),("Sequential mode expected"));
  ARCANE_ASSERT((m_global_entry_count == 0),("Unexpected global entries (%d)",m_global_entry_count));

  // Très similaire à la section parallèle :
  // 4 - Indexation locale
  /* La politique naive ici appliquée est de numéroter tous les
   * (Entry,Item) locaux d'abord.
   */

  // Table de ré-indexation (EntryIndex->Integer)
  Arcane::IntegerUniqueArray entry_reindex(m_local_entry_count+m_local_removed_entry_count);
  entry_reindex.fill(-1); // valeur de type Erreur par défaut

  // Calcul de la taille des indices par entrée
  reserveEntries(entry_index);

  // Mise à jour du contenu des entrées
  // Pas d'offset car séquentiel
  m_global_entry_offset = 0;

  // C'est ici et uniquement ici qu'est matérialisé l'ordre des entrées
  Integer currentEntryIndex = m_global_entry_offset; // commence par l'offset local
  for(EntryIndexMap::iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  {
    ARCANE_ASSERT((i->m_entry != NULL),("Unexpected null entry"));
    ARCANE_ASSERT((i->m_owner == m_local_owner),("Item cannot be non-local for sequential mode"));
    // Numérotation locale only !
    const Integer newIndex = currentEntryIndex++;
    entry_reindex[i->m_index+m_global_entry_count] = newIndex; // Table de translation
    i->m_index = newIndex;
  }

  m_global_entry_count = m_local_entry_count;
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
reserveEntries(const EntryIndexMap & entry_index)
{
  // Calcul de la taille des indices par entrée
  std::map<const EntryImpl *,Integer> count_table;
  for(EntryIndexMap::const_iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  {
    const EntryImpl * entryImpl = i->m_entry;
    count_table[entryImpl]++;
  }

  // Dimensionnement des buffers de chaque entrée
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
  {
    MyEntryImpl * entry = i->second;
    entry->reserve(count_table[entry]);
    //      if (m_trace) m_trace->pinfo() << "Entry " << entry->getName() << " size = " << count_table[entry];
  }

  //   // Calcul de la taille des indices par entrée
  //   IntegerUniqueArray count_table(m_creation_index, 0);
  //   for(EntryIndexMap::const_iterator i = entry_index.begin(); i != entry_index.end(); ++i)
  //     {
  //       count_table[i->m_entry->getCreationIndex()]++;
  //     }

  //   // Dimensionnement des buffers de chaque entrée
  //   for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
  //     {
  //       MyEntryImpl * entry = i->second;
  //       entry->reserve(count_table[entry->getCreationIndex()]);
  //       //      if (m_trace) m_trace->pinfo() << "Entry " << entry->getName() << " size = " << count_table[entry];
  //     }
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManager::
getIndex(const Entry & entry, const Item & item) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ARCANE_ASSERT((entry.manager() == this),("Incompatible entry from another manager"));
  MyEntryImpl * myEntry = static_cast<MyEntryImpl*>(entry.internal());
  ConstArrayView<Integer> localIds = myEntry->getAllLocalIds();
  ConstArrayView<Integer> indices  = myEntry->getAllIndexes();

  const Integer itemLocalId = item.localId();
  Integer i = 0;
  while(i<localIds.size() && localIds[i] != itemLocalId) ++i;
  if (i == localIds.size()) throw FatalErrorException(A_FUNCINFO,"Cannot get undefined entry index");
  return indices[i];
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
getIndex(const ScalarIndexSet & entry, const Arcane::ItemVectorView & items, ArrayView<Integer> indexes) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");
  if (items.size() != indexes.size())
    throw FatalErrorException(A_FUNCINFO,"Inconsistent sizes");

  ARCANE_ASSERT((entry.manager() == this),("Incompatible entry from another manager"));
  MyEntryImpl * myEntry = static_cast<MyEntryImpl*>(entry.internal());

  ConstArrayView<Integer> localIds = myEntry->getAllLocalIds();
  ConstArrayView<Integer> indices  = myEntry->getAllIndexes();

  ENUMERATE_ITEM(iitem,items) {
    const Integer itemLocalId = iitem.localId();
    Integer i = 0;
    while(i<localIds.size() && localIds[i] != itemLocalId) ++i;
    if (i == localIds.size()) throw FatalErrorException(A_FUNCINFO,"Cannot get undefined entry index");
    indexes[iitem.index()] = indices[i];
  }
}

/*---------------------------------------------------------------------------*/

IntegerUniqueArray
BasicIndexManager::
getIndexes(const Entry & entry) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ARCANE_ASSERT((entry.manager() == this),("Incompatible entry from another manager"));
  MyEntryImpl * en = static_cast<MyEntryImpl*>(entry.internal());
  ARCANE_ASSERT((en != NULL),("Unexpected null entry"));
  const IAbstractFamily & family = en->getFamily();
  IntegerUniqueArray allIds(family.maxLocalId(),nullIndex());
  const IntegerConstArrayView allIndices = en->getAllIndexes();
  const IntegerConstArrayView allLocalIds = en->getAllLocalIds();
  const Integer size = allIndices.size();
  for(Integer i=0;i<size;++i)
    allIds[allLocalIds[i]] = allIndices[i];
  return allIds;
}

/*---------------------------------------------------------------------------*/

UniqueArray2<Integer>
BasicIndexManager::
getIndexes(const VectorIndexSet & entries) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  Integer max_family_size = 0;
  for(Integer entry=0;entry<entries.size();++entry)
  {
    // controles uniquement en première passe
    ARCANE_ASSERT((entries[entry].manager() == this),("Incompatible entry from another manager"));
    MyEntryImpl * en = static_cast<MyEntryImpl*>(entries[entry].internal());
    ARCANE_ASSERT((en != NULL),("Unexpected null entry"));
    const IAbstractFamily & family = en->getFamily();
    max_family_size = math::max(max_family_size,family.maxLocalId());
  }

  UniqueArray2<Integer> allIds(max_family_size,entries.size());
  allIds.fill(nullIndex());

  for(Integer entry=0;entry<entries.size();++entry)
  {
    MyEntryImpl * en = static_cast<MyEntryImpl*>(entries[entry].internal());
    const IntegerConstArrayView allIndices = en->getAllIndexes();
    const IntegerConstArrayView allLocalIds = en->getAllLocalIds();
    const Integer size = allIndices.size();
    for(Integer i=0;i<size;++i)
      allIds[allLocalIds[i]][entry] = allIndices[i];
  }
  return allIds;
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
stats(Integer & globalSize,
      Integer & minLocalIndex,
      Integer & localSize) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  globalSize = m_global_entry_count;
  minLocalIndex = m_global_entry_offset;
  localSize = m_local_entry_count;
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManager::
globalSize() const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  return m_global_entry_count;
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManager::
minLocalIndex() const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  return m_global_entry_offset;
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManager::
localSize() const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  return m_local_entry_count;
}

/*---------------------------------------------------------------------------*/

IIndexManager::EntryEnumerator
BasicIndexManager::
enumerateEntry() const
{
  return EntryEnumerator(new MyEntryEnumeratorImpl(m_entry_set));
}

/*---------------------------------------------------------------------------*/

bool
BasicIndexManager::EntryIndexComparator::
operator()(const BasicIndexManager::InternalEntryIndex & a,
           const BasicIndexManager::InternalEntryIndex & b) const
{
  const MyEntryImpl * aEntry = a.m_entry;
  const MyEntryImpl * bEntry = b.m_entry;
  if (a.m_kind != b.m_kind)
    return a.m_kind < b.m_kind;
  else if (a.m_uid != b.m_uid)
    return a.m_uid < b.m_uid;
  else
    return aEntry->getCreationIndex() < bEntry->getCreationIndex();
  // return a.m_creation_index < b.m_creation_index;
}

/*---------------------------------------------------------------------------*/
IIndexManager::ScalarIndexSet
BasicIndexManager::
buildScalarIndexSet(const String name, Arcane::IItemFamily * item_family)
{
  Integer kind = kindFromItemFamily(item_family);
  std::shared_ptr<IAbstractFamily> & family = m_abstract_families[kind];
  if (!family) family.reset(new ItemAbstractFamily(item_family));
  Entry en = buildEntry(name, family.get(), kind);
  return en;
}

void
BasicIndexManager::
defineIndex(IIndexManager::ScalarIndexSet& entry, const Arcane::ItemGroup & itemGroup)
{
  defineIndex(entry,itemGroup.view().localIds());
}

IIndexManager::ScalarIndexSet
BasicIndexManager::
buildScalarIndexSet(const String name, const Arcane::ItemGroup & itemGroup)
{
  Arcane::IItemFamily * item_family = itemGroup.itemFamily();
  Integer kind = kindFromItemFamily(item_family);
  std::shared_ptr<IAbstractFamily> & family = m_abstract_families[kind];
  if (!family) family.reset(new ItemAbstractFamily(item_family));
  Entry en = buildEntry(name, family.get(), kind);
  defineIndex(en,itemGroup.view().localIds());
  return en;
}

/*---------------------------------------------------------------------------*/

IIndexManager::ScalarIndexSet
BasicIndexManager::
buildScalarIndexSet(const String name, const IntegerConstArrayView localIds, const IAbstractFamily & family)
{
  Entry en = buildEntry(name, &family, addNewAbstractFamily(&family));
  defineIndex(en,localIds);
  return en;
}

/*---------------------------------------------------------------------------*/

IIndexManager::ScalarIndexSet
BasicIndexManager::
buildScalarIndexSet(const String name, const IAbstractFamily & family)
{
  Int32UniqueArray localIds = family.allLocalIds();
  Entry en = buildEntry(name, &family, addNewAbstractFamily(&family));
  defineIndex(en,localIds.view());
  return en;
}

/*---------------------------------------------------------------------------*/

IIndexManager::VectorIndexSet
BasicIndexManager::
buildVectorIndexSet(const String name, const Arcane::ItemGroup & itemGroup, const Integer n)
{
  VectorIndexSet ens(n);
  const IntegerConstArrayView localIds = itemGroup.view().localIds();
  Arcane::IItemFamily * item_family = itemGroup.itemFamily();
  Integer kind = kindFromItemFamily(item_family);
  std::shared_ptr<IAbstractFamily> & family = m_abstract_families[kind];
  if (!family) family.reset(new ItemAbstractFamily(item_family));
  for(Integer i=0;i<n;++i)
  {
    ens[i] = buildEntry(String::format("{0}[{1}]",name,i),family.get(),itemGroup.itemKind());
    defineIndex(ens[i],localIds);
  }
  return ens;
}

/*---------------------------------------------------------------------------*/

IIndexManager::VectorIndexSet
BasicIndexManager::
buildVectorIndexSet(const String name, const IntegerConstArrayView localIds, const IAbstractFamily & family, const Integer n)
{
  VectorIndexSet ens(n);
  for(Integer i=0;i<n;++i)
  {
    ens[i] = buildEntry(String::format("{0}[{1}]",name,i),&family,addNewAbstractFamily(&family));
    defineIndex(ens[i],localIds);
  }
  return ens;
}

/*---------------------------------------------------------------------------*/

IIndexManager::VectorIndexSet
BasicIndexManager::
buildVectorIndexSet(const String name, const IAbstractFamily & family, const Integer n)
{
  Int32UniqueArray localIds = family.allLocalIds();

  VectorIndexSet ens(n);
  for(Integer i=0;i<n;++i)
  {
    ens[i] = buildEntry(String::format("{0}[{1}]",name,i),&family,addNewAbstractFamily(&family));
    defineIndex(ens[i],localIds.view());
  }
  return ens;
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManager::
keepAlive(const IAbstractFamily * family)
{
  std::map<const IAbstractFamily *, Integer>::iterator finder = m_abstract_family_to_kind_map.find(family);
  if (finder == m_abstract_family_to_kind_map.end())
    return; // pas connu => on ne fait rien
  if (m_abstract_families[finder->second])
    throw FatalErrorException(A_FUNCINFO, "Already known as kept alive abstract family");

  IAbstractFamily * new_family = family->clone();
  m_abstract_families[finder->second].reset(new_family);
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
  {
    MyEntryImpl * entry = i->second;
    if (&entry->getFamily() == family)
      entry->resetFamily(new_family);
  }
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManager::
addNewAbstractFamily(const IAbstractFamily * family)
{
  std::map<const IAbstractFamily *, Integer>::iterator finder = m_abstract_family_to_kind_map.find(family);
  if (finder == m_abstract_family_to_kind_map.end())
  {
    const Integer newKind = m_abstract_family_base_kind++;
    ARCANE_ASSERT((newKind<KIND_SHIFT),("Unexpected kind overflow"));
    m_abstract_family_to_kind_map[family] = newKind;
    m_abstract_families[newKind] = std::shared_ptr<IAbstractFamily>(); // this place will be used when the family memory management will be delegated to this class
    return newKind;
  }
  else
  {
    return finder->second;
  }
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManager::
kindFromItemFamily(const Arcane::IItemFamily * family)
{
  IMesh * mesh = family->mesh();
  std::pair<std::map<IMesh *, Integer>::iterator, bool> inserter = m_item_family_meshes.insert(std::pair<IMesh*,Integer>(mesh,-1));
  std::map<IMesh *, Integer>::iterator mesh_iterator = inserter.first;
  if (inserter.second) // new element
    mesh_iterator->second = m_item_family_meshes.size();
  const Integer mesh_id = mesh_iterator->second;
  const Integer base_kind = family->itemKind();
  ARCANE_ASSERT((base_kind<KIND_SHIFT),("Unexpected kind overflow"));
  const Integer newKind = mesh_id * KIND_SHIFT + base_kind;
  return newKind;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
