// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "BasicIndexManagerImpl.h"
#include <arcane/IItemFamily.h>
#include <arcane/utils/Collection.h>
#include <arcane/utils/Enumerator.h>
#include <arcane/IParallelMng.h>
#include <arcane/ISerializeMessageList.h>
#include <arcane/SerializeMessage.h>
#include <arcane/utils/CString.h>
#include <arcane/ArcaneVersion.h>
#include <arcane/utils/Math.h>

#include <map>
#include <list>

#include "VariableUpdateImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/** \brief Squelette de l'implementation locale de Entry
 *  Les vraies implémentations sont MyVariableEntryImpl et MyAbstractEntryImpl.
 *  Cette classe permet de la factorisation de code
 */
class BasicIndexManagerImpl::MyEntryImpl :
  public IIndexManager::EntryImpl
{
  friend class BasicIndexManagerImpl;
public:
  //! Constructeur par défaut
  MyEntryImpl(const Integer creationIndex)
    : m_creation_index(creationIndex)
  {
    ;
  }

  virtual ~MyEntryImpl()
  {
    ;
  }

  ConstArrayView<Integer> getIndex() const
  {
    return ConstArrayView<Integer>(indices);
  }

  void setInitializer(Initializer * initializer)
  {
    m_initializer = initializer;
  }

  bool needInit() const
  {
    return m_initializer.get() != NULL;
  }

  void initValues(Array<Real> & values)
  {
    //ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));
    //ne peut etre utiliser avec des variables vectorielles (dim=1)
    m_initializer->init(items,values);
  }

  void setUpdater(Updater * updater)
  {
    m_updater = updater;
  }

  bool needUpdate() const
  {
    return m_updater.get() != NULL;
  }

  void updateValues(const Array<Real> & values)
  {
    //ARCANE_ASSERT((items.size() == values.size()),("Incompatible items and values sizes"));
    //ne peut etre utiliser avec des variables vectorielles (dim=1)
    m_updater->update(items,values);
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

protected:
  //! Préparation des buffers d'indices et d'items
  /*! Utilisation réservée au 'friend' BasicIndexManagerImpl */
  void reserve(const Integer n)
  {
    items.reserve(n);
    indices.reserve(n);
  }

  //! Ajout d'un item/indice pour l'entrée courante
  /*! Utilisation réservée au 'friend' BasicIndexManagerImpl */
  void add(const Item & item, const Integer id)
  {
    items.add(item);
    indices.add(id);
  }

protected:
  BufferT<Item> items;
  BufferT<Integer> indices;

private:
  AutoRefT<Initializer> m_initializer;
  AutoRefT<Updater> m_updater;

  std::map<String,String> m_tags;
  Integer m_creation_index;

public:
  Integer getCreationIndex() const { return m_creation_index; }
};

/*---------------------------------------------------------------------------*/

struct MyVariableEntryImpl :
  public BasicIndexManagerImpl::MyEntryImpl
{
  friend class BasicIndexManagerImpl;
public:
  //! Constructeur d'une entrée associée à une variable
  MyVariableEntryImpl(IVariable * ivar, const Integer creationIndex) :
    MyEntryImpl(creationIndex),
    m_ivariable(ivar)
  {
    ;
  }

  ~MyVariableEntryImpl()
  {
    ;
  }

  IVariable * getVariable() const
  {
    return m_ivariable;
  }

  String getName() const
  {
    return m_ivariable->name();
  }

  eItemKind getKind() const
  {
    return m_ivariable->itemKind();
  }

  const IItemFamily * getItemFamily() const
  {
    return m_ivariable->itemFamily();
  }

private:
  IVariable * m_ivariable;
};

/*---------------------------------------------------------------------------*/

struct MyAbstractEntryImpl :
  public BasicIndexManagerImpl::MyEntryImpl
{
  friend class BasicIndexManagerImpl;
public:
  //! Constructeur d'une entrée abstraite
  MyAbstractEntryImpl(const String name, const IItemFamily * itemFamily, const Integer creationIndex) :
    MyEntryImpl(creationIndex),
    m_name(name),
    m_item_family(itemFamily)
  {
    ;
  }

  ~MyAbstractEntryImpl()
  {
    ;
  }

  IVariable * getVariable() const
  {
    return NULL;
  }

  String getName() const
  {
    return m_name;
  }

  eItemKind getKind() const
  {
    return m_item_family->itemKind();
  }

  const IItemFamily * getItemFamily() const
  {
    return m_item_family;
  }

private:
  const String m_name;
  const IItemFamily * m_item_family;
};

/*---------------------------------------------------------------------------*/

class BasicIndexManagerImpl::MyEquationImpl :
  public IIndexManager::EquationImpl
{
  friend class BasicIndexManagerImpl;
public:
  //! Constructeur
  MyEquationImpl(const String & name, const Entry & entry) :
    m_name(name),
    m_item_family(entry.getItemFamily()),
    m_entry(NULL)
  {
    m_entry = dynamic_cast<MyEntryImpl*>(entry.internal());
    ARCANE_ASSERT((m_entry != NULL),("Undefined Entry"));
  }

  //! Constructeur
  MyEquationImpl(const String & name, const IItemFamily * itemFamily) :
    m_name(name),
    m_item_family(itemFamily),
    m_entry(NULL)
  {
    throw FatalErrorException("Not implemented in BasicIndexManagerImpl::prepare");
  }

  virtual ~MyEquationImpl()
  {
    ;
  }

  Entry getEntry() const
  {
    return m_entry;
  }

  String getName() const
  {
    return m_name;
  }

  eItemKind getKind() const
  {
    return m_item_family->itemKind();
  }

  const IItemFamily * getItemFamily() const
  {
    return m_item_family;
  }

public:
  //! Acces à l'entrée associée sous sa forme primitive
  /*! Extension spécifique à MyEquationImpl de l'interface */
  MyEntryImpl * getInternalEntry() const
  {
    return m_entry;
  }

private:
  String m_name;
  const IItemFamily * m_item_family;
  MyEntryImpl * m_entry;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct BasicIndexManagerImpl::EntrySendRequest
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

  SerializeMessage * comm;
  Integer count;
};

/*---------------------------------------------------------------------------*/

struct BasicIndexManagerImpl::EntryRecvRequest
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

  SerializeMessage * comm;
  Array<Int64> ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicIndexManagerImpl::MyEntryEnumeratorImpl
  : public IIndexManager::EntryEnumeratorImpl
{
protected:
  EntrySet::iterator m_iter, m_end;
public:
  MyEntryEnumeratorImpl(EntrySet & entries)
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

BasicIndexManagerImpl::
~BasicIndexManagerImpl()
{
  this->init();
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
init()
{
  m_state = Initialized;

  m_entry_index.clear();
  m_equation_index.clear();

  m_entry_reindex.clear();
  m_equation_reindex.clear();

  m_local_entry_count = 0;
  m_global_entry_count = 0;
  m_local_equation_count = 0;

  m_var_entry.clear();

  // Destruction des structure de type entry et equation
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
    {
      delete i->second;
    }
  m_entry_set.clear();

  for(EquationSet::iterator i = m_equation_set.begin(); i != m_equation_set.end(); ++i)
    {
      delete i->second;
    }
  m_equation_set.clear();
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
setTraceMng(ITraceMng * traceMng)
{
  m_trace = traceMng;
}

/*---------------------------------------------------------------------------*/

IIndexManager::Entry
BasicIndexManagerImpl::
buildAbstractEntry(const String name, const IItemFamily * itemFamily)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  // Recherche de l'entrée d'une variable
  std::pair<EntrySet::iterator,bool> lookup
    = m_entry_set.insert(EntrySet::value_type(name,NULL));
  if (lookup.second)
    {
      MyEntryImpl * entry = new MyAbstractEntryImpl(name,itemFamily,m_creation_index++);
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
BasicIndexManagerImpl::
buildVariableEntry(IVariable * ivar, const eVariableAccess mode)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  // Recherche de l'entrée d'une variable
  std::pair<VarEntryMap::iterator,bool> lookup
    = m_var_entry.insert(VarEntryMap::value_type(ivar,NULL));
  if (lookup.second)
    {
      MyEntryImpl * entry = new MyVariableEntryImpl(ivar,m_creation_index++);
      ARCANE_ASSERT((ivar == entry->getVariable()),("Inconsistent variable access"));
      std::pair<EntrySet::iterator,bool> slookup
        = m_entry_set.insert(EntrySet::value_type(entry->getName(),entry));
      if (not slookup.second)
        { // Big Pb si l'insertion dans la table d'indexation par nom n'est pas possible
          throw FatalErrorException(A_FUNCINFO,"Variable entry conflicting with Abstract entry");
          return NULL;
        }
      lookup.first->second = entry;

      switch (mode)
        {
        case Direct:
          entry->setInitializer(new MeshVariableCopyInitializer(ivar));
          entry->setUpdater(new MeshVariableCopyUpdater(ivar));
          break;
        case Incremental:
          entry->setInitializer(new MeshVariableZeroInitializer(ivar));
          entry->setUpdater(new MeshVariableCumulativeUpdater(ivar));
          break;
        case Undefined:
          break;
        default:
          throw FatalErrorException("Undefined eVariableAccess enum value");
        }
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
BasicIndexManagerImpl::
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

IIndexManager::Equation
BasicIndexManagerImpl::
buildEquation(const String name, const IItemFamily * itemFamily)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  throw FatalErrorException(A_FUNCINFO,"Equation not associated to Entry are not allowed");

  // Recherche de l'entrée d'une variable
  std::pair<EquationSet::iterator,bool> lookup
    = m_equation_set.insert(EquationSet::value_type(name,NULL));
  if (lookup.second)
    {
      MyEquationImpl * equation = new MyEquationImpl(name,itemFamily);
      lookup.first->second = equation;
      return equation;
    }
  else
    {
      throw FatalErrorException(A_FUNCINFO,"Already defined equation");
      return NULL;
    }
}

/*---------------------------------------------------------------------------*/

IIndexManager::Equation
BasicIndexManagerImpl::
buildEquation(const String name, const Entry & entry)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  // Recherche de l'entrée d'une variable
  std::pair<EquationSet::iterator,bool> lookup
    = m_equation_set.insert(EquationSet::value_type(name,NULL));
  if (lookup.second)
    {
      MyEquationImpl * equation = new MyEquationImpl(name,entry);
      lookup.first->second = equation;
      return equation;
    }
  else
    {
      throw FatalErrorException(A_FUNCINFO,"Already defined equation");
      return NULL;
    }
}

/*---------------------------------------------------------------------------*/

IIndexManager::Equation
BasicIndexManagerImpl::
getEquation(const String name)
{
  const EquationSet::const_iterator lookup = m_equation_set.find(name);
  if (lookup == m_equation_set.end())
    {
      throw FatalErrorException(A_FUNCINFO,"Cannot get undefined equation index");
    }

  return lookup->second;
}

/*---------------------------------------------------------------------------*/

IIndexManager::EntryIndex
BasicIndexManagerImpl::
defineEntryIndex(const Entry & entry, const Item & item)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  const InternalEntryIndex entryIndex(dynamic_cast<MyEntryImpl*>(entry.internal()),item);
  std::pair<EntryIndexMap::iterator,bool> lookup
    = m_entry_index.insert(EntryIndexMap::value_type(entryIndex,0));

  if (lookup.second)
    { // nouvelle entrée
      if (item.isOwn())
        {
          return (lookup.first->second = +(m_local_entry_count++));
        }
      else
        {
          return (lookup.first->second = -(++m_global_entry_count));
        }
    }
  else
    { // existe déjà
      return lookup.first->second;
    }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
defineEntryIndex(const Entry & entry, const ItemGroup & itemGroup)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ENUMERATE_ITEM(i,itemGroup)
    {
      const Item & item = *i;
      const InternalEntryIndex entryIndex(dynamic_cast<MyEntryImpl*>(entry.internal()),item);
      std::pair<EntryIndexMap::iterator,bool> lookup
        = m_entry_index.insert(EntryIndexMap::value_type(entryIndex,0));

      if (lookup.second)
        { // nouvelle entrée
          if (item.isOwn())
            {
              (lookup.first->second = +(m_local_entry_count++));
            }
          else
            {
              (lookup.first->second = -(++m_global_entry_count));
            }
        }
    }
}

/*---------------------------------------------------------------------------*/

IIndexManager::EquationIndex
BasicIndexManagerImpl::
defineEquationIndex(const Equation & equation, const Item & item)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  if (not item.isOwn())
    throw FatalErrorException(A_FUNCINFO,"Cannot define non-local equation");

  const InternalEquationIndex equationIndex(dynamic_cast<MyEquationImpl*>(equation.internal()),item);
  std::pair<EquationIndexMap::iterator,bool> lookup
    = m_equation_index.insert(EquationIndexMap::value_type(equationIndex,0));

  if (lookup.second)
    { // nouvelle entrée
      return (lookup.first->second = (m_local_equation_count++));
    }
  else
    { // existe déjà
      return lookup.first->second;
    }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
defineEquationIndex(const Equation & equation, const ItemGroup & itemGroup)
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ENUMERATE_ITEM(i,itemGroup)
    {
      const Item & item = *i;
      if (not item.isOwn())
        throw FatalErrorException(A_FUNCINFO,"Cannot define non-local equation");

      const InternalEquationIndex equationIndex(dynamic_cast<MyEquationImpl*>(equation.internal()),item);
      std::pair<EquationIndexMap::iterator,bool> lookup
        = m_equation_index.insert(EquationIndexMap::value_type(equationIndex,0));

      if (lookup.second)
        { // nouvelle entrée
          (lookup.first->second = (m_local_equation_count++));
        }
    }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
prepare()
{
  if (m_state != Initialized)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  ARCANE_ASSERT(((Integer)m_entry_index.size() == m_local_entry_count + m_global_entry_count),("Inconsistent global size"));
  ARCANE_ASSERT(((Integer)m_equation_index.size() == m_local_equation_count),("Inconsistent local size"));
  if (m_trace) {
    m_trace->info() << "Entry ordering :";
    for(EntrySet::iterator i=m_entry_set.begin(); i != m_entry_set.end(); ++i)
      {
        m_trace->info() << "\tEntry '" << i->first << "' placed at rank " << i->second->getCreationIndex();
      }
    m_trace->info() << "Equation / Entry counts = " << m_local_equation_count << " / " << m_local_entry_count;
  }

  if (m_parallel_mng->isParallel() and m_parallel_mng->nbSubDomain() > 1)
    parallel_prepare();
  else
    sequential_prepare();

  m_state = Prepared;
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
parallel_prepare()
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
  for(EntryIndexMap::const_iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      const InternalEntryIndex & entryIndex = i->first;
      EntryImpl * entryImpl = entryIndex.first;
      const Item & item = entryIndex.second;
      if (not item.isOwn())
        {
          // 	  if (m_trace) m_trace->pinfo() << item.localId() << " : " << item.uniqueId() << " is owned by " << item.owner() << " with localIndex=" << i->second;
          sendRequests[item.owner()][entryImpl].count++;
        }
      else
        {
          // 	  if (m_trace) m_trace->pinfo() << item.localId() << " : " << item.uniqueId() << " is local with localIndex=" << i->second;
        }
    }

  // Liste de synthèse des messages (emissions / réceptions)
  messageList = m_parallel_mng->createSerializeMessageList();

  // Contruction de la table de communications + préparation des messages d'envoi
  Array<Integer> sendToDomains(2*m_parallel_mng->commSize(),0);

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
  for(EntryIndexMap::const_iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      const InternalEntryIndex & entryIndex = i->first;
      EntryImpl * entryImpl = entryIndex.first;
      const Item & item = i->first.second;
      if (not item.isOwn())
        sendRequests[item.owner()][entryImpl].comm->buffer().put(item.uniqueId().asInt64());
    }

  // Réception des annonces de demandes (les nombres d'entrée + taille)
  Array<Integer> recvFromDomains(2*m_parallel_mng->commSize());
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

#warning "CHECK: optimisable ?"
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

        // Passage de l'uid à l'item associé (travaille sur place : pas de recopie)
        Array<Int64> & ids = recvRequest.ids;
        Array<Int32> lids;
        lids.resize(ids.size());
        
#warning "OLD: not a const attribute access ; check internal Arcane implementation"
        IItemFamily * itemFamily = const_cast<IItemFamily*>(currentEntry->getItemFamily());
        itemFamily->itemsUniqueIdToLocalId(lids,ids,true);
        ItemInternalList list = itemFamily->itemsInternal();

        // Vérification d'intégrité : toutes les entrées demandées sont définies localement
        for(Integer j=0; j < uidCount; ++j)
          {
            Item currentItem(list[lids[j]]);
            if (not currentItem.isOwn())
              throw FatalErrorException("Non local EntryIndex requested");

            EntryIndexMap::const_iterator lookup = m_entry_index.find(InternalEntryIndex(currentEntry,currentItem));
            if (lookup == m_entry_index.end())
              throw FatalErrorException("Not locally defined entry requested");

            // Mise en place de la pre-valeur retour [avant renumérotation locale] (EntryIndex écrit sur un Int64)
            lids[j] = lookup->second;
          }
      }

      { // Préparation des retours
        Integer dest = recvRequest.comm->destSubDomain(); // Attention à l'ordre bizarre
        Integer orig = recvRequest.comm->origSubDomain(); //       de SerializeMessage
        delete recvRequest.comm;
        recvRequest.comm = new SerializeMessage(orig,dest,ISerializeMessage::MT_Send);
        messageList->addMessage(recvRequest.comm);

        SerializeBuffer & sbuf = recvRequest.comm->buffer();
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
  /*  Pour que l'indexation des equations portées par une Entry soit *
   * diagonale avec les dites Entry, il est ici supposé que :
   * - pour (Equation,Item) il existe (Entry,Item) où Entry est l'Entry *
   * associée à Equation,
   * - bien sûr que l'Item associé est local,
   * - deux Equations ne peuvent être portées par le même Entry.
   *
   * La politique naive ici appliquée est de numéroter tous les
   * (Entry,Item) locaux d'abord, puis d'affecter l'Index ainsi
   * calculée aux équations (le système ne comprend donc pas
   * d'équations non associées à une Entry et est obligatoirement
   * localement carré)
   */
  // Calcul de des offsets globaux sur Entry et Equation (via les tailles locales)
  Array<Integer> allLocalSizes(m_parallel_mng->commSize()*2);
  Array<Integer> myLocalSize(2);
  myLocalSize[0] = m_local_equation_count;
  myLocalSize[1] = m_local_entry_count;
  m_parallel_mng->allGather(myLocalSize,allLocalSizes);

  // Table de ré-indexation (EquationIndex->Integer et EntryIndex->Integer)
  m_entry_reindex.resize(m_local_entry_count+m_global_entry_count);
  m_entry_reindex.fill(-1); // valeur de type Erreur par défaut
  m_equation_reindex.resize(m_local_equation_count);
  m_equation_reindex.fill(-1);

  // Calcul de la taille des indices par entrée (pour mise à jour optimisée des Initializer et Updater)
  std::map<const EntryImpl *,Integer> count_table;
  for(EntryIndexMap::const_iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      const EntryImpl * entryImpl = i->first.first;
      count_table[entryImpl]++;
    }

  // Dimensionnement des buffers de chaque entrée
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
    {
      MyEntryImpl * entry = i->second;
      entry->reserve(count_table[entry]);
//      if (m_trace) m_trace->pinfo() << "Entry " << entry->getName() << " size = " << count_table[entry];
    }

  // Mise à jour du contenu des entrées
  m_global_equation_offset = m_global_entry_offset = 0;
  for(Integer i=0; i<m_parallel_mng->commRank(); ++i)
    {
      m_global_equation_offset += allLocalSizes[2*i+0];
      m_global_entry_offset    += allLocalSizes[2*i+1];
    }
  // Utiliser MPI_Scan ? (equivalent Arcane)
  ARCANE_ASSERT((m_global_equation_offset == m_global_entry_offset),("Inconsistency detected")); // Vu que Equation <-> Entry dans cette implémentation

  Integer currentEntryIndex = m_global_entry_offset; // commence par l'offset local
  for(EntryIndexMap::iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      EntryImpl * entryImpl = i->first.first;
      MyEntryImpl * myEntryImpl = dynamic_cast<MyEntryImpl*>(entryImpl);
      ARCANE_ASSERT((myEntryImpl != NULL),("Unexpected null entry"));
      const Item & item = i->first.second;
      if (item.isOwn())
        { // Numérotation locale !
          const Integer newIndex = currentEntryIndex++;
          myEntryImpl->add(item,newIndex);
          m_entry_reindex[i->second+m_global_entry_count] = newIndex; // Table de translation
          i->second = newIndex;
        }
    }

  // Indexation directement suivant l'indexation des Entry
  for(EquationIndexMap::iterator i = m_equation_index.begin(); i != m_equation_index.end(); ++i)
    {
      EquationImpl * equationImpl = i->first.first;
      MyEquationImpl * myEquationImpl = dynamic_cast<MyEquationImpl*>(equationImpl);
      ARCANE_ASSERT((myEquationImpl != NULL),("Unexpected null equation"));
      const Item & item = i->first.second;
      ARCANE_ASSERT((item.isOwn()),("Only own item allowed")); // On ne peut indexer que des équations locales!
      EntryIndexMap::const_iterator ifind = m_entry_index.find(InternalEntryIndex(myEquationImpl->getInternalEntry(),item));
      if (ifind != m_entry_index.end())
        {
          m_equation_reindex[i->second] = ifind->second; // Table de translation
          i->second = ifind->second;
        }
      else
        {
          throw FatalErrorException(A_FUNCINFO,"Illegal equation indexation found");
        }
    }

  // 5 - Envoie des retours (EntryIndex globaux)
  for(RecvRequests::iterator i = recvRequests.begin(); i != recvRequests.end(); ++i)
    {
      EntryRecvRequest & recvRequest = *i;
      SerializeBuffer& sbuf = recvRequest.comm->buffer();
      Array<Int64> & ids = recvRequest.ids;
      for(Integer j = 0; j<ids.size(); ++j)
        {
          sbuf.putInteger(m_entry_reindex[ids[j]+m_global_entry_count]); // Via la table de réindexation
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
      SerializeMessage * message = *i;
      const Integer origDomainId = message->destSubDomain();
      SerializeBuffer& sbuf = message->buffer();
      sbuf.setMode(ISerializer::ModeGet);
      String nameString;
      sbuf.get(nameString);
      ARCANE_ASSERT((fastReturnMap[nameString][origDomainId] != NULL),("Inconsistency detected"));
      EntrySendRequest & request = *fastReturnMap[nameString][origDomainId];
      request.comm = *i; // Reconnection pour accès rapide depuis l'EntrySendRequest
      const Integer idCount = sbuf.getInteger();
      ARCANE_ASSERT((request.count == idCount),("Inconsistency detected"));
    }

  // Destribution des reponses
  // Par parcours dans ordre initial (celui de la demande)
  for(EntryIndexMap::iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      const InternalEntryIndex & entryIndex = i->first;
      const Item & item = entryIndex.second;
      if (not item.isOwn())
        {
          EntryImpl * entryImpl = entryIndex.first;
          EntrySendRequest & request = sendRequests[item.owner()][entryImpl];
          ARCANE_ASSERT((request.count > 0),("Unexpected empty request"));
          --request.count;
          SerializeBuffer& sbuf = request.comm->buffer();
          const Integer newIndex = sbuf.getInteger();
          m_entry_reindex[i->second+m_global_entry_count] = newIndex;
          i->second = newIndex;
        }
    }

  // Calcul de la taille global d'indexation (donc du système associé)
  m_global_equation_count = 0;
  m_global_entry_count = 0;
  for(Integer i=0; i<m_parallel_mng->commSize(); ++i)
    {
      m_global_equation_count += allLocalSizes[2*i+1];
      m_global_entry_count    += allLocalSizes[2*i+1];
    }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
sequential_prepare()
{
  ARCANE_ASSERT((not m_parallel_mng->isParallel()),("Sequential mode expected"));
  ARCANE_ASSERT((m_global_entry_count == 0),("Unexpected global entries (%d)",m_global_entry_count));

//   m_trace->fatal() << __PRETTY_FUNCTION__ << " not tested";

  // Très similaire à la section parallèle :
  // 4 - Indexation locale
  /*  Pour que l'indexation des equations portées par une Entry soit *
   * diagonale avec les dites Entry, il est ici supposé que :
   * - pour (Equation,Item) il existe (Entry,Item) où Entry est l'Entry *
   * associée à Equation,
   * - bien sûr que l'Item associé est local,
   * - deux Equations ne peuvent être portées par le même Entry.
   *
   * La politique naive ici appliquée est de numéroter tous les
   * (Entry,Item) locaux d'abord, puis d'affecter l'Index ainsi
   * calculée aux équations (le système ne comprend donc pas
   * d'équations non associées à une Entry et est obligatoirement
   * localement carré)
   */

  // Table de ré-indexation (EquationIndex->Integer et EntryIndex->Integer)
  m_entry_reindex.resize(m_local_entry_count);
  m_entry_reindex.fill(-1); // valeur de type Erreur par défaut
  m_equation_reindex.resize(m_local_equation_count);
  m_equation_reindex.fill(-1);

  // Calcul de la taille des indices par entrée (pour mise à jour optimisée des Initializer et Updater)
  std::map<const EntryImpl *,Integer> count_table;
  for(EntryIndexMap::const_iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      const EntryImpl * entryImpl = i->first.first;
      count_table[entryImpl]++;
    }

  // Dimensionnement des buffers de chaque entrée
  for(EntrySet::iterator i = m_entry_set.begin(); i != m_entry_set.end(); ++i)
    {
      MyEntryImpl * entry = i->second;
      entry->reserve(count_table[entry]);
//      if (m_trace) m_trace->pinfo() << "Entry " << entry->getName() << " size = " << count_table[entry];
    }

  // Mise à jour du contenu des entrées
  // Pas d'offset car séquentiel
  m_global_equation_offset = m_global_entry_offset = 0;

  Integer currentEntryIndex = m_global_entry_offset; // commence par l'offset local
  for(EntryIndexMap::iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    {
      EntryImpl * entryImpl = i->first.first;
      MyEntryImpl * myEntryImpl = dynamic_cast<MyEntryImpl*>(entryImpl);
      ARCANE_ASSERT((myEntryImpl != NULL),("Unexpected null entry"));
      const Item & item = i->first.second;
      if (item.isOwn())
        { // Numérotation locale !
          const Integer newIndex = currentEntryIndex++;
          myEntryImpl->add(item,newIndex);
          m_entry_reindex[i->second+m_global_entry_count] = newIndex; // Table de translation
          i->second = newIndex;
        }
    }

  // Indexation directement suivant l'indexation des Entry
  for(EquationIndexMap::iterator i = m_equation_index.begin(); i != m_equation_index.end(); ++i)
    {
      EquationImpl * equationImpl = i->first.first;
      MyEquationImpl * myEquationImpl = dynamic_cast<MyEquationImpl*>(equationImpl);
      ARCANE_ASSERT((myEquationImpl != NULL),("Unexpected null equation"));
      const Item & item = i->first.second;
      ARCANE_ASSERT((item.isOwn()),("Only own item allowed")); // On ne peut indexer que des équations locales!
      EntryIndexMap::const_iterator ifind = m_entry_index.find(InternalEntryIndex(myEquationImpl->getInternalEntry(),item));
      if (ifind != m_entry_index.end())
        {
          m_equation_reindex[i->second] = ifind->second; // Table de translation
          i->second = ifind->second;
        }
      else
        {
          throw FatalErrorException(A_FUNCINFO,"Illegal equation indexation found");
        }
    }

  m_global_entry_count = m_local_entry_count;
  m_global_equation_count = m_local_equation_count;
}

/*---------------------------------------------------------------------------*/

IIndexManager::Entry
BasicIndexManagerImpl::
getVariableEntry(IVariable * ivar) const
{
  // Recherche de l'entrée d'une variable
  const VarEntryMap::const_iterator lookup = m_var_entry.find(ivar);
  if (lookup == m_var_entry.end())
    throw FatalErrorException(A_FUNCINFO,"Undefined var entry");

  return lookup->second;
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManagerImpl::
getEntryIndex(const Entry & entry, const Item & item) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  EntryIndexMap::const_iterator ifind = m_entry_index.find(InternalEntryIndex(dynamic_cast<MyEntryImpl*>(entry.internal()),item));
  if (ifind != m_entry_index.end())
    {
      return ifind->second;
    }
  else
    {
      throw FatalErrorException(A_FUNCINFO,"Cannot get undefined entry index");
    }
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManagerImpl::
getEquationIndex(const Equation & equation, const Item & item) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  EquationIndexMap::const_iterator ifind = m_equation_index.find(InternalEquationIndex(dynamic_cast<MyEquationImpl*>(equation.internal()),item));

  if (ifind != m_equation_index.end())
    {
      return ifind->second;
    }
  else
    {
      throw FatalErrorException(A_FUNCINFO,"Cannot get undefined equation index");
    }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
getEntryIndex(const Entry & entry, const ItemVectorView & items, ArrayView<Integer> indexes) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");
  if (items.size() != indexes.size())
    throw FatalErrorException(A_FUNCINFO,"Inconsistent sizes");

  ENUMERATE_ITEM(iitem,items) {
    EntryIndexMap::const_iterator ifind = m_entry_index.find(InternalEntryIndex(dynamic_cast<MyEntryImpl*>(entry.internal()),*iitem));
    if (ifind != m_entry_index.end())
      {
        indexes[iitem.index()] = ifind->second;
      }
    else
      {
        throw FatalErrorException(A_FUNCINFO,"Cannot get undefined entry index");
      }
  }
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
getEquationIndex(const Equation & equation, const ItemVectorView & items, ArrayView<Integer> indexes) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");
  if (items.size() != indexes.size())
    throw FatalErrorException(A_FUNCINFO,"Inconsistent sizes");

  ENUMERATE_ITEM(iitem,items) {
    EquationIndexMap::const_iterator ifind = m_equation_index.find(InternalEquationIndex(dynamic_cast<MyEquationImpl*>(equation.internal()),*iitem));
    if (ifind != m_equation_index.end())
      {
        indexes[iitem.index()] = ifind->second;
      }
    else
      {
        throw FatalErrorException(A_FUNCINFO,"Cannot get undefined equation index");
      }
  }
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManagerImpl::
getEntryIndex(const EntryIndex & index) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  if (index<m_local_entry_count-m_entry_reindex.size() or index>=m_local_entry_count)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent entry index requested");

  // Plus compliqué car le nombre d'entrée initialement non locale n'a pas été conservé
  return m_entry_reindex[index+m_entry_reindex.size()-m_local_entry_count];
}

/*---------------------------------------------------------------------------*/

Integer
BasicIndexManagerImpl::
getEquationIndex(const EquationIndex & index) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  if (index<0 or index>=m_local_equation_count)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent equation index requested");

  return m_equation_reindex[index];
}

/*---------------------------------------------------------------------------*/

IntegerArray
BasicIndexManagerImpl::
getEntryIndexes(const Entry & entry) const
{
  MyEntryImpl * en = dynamic_cast<MyEntryImpl*>(entry.internal());
  ARCANE_ASSERT((en != NULL),("Unexpected null equation"));
  const IItemFamily * family = NULL;
  if (dynamic_cast<MyVariableEntryImpl*>(en)) {
    family = dynamic_cast<MyVariableEntryImpl*>(en)->getItemFamily();
  } else if (dynamic_cast<MyAbstractEntryImpl*>(en)) {
    family = dynamic_cast<MyAbstractEntryImpl*>(en)->getItemFamily();
  } else {
    throw FatalErrorException("Unknown Entry implementation");
  }
  IntegerArray allIds(family->maxLocalId(),-1);
  for(EntryIndexMap::const_iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
    if (i->first.first == en)
      allIds[i->first.second.localId()] = i->second;
  return allIds;
}

/*---------------------------------------------------------------------------*/

Array2<Integer>
BasicIndexManagerImpl::
getVecEntryIndexes(ConstArrayView<Entry> entries) const
{
  Integer max_family_size = 0;
  for(Integer entry=0;entry<entries.size();++entry)
  {
    MyEntryImpl * en = dynamic_cast<MyEntryImpl*>(entries[entry].internal());
    ARCANE_ASSERT((en != NULL),("Unexpected null equation"));
    const IItemFamily * family = NULL;
    if (dynamic_cast<MyVariableEntryImpl*>(en)) {
      family = dynamic_cast<MyVariableEntryImpl*>(en)->getItemFamily();
    } else if (dynamic_cast<MyAbstractEntryImpl*>(en)) {
      family = dynamic_cast<MyAbstractEntryImpl*>(en)->getItemFamily();
    } else {
      throw FatalErrorException("Unknown Entry implementation");
    }
    max_family_size = math::max(max_family_size,family->maxLocalId());
  }

  Array2<Integer> allIds(max_family_size,entries.size());
  allIds.fill(-1);

  for(Integer entry=0;entry<entries.size();++entry)
  {
    MyEntryImpl * en = dynamic_cast<MyEntryImpl*>(entries[entry].internal());
    ARCANE_ASSERT((en != NULL),("Unexpected null equation"));
    const IItemFamily * family = NULL;
    if (dynamic_cast<MyVariableEntryImpl*>(en)) {
      family = dynamic_cast<MyVariableEntryImpl*>(en)->getItemFamily();
    } else if (dynamic_cast<MyAbstractEntryImpl*>(en)) {
      family = dynamic_cast<MyAbstractEntryImpl*>(en)->getItemFamily();
    } else {
      throw FatalErrorException("Unknown Entry implementation");
    }
    for(EntryIndexMap::const_iterator i = m_entry_index.begin(); i != m_entry_index.end(); ++i)
      if (i->first.first == en)
        allIds[i->first.second.localId()][entry] = i->second;
  }
  return allIds;
}

/*---------------------------------------------------------------------------*/

IntegerArray
BasicIndexManagerImpl::
getEquationIndexes(const Equation & equation) const
{
  MyEquationImpl * eq = dynamic_cast<MyEquationImpl*>(equation.internal());
  ARCANE_ASSERT((eq != NULL),("Unexpected null equation"));
  const IItemFamily * family = eq->getItemFamily();
  IntegerArray allIds(family->maxLocalId(),-1);
  for(EquationIndexMap::const_iterator i = m_equation_index.begin(); i != m_equation_index.end(); ++i)
    if (i->first.first == eq)
      allIds[i->first.second.localId()] = i->second;
  return allIds;
}

/*---------------------------------------------------------------------------*/

Array2<Integer>
BasicIndexManagerImpl::
getVecEquationIndexes(ConstArrayView<Equation> equations) const
{
  Integer max_family_size = 0;
  for(Integer ieq=0;ieq<equations.size();++ieq)
  {
    MyEquationImpl * eq = dynamic_cast<MyEquationImpl*>(equations[ieq].internal());
    ARCANE_ASSERT((eq != NULL),("Unexpected null equation"));
    const IItemFamily * family = eq->getItemFamily();
    max_family_size = math::max(max_family_size,family->maxLocalId());
  }

  Array2<Integer> allIds(max_family_size,equations.size()); 
  allIds.fill(-1);

  for(Integer ieq=0;ieq<equations.size();++ieq)
  {
    MyEquationImpl * eq = dynamic_cast<MyEquationImpl*>(equations[ieq].internal());
    ARCANE_ASSERT((eq != NULL),("Unexpected null equation"));

    for(EquationIndexMap::const_iterator i = m_equation_index.begin(); i != m_equation_index.end(); ++i)
      if (i->first.first == eq)
        allIds[i->first.second.localId()][ieq] = i->second;
  }

  return allIds;
}

/*---------------------------------------------------------------------------*/

bool
BasicIndexManagerImpl::
hasCompatibleIndexOrder(const Entry & entry, const Equation & equation) const
{
  MyEntryImpl * en = dynamic_cast<MyEntryImpl*>(entry.internal());
  MyEquationImpl * eq = dynamic_cast<MyEquationImpl*>(equation.internal());
  ARCANE_ASSERT((en != NULL),("Unexpected null equation"));
  ARCANE_ASSERT((eq != NULL),("Unexpected null equation"));
  return (en == eq->getInternalEntry());
}

/*---------------------------------------------------------------------------*/

void
BasicIndexManagerImpl::
stats(Integer & totalSize,
      Integer & minLocalIndex,
      Integer & localSize) const
{
  if (m_state != Prepared)
    throw FatalErrorException(A_FUNCINFO,"Inconsistent state");

  // Vérif pour système bien carré
  ARCANE_ASSERT((m_global_entry_count == m_global_equation_count &&
                 m_global_entry_offset == m_global_equation_offset &&
                 m_local_entry_count == m_local_equation_count),
                ("Unexpected non square system: [%d,%d,%d] vs [%d,%d,%d]",
                 m_global_entry_count,m_global_entry_offset,m_local_entry_count,
                 m_global_equation_count,m_global_equation_offset,m_local_equation_count)
                );

  totalSize = m_global_equation_count;
  minLocalIndex = m_global_equation_offset;
  localSize = m_local_equation_count;
}

/*---------------------------------------------------------------------------*/

IIndexManager::EntryEnumerator
BasicIndexManagerImpl::
enumerateEntry()
{
  return EntryEnumerator(new MyEntryEnumeratorImpl(m_entry_set));
}

/*---------------------------------------------------------------------------*/

bool
BasicIndexManagerImpl::EntryIndexComparator::
operator()(const BasicIndexManagerImpl::InternalEntryIndex & a,
           const BasicIndexManagerImpl::InternalEntryIndex & b) const
{
  const Item & aItem = a.second;
  const Item & bItem = b.second;
  if (aItem.kind() != bItem.kind())
    return aItem.kind() < bItem.kind();
  else if (aItem.uniqueId() != bItem.uniqueId())
    return aItem.uniqueId() < bItem.uniqueId();
  else
    return a.first->getCreationIndex() < b.first->getCreationIndex();
}

/*---------------------------------------------------------------------------*/

bool
BasicIndexManagerImpl::EquationIndexComparator::
operator()(const BasicIndexManagerImpl::InternalEquationIndex & a,
           const BasicIndexManagerImpl::InternalEquationIndex & b) const
{
  const Item & aItem = a.second;
  const Item & bItem = b.second;
  if (aItem.kind() != bItem.kind())
    return aItem.kind() < bItem.kind();
  else if (aItem.uniqueId() != bItem.uniqueId())
    return aItem.uniqueId() < bItem.uniqueId();
  else
    return a.first->getInternalEntry()->getCreationIndex() < b.first->getInternalEntry()->getCreationIndex();
}

/*---------------------------------------------------------------------------*/
