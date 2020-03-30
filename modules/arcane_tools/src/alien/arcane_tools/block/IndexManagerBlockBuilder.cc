#include "alien/arcane_tools/block/IndexManagerBlockBuilder.h"

#ifdef ALIEN_USE_ARCANE
#include <arcane/IParallelMng.h>
#include <arcane/SerializeMessage.h>
#include <arcane/ISerializeMessageList.h>
#endif // ALIEN_USE_ARCANE

#include <map>
#include <list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  IndexManagerBlockBuilder::IndexManagerBlockBuilder(
      IIndexManager& index_mng, const VectorDistribution& distribution)
  : m_index_mng(&index_mng)
  {
    if (not m_index_mng->isPrepared())
      m_index_mng->prepare();

    Arccore::Integer global_size, local_size;
    m_index_mng->stats(global_size, m_offset, local_size);

    if (global_size != distribution.globalSize() || local_size != distribution.localSize()
        || m_offset != distribution.offset()) {
      throw Arccore::FatalErrorException(
          A_FUNCINFO, "IndexManager doesn't match Distribution");
    }

    m_sizes.resize(local_size);
  }

  /*---------------------------------------------------------------------------*/

  IndexManagerBlockBuilder::IndexManagerBlockBuilder(
      IIndexManager& index_mng, const MatrixDistribution& distribution)
  : m_index_mng(&index_mng)
  {
    if (not m_index_mng->isPrepared())
      m_index_mng->prepare();

    Arccore::Integer global_size, local_size;
    m_index_mng->stats(global_size, m_offset, local_size);

    if (global_size != distribution.globalRowSize()
        || local_size != distribution.localRowSize()
        || m_offset != distribution.rowOffset()) {
      throw Arccore::FatalErrorException(
          A_FUNCINFO, "IndexManager doesn't match Distribution");
    }

    m_sizes.resize(local_size);
  }

/*---------------------------------------------------------------------------*/

#ifdef ALIEN_USE_ARCANE
  struct IndexManagerBlockBuilder::EntrySendRequest
  {
    EntrySendRequest()
    : comm(NULL)
    , count(0)
    {
    }

    ~EntrySendRequest()
    {
      // Valide même si comm vaut NULL
      delete comm;
    }
    Arcane::SerializeMessage* comm;
    Arccore::Integer count;
  };
#endif

/*---------------------------------------------------------------------------*/

#ifdef ALIEN_USE_ARCANE
  struct IndexManagerBlockBuilder::EntryRecvRequest
  {
    EntryRecvRequest()
    : comm(nullptr)
    {
    }

    ~EntryRecvRequest()
    {
      // Valide même si comm vaut NULL
      delete comm;
    }

    Arcane::SerializeMessage* comm;
    Arccore::UniqueArray<Arccore::Integer> ids;
  };
#endif

  /*---------------------------------------------------------------------------*/

  void IndexManagerBlockBuilder::compute() const
  {
#ifdef ALIEN_USE_ARCANE

    Arccore::Integer m_local_size;
    Arccore::Integer m_max_size;
    Arccore::Integer m_offset;

    using ValuePerBlock = VMap<Arccore::Integer, Arccore::Integer>;

    ValuePerBlock m_ghost_sizes;
    ValuePerBlock m_ghost_offsets;

    Arccore::UniqueArray<Arccore::Integer> m_local_sizes;
    Arccore::UniqueArray<Arccore::Integer> m_local_offsets;

    auto* parallel_mng = m_index_mng->parallelMng();

    const Arccore::Integer offset = m_index_mng->minLocalIndex();

    const Arccore::Integer size = m_sizes.size();

    m_local_sizes.resize(size);
    m_local_offsets.resize(size + 1);

    Arccore::Integer sum = 0, max_size = 0;
    for (Arccore::Integer i = 0; i < size; ++i) {
      const Arccore::Integer block_size = m_sizes[i];
      max_size = (max_size > block_size) ? max_size : block_size;
      const Arccore::Integer index = i + offset;
      m_ghost_sizes[index] = m_local_sizes[i] = block_size;
      ghost_sizes[index] = block_size;
      m_ghost_offsets[index] = m_local_offsets[i] = sum;
      sum += block_size;
    }
    m_local_offsets[size] = sum;
    m_max_size = max_size;
    m_local_size = sum;

    const bool is_parallel = (parallel_mng != NULL) && (parallel_mng->commSize() > 1);

    m_offset = 0;
    if (is_parallel) {
      Arccore::UniqueArray<Arccore::Integer> local_sizes(parallel_mng->commSize());
      parallel_mng->allGather(
          Arccore::ArrayView<Arccore::Integer>(1, &m_local_size), local_sizes);
      for (Arccore::Integer i = 0; i < parallel_mng->commRank(); ++i)
        m_offset += local_sizes[i];
    }

    // TODO
    // if(not is_parallel) return;

    const Arccore::Integer rank = parallel_mng->commRank();

    using SendRequestByEntry = std::map<Arccore::String, EntrySendRequest>;
    using SendRequests = std::map<Arccore::Integer, SendRequestByEntry>;
    SendRequests sendRequests;

    using Owners = Arccore::UniqueArray<Arccore::Integer>;
    using OwnersByEntry = std::map<Arccore::String, Owners>;
    OwnersByEntry owners_by_entry;

    for (auto e = m_index_mng->enumerateEntry(); e.hasNext(); ++e) {
      // ConstArrayView<Integer> all_indexes = e->getAllIndexes();
      Arccore::ConstArrayView<Arccore::Integer> all_lids = e->getAllLocalIds();
      Owners owners = e->getFamily().owners(all_lids);
      const Arccore::String name = e->getName();
      owners_by_entry[name] = owners;
      for (Arccore::Integer i = 0; i < owners.size(); ++i) {
        if (owners[i] != rank)
          sendRequests[owners[i]][name].count++;
      }
    }

    auto* messageList = parallel_mng->createSerializeMessageList();

    Arccore::UniqueArray<Arccore::Integer> sendToDomains(2 * parallel_mng->commSize(), 0);

    for (auto i = sendRequests.begin(); i != sendRequests.end(); ++i) {
      const Arccore::Integer destDomainId = i->first;
      SendRequestByEntry& requests = i->second;
      for (auto j = requests.begin(); j != requests.end(); ++j) {
        EntrySendRequest& request = j->second;
        const Arccore::String& nameString = j->first;
        sendToDomains[2 * destDomainId + 0] += 1;
        sendToDomains[2 * destDomainId + 1] += request.count;
        request.comm = new Arcane::SerializeMessage(
            parallel_mng->commRank(), destDomainId, Arcane::ISerializeMessage::MT_Send);
        messageList->addMessage(request.comm);
        auto& sbuf = request.comm->buffer();
        sbuf.setMode(Arcane::ISerializer::ModeReserve);
        sbuf.reserve(nameString); // Chaine de caractère du nom de l'entrée
        sbuf.reserveInteger(1); // Nb d'item
        sbuf.reserve(Arcane::DT_Int32, request.count); // Les indices demandés
        sbuf.allocateBuffer();
        sbuf.setMode(Arcane::ISerializer::ModePut);
        sbuf.put(nameString);
        sbuf.put(request.count);
      }
    }

    for (auto e = m_index_mng->enumerateEntry(); e.hasNext(); ++e) {
      const Arccore::String name = e->getName();
      Owners& owners = owners_by_entry[name];
      Arccore::ConstArrayView<Arccore::Integer> all_indexes = e->getAllIndexes();
      for (Arccore::Integer i = 0; i < owners.size(); ++i) {
        if (owners[i] != rank)
          sendRequests[owners[i]][name].comm->buffer().put(all_indexes[i]);
      }
    }

    Arccore::UniqueArray<Arccore::Integer> recvFromDomains(2 * parallel_mng->commSize());
    parallel_mng->allToAll(sendToDomains, recvFromDomains, 2);

    using RecvRequests = std::list<EntryRecvRequest>;
    RecvRequests recvRequests;

    for (Arccore::Integer isd = 0, nsd = parallel_mng->commSize(); isd < nsd; ++isd) {
      Arccore::Integer recvCount = recvFromDomains[2 * isd + 0];
      while (recvCount-- > 0) {
        auto* recvMsg = new Arcane::SerializeMessage(
            parallel_mng->commRank(), isd, Arcane::ISerializeMessage::MT_Recv);
        recvRequests.push_back(EntryRecvRequest());
        EntryRecvRequest& recvRequest = recvRequests.back();
        recvRequest.comm = recvMsg;
        messageList->addMessage(recvMsg);
      }
    }

    messageList->processPendingMessages();
    messageList->waitMessages(Arccore::MessagePassing::WaitAll);
    delete messageList;
    messageList = NULL;

    messageList = parallel_mng->createSerializeMessageList();

    for (auto i = recvRequests.begin(); i != recvRequests.end(); ++i) {
      EntryRecvRequest& recvRequest = *i;
      Arccore::String nameString;
      Arccore::Integer uidCount;

      {
        auto& sbuf = recvRequest.comm->buffer();
        sbuf.setMode(Arcane::ISerializer::ModeGet);

        sbuf.get(nameString);
        uidCount = sbuf.getInteger();
        recvRequest.ids.resize(uidCount);
        sbuf.get(recvRequest.ids);
        ALIEN_ASSERT((uidCount == recvRequest.ids.size()), ("Inconsistency detected"));
      }

      {
        const Arccore::Integer dest =
            recvRequest.comm->destRank(); // Attention à l'ordre bizarre
        const Arccore::Integer orig =
            recvRequest.comm->origRank(); //       de SerializeMessage
        delete recvRequest.comm;
        recvRequest.comm =
            new Arcane::SerializeMessage(orig, dest, Arcane::ISerializeMessage::MT_Send);
        messageList->addMessage(recvRequest.comm);

        auto& sbuf = recvRequest.comm->buffer();
        sbuf.setMode(Arcane::ISerializer::ModeReserve);
        sbuf.reserve(nameString); // Chaine de caractère du nom de l'entrée
        sbuf.reserveInteger(1); // Nb d'item
        sbuf.reserveInteger(uidCount); // Les tailles
        sbuf.allocateBuffer();
        sbuf.setMode(Arcane::ISerializer::ModePut);
        sbuf.put(nameString);
        sbuf.put(uidCount);
      }
    }
    {
      for (auto i = recvRequests.begin(); i != recvRequests.end(); ++i) {
        EntryRecvRequest& recvRequest = *i;
        auto& sbuf = recvRequest.comm->buffer();
        Arccore::UniqueArray<Arccore::Integer>& ids = recvRequest.ids;
        for (Arccore::Integer j = 0; j < ids.size(); ++j) {
          sbuf.putInteger(m_sizes[ids[j] - offset]);
        }
      }
    }

    using ReturnedRequests = std::list<Arcane::SerializeMessage*>;
    ReturnedRequests returnedRequests;

    using SubFastReturnMap = std::map<Arccore::Integer, EntrySendRequest*>;
    using FastReturnMap = std::map<Arccore::String, SubFastReturnMap>;
    FastReturnMap fastReturnMap;

    for (auto i = sendRequests.begin(); i != sendRequests.end(); ++i) {
      const Arccore::Integer destDomainId = i->first;
      SendRequestByEntry& requests = i->second;
      for (SendRequestByEntry::iterator j = requests.begin(); j != requests.end(); ++j) {
        EntrySendRequest& request = j->second;
        const Arccore::String nameString = j->first;
        delete request.comm;
        request.comm = NULL;
        auto* msg = new Arcane::SerializeMessage(
            parallel_mng->commRank(), destDomainId, Arcane::ISerializeMessage::MT_Recv);
        returnedRequests.push_back(msg);
        messageList->addMessage(msg);

        fastReturnMap[nameString][destDomainId] = &request;
      }
    }

    messageList->processPendingMessages();
    messageList->waitMessages(Arccore::MessagePassing::WaitAll);
    delete messageList;
    messageList = NULL;

    for (auto i = returnedRequests.begin(); i != returnedRequests.end(); ++i) {
      Arcane::SerializeMessage* message = *i;
      const Arccore::Integer origDomainId = message->destRank();
      auto& sbuf = message->buffer();
      sbuf.setMode(Arcane::ISerializer::ModeGet);
      Arccore::String nameString;
      sbuf.get(nameString);
      ALIEN_ASSERT(
          (fastReturnMap[nameString][origDomainId] != NULL), ("Inconsistency detected"));
      EntrySendRequest& request = *fastReturnMap[nameString][origDomainId];
      request.comm = *i;
#ifdef ARCANE_DEBUG_ASSERT
      const Arccore::Integer idCount = sbuf.getInteger();
      ALIEN_ASSERT((request.count == idCount), ("Inconsistency detected"));
#else
      sbuf.getInteger();
#endif
    }

    for (auto e = m_index_mng->enumerateEntry(); e.hasNext(); ++e) {
      Arccore::ConstArrayView<Arccore::Integer> all_indexes = e->getAllIndexes();
      Owners owners = owners_by_entry[(*e).getName()];
      for (Arccore::Integer i = 0; i < owners.size(); ++i) {
        const Arccore::Integer index = all_indexes[i];
        if (owners[i] != rank) {
          EntrySendRequest& request = sendRequests[owners[i]][e->getName()];
          ALIEN_ASSERT((request.count > 0), ("Unexpected empty request"));
          --request.count;
          const Arccore::Integer block_size = request.comm->buffer().getInteger();
          m_ghost_sizes[index] = block_size;
          ghost_sizes[index] = block_size;
          m_max_size = (m_max_size > block_size) ? m_max_size : block_size;
          m_ghost_offsets[index] = sum;
          sum += block_size;
        }
      }
    }
#else
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Not yet implemented in External Mode");
#endif
    /*
    if(parallel_mng->commRank()==0)
    {
      std::cout << "Blocks in IndexManagerBlockBuilder:\n";
      std::cout << "m_local_scalarized_size: " << m_local_size << "\n";
      std::cout << "m_global_scalarized_size: " << m_global_size  << "\n";
      std::cout << "m_scalarized_offset: " << m_offset << "\n";
      std::cout << "m_max_block_size: " << m_max_size << "\n";
      std::cout << "m_local_sizes.size(): " << m_local_sizes.size() << "\n";
      std::cout << "m_local_offsets.size(): " << m_local_offsets.size() << "\n";
      std::cout << "m_all_sizes.size(): " << m_ghost_sizes.size() << "\n";
      std::cout << "m_all_offsets.size(): " << m_ghost_offsets.size() << "\n";

      std::cout << "m_local_sizes: \n";
      for(Integer i=0;i<m_local_sizes.size();++i)
        std::cout << m_local_sizes[i] << "\n";
      std::cout << "m_local_offsets: \n";
      for(Integer i=0;i<m_local_offsets.size();++i)
        std::cout << m_local_offsets[i] << "\n";
      std::cout << "m_all_sizes: \n";
      for(ValuePerBlock::const_iterator it = m_ghost_sizes.begin(); it !=
    m_ghost_sizes.end(); ++it)
      {
        std::cout << "index: " << it.key() << " value: " << it.value() << "\n";
      }
      std::cout << "m_all_offsets: \n";
      for(ValuePerBlock::const_iterator it = m_ghost_offsets.begin(); it !=
    m_ghost_offsets.end(); ++it)
      {
        std::cout << "index: " << it.key() << " value: " << it.value() << "\n";
      }
    }
    */
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
