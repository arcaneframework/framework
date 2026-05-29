// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMessageQueue.h                                        (C) 2000-2024 */
/*                                                                           */
/* Message file for a hybrid MPI/Shared Memory implementation.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_HYBRIDMESSAGEQUEUE_H
#define ARCANE_PARALLEL_THREAD_HYBRIDMESSAGEQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/parallel/thread/ISharedMemoryMessageQueue.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/Parallel.h"
#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class MpiParallelMng;
}

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Correspondence information between the different ranks
 * of a communicator.
 */
class FullRankInfo
{
 public:

  static FullRankInfo compute(MP::MessageRank rank,Int32 local_nb_rank)
  {
    Int32 r = rank.value();
    FullRankInfo fri;
    fri.m_local_rank.setValue(r % local_nb_rank);
    fri.m_global_rank.setValue(r);
    fri.m_mpi_rank.setValue(r / local_nb_rank);
    return fri;
  }
  friend std::ostream& operator<<(std::ostream& o,const FullRankInfo& fri);

 public:

  //! Local rank within the threads
  MP::MessageRank localRank() const { return m_local_rank; }
  Int32 localRankValue() const { return m_local_rank.value(); }
  //! Global rank within the communicator
  MP::MessageRank globalRank() const { return m_global_rank; }
  Int32 globalRankValue() const { return m_global_rank.value(); }
  //! Associated MPI rank
  MP::MessageRank mpiRank() const { return m_mpi_rank; }
  Int32 mpiRankValue() const { return m_mpi_rank.value(); }

 private:

  //! Local rank within the threads
  MP::MessageRank m_local_rank;
  //! Global rank within the communicator
  MP::MessageRank m_global_rank;
  //! Associated MPI rank
  MP::MessageRank m_mpi_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsulates source/destination information.
 */
class SourceDestinationFullRankInfo
{
 public:

  SourceDestinationFullRankInfo(FullRankInfo s,FullRankInfo d)
  : m_source(s), m_destination(d){}

 public:

  FullRankInfo source() const { return m_source; }
  FullRankInfo destination() const { return m_destination; }
  bool isSameMpiRank() const
  {
    return m_source.mpiRank()==m_destination.mpiRank();
  }

 private:

  FullRankInfo m_source;
  FullRankInfo m_destination;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class to calculate a tag containing sender and receiver information
 * from a user tag.
 *
 * For tag calculation, the tag must include information about the original
 * and destination local ranks in the MPI tag, otherwise it is not possible
 * to know who to receive/send to. For this calculation, we use the higher
 * bits of MAX_USER_TAG_BIT to preserve both the source and destination ranks.
 * Since there is a limit to possible MPI tag values, the value
 * of MAX_USER_TAG_BIT must be relatively small. It must be that
 * ((1<<MAX_USER_TAG_BIT) * n * n) is less than the maximum allowed MPI tag
 * value (where n is the number of local threads).
 */
class RankTagBuilder
{
 public:
  //! We allow 2^14 tags, i.e., 16384.
  // NOTE: theoretically, this value can be calculated dynamically by taking
  // into account the max valid value for MPI (via MPI_Comm_get_attribute(MPI_TAG_UB))
  // and the maximum number of local threads. However, this would make
  // the code too dependent on the implementation.
  // Note that in the MPI standard, it is possible to have only
  // 2^15 (32767) values for tags.
  static constexpr Int32 MAX_USER_TAG_BIT = 14;
  static constexpr Int32 MAX_USER_TAG =  1 << MAX_USER_TAG_BIT;
 public:
  //! Constructs an instance for \a local_nb_rank.
  RankTagBuilder(Int32 nb_rank) : m_nb_rank(nb_rank) {}
  Int32 nbLocalRank() const { return m_nb_rank; }
  FullRankInfo rank(MessageRank user_rank) const
  {
    return FullRankInfo::compute(user_rank,m_nb_rank);
  }
  SourceDestinationFullRankInfo rank(MessageRank rank1,MessageRank rank2) const
  {
    auto x1 = FullRankInfo::compute(rank1,m_nb_rank);
    auto x2 = FullRankInfo::compute(rank2,m_nb_rank);
    return SourceDestinationFullRankInfo(x1,x2);
  }
  MessageTag tagForSend(MessageTag user_tag,FullRankInfo orig,FullRankInfo dest) const
  {
    return _tag(user_tag,dest.localRank(),orig.localRank());
  }
  MessageTag tagForSend(MessageTag user_tag,SourceDestinationFullRankInfo fri) const
  {
    return tagForSend(user_tag,fri.source(),fri.destination());
  }
  MessageTag tagForReceive(MessageTag user_tag,FullRankInfo orig,FullRankInfo dest) const
  {
    return _tag(user_tag,orig.localRank(),dest.localRank());
  }
  MessageTag tagForReceive(MessageTag user_tag,MessageRank orig_local,MessageRank dest_local) const
  {
    return _tag(user_tag,orig_local,dest_local);
  }
  MessageTag tagForReceive(MessageTag user_tag,SourceDestinationFullRankInfo fri) const
  {
    return tagForReceive(user_tag,fri.source(),fri.destination());
  }
  //! Retrieves the rank from the tag. This is the inverse operation of _tag()
  Int32 getReceiveRankFromTag(MessageTag internal_tag) const
  {
    Int32 t = internal_tag.value() >> MAX_USER_TAG_BIT;
    return t % m_nb_rank;
  }
 private:
  MessageTag _tag(MessageTag user_tag,MessageRank orig_local,MessageRank dest_local) const
  {
    Int64 utag = user_tag.value();
    if (utag>MAX_USER_TAG)
      ARCANE_FATAL("User tag is too big v={0} max={1}",utag,MAX_USER_TAG);
    Int32 d = dest_local.value();
    Int32 o = orig_local.value();
    Int64 new_tag = (o*m_nb_rank + d) << MAX_USER_TAG_BIT;
    new_tag += utag;
    return MessageTag(CheckedConvert::toInt32(new_tag));
  }
 private:
  Int32 m_nb_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a message queue with threads.
 * \warning All methods of this class must be thread-safe.
 */
class HybridMessageQueue
: public TraceAccessor
{
 public:

  HybridMessageQueue(ISharedMemoryMessageQueue* thread_queue,MpiParallelMng* mpi_pm,
                     Int32 local_nb_rank);

 public:

  void waitAll(ArrayView<Request> requests);
  void waitSome(Int32 rank,ArrayView<Request> requests,
                ArrayView<bool> requests_done,bool is_non_blocking);
 public:

  Request addReceive(const PointToPointMessageInfo& message,ReceiveBufferInfo buf);
  Request addSend(const PointToPointMessageInfo& message,SendBufferInfo buf);
  MessageId probe(const MP::PointToPointMessageInfo& message);
  MP::MessageSourceInfo legacyProbe(const MP::PointToPointMessageInfo& message);
  const RankTagBuilder& rankTagBuilder() const { return m_rank_tag_builder; }

 private:

  ISharedMemoryMessageQueue* m_thread_queue;
  MpiParallelMng* m_mpi_parallel_mng;
  MpiAdapter* m_mpi_adapter;
  Int32 m_local_nb_rank;
  RankTagBuilder m_rank_tag_builder;
  Int32 m_debug_level = 0;
  bool m_is_allow_null_rank_for_any_source = true;

 private:

  Request _addReceiveRankTag(const PointToPointMessageInfo& message,ReceiveBufferInfo buf_info);
  Request _addReceiveMessageId(const PointToPointMessageInfo& message,ReceiveBufferInfo buf_info);
  void _checkValidRank(MessageRank rank);
  void _checkValidSource(const PointToPointMessageInfo& message);
  SourceDestinationFullRankInfo _getFullRankInfo(const PointToPointMessageInfo& message)
  {
    return m_rank_tag_builder.rank(message.emiterRank(),message.destinationRank());
  }
  PointToPointMessageInfo
  _buildSharedMemoryMessage(const PointToPointMessageInfo& message,
                            const SourceDestinationFullRankInfo& fri);
  PointToPointMessageInfo
  _buildMPIMessage(const PointToPointMessageInfo& message,
                   const SourceDestinationFullRankInfo& fri);
  Integer _testOrWaitSome(Int32 rank,ArrayView<Request> requests,
                          ArrayView<bool> requests_done);
  MessageId _probe(const MP::PointToPointMessageInfo& message, bool use_message_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
