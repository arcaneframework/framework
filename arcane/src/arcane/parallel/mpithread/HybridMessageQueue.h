// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMessageQueue.h                                        (C) 2000-2022 */
/*                                                                           */
/* File de messages pour une implémentation hybride MPI/Mémoire partagée.    */
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
 * \brief Informations de correspondances entre les différents rangs
 * d'un communicateur.
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

  //! Rang local dans les threads
  MP::MessageRank localRank() const { return m_local_rank; }
  Int32 localRankValue() const { return m_local_rank.value(); }
  //! Rang global dans le communicateur
  MP::MessageRank globalRank() const { return m_global_rank; }
  Int32 globalRankValue() const { return m_global_rank.value(); }
  //! Rang MPI du 
  MP::MessageRank mpiRank() const { return m_mpi_rank; }
  Int32 mpiRankValue() const { return m_mpi_rank.value(); }

 private:

  //! Rang local dans les threads
  MP::MessageRank m_local_rank;
  //! Rang global dans le communicateur
  MP::MessageRank m_global_rank;
  //! Rang MPI associé
  MP::MessageRank m_mpi_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule les informations source/destination.
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
 * \brief Classe pour calculer à partir d'un tag utilisateur un
 * tag contenant les informations de l'envoyeur et du réceptionneur.
 *
 * Pour le calcul du tag, il faut intégrer au tag MPI les informations
 * sur le rang local d'origine et de destination sinon il n'est pas possible
 * de savoir à qui recevoir/envoyer. Pour ce calcul, on utilise
 * les bits de poids supérieur à MAX_USER_TAG_BIT pour conserver
 * à la fois le rang de la source et de la destination. Comme il existe
 * une limite aux valeurs possibles pour les tags MPI, la valeur
 * de MAX_USER_TAG_BIT doit être relativement faible. Il faut que
 * ((1<<MAX_USER_TAG_BIT) * n * n) soit inférieur à la valeur
 * maximale des tags MPI autorisés (avec \a n le nombre de threads locaux).
 */
class RankTagBuilder
{
 public:
  //! On autorise 2^14 tags soit 16384.
  // NOTE: en théorie on peut calculer dynamiquement cette valeur en prenant
  // en compte le max valide pour MPI (via MPI_Comm_get_attribute(MPI_TAG_UB))
  // et le nombre de threads locaux maximun. Cependant, cela rendrait
  // le code trop dépendant de l'implémentation.
  // A noter que dans la norme MPI, il est possible de n'avoir que
  // 2^15 (32767) valeurs pour les tag.
  static constexpr Int32 MAX_USER_TAG_BIT = 14;
  static constexpr Int32 MAX_USER_TAG =  1 << MAX_USER_TAG_BIT;
 public:
  //! Construit une instance pour \a nb_rank locaux.
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
  //! Récupère le rang à partir du tag. Il s'agit de l'opération inverse de _tag()
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
 * \brief Interface d'une file de messages avec les threads.
 * \warning Toutes les méthodes de cette classe doivent être thread-safe.
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
