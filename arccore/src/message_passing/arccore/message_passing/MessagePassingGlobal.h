// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingGlobal.h                                      (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'MessagePassing' de 'Arccore'.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGGLOBAL_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_message_passing)
#define ARCCORE_MESSAGEPASSING_EXPORT ARCCORE_EXPORT
#define ARCCORE_MESSAGEPASSING_EXTERN_TPL
#else
#define ARCCORE_MESSAGEPASSING_EXPORT ARCCORE_IMPORT
#define ARCCORE_MESSAGEPASSING_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISerializer;
class ITimeMetricCollector;
}
namespace Arccore
{
using Arcane::ISerializer;
using Arcane::ITimeMetricCollector;
}

namespace Arcane::MessagePassing
{
/*!
 * \brief Numéro correspondant à un rang nul.
 *
 * La signification du rang nul dépend de la situation.
 *
 * \sa MessageRank.
 */
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

//! Numéro correspondant à un rang nul
static const Int32 A_NULL_TAG_VALUE = static_cast<Int32>(-1);

//! Numéro correspondant à MPI_ANY_SOURCE
static const Int32 A_ANY_SOURCE_RANK = static_cast<Int32>(-2);

//! Numéro correspondant à MPI_PROC_NULL
static const Int32 A_PROC_NULL_RANK = static_cast<Int32>(-3);

class Communicator;
class SubRequestCompletionInfo;
class IRequestCreator;
class IRequestList;
class ISerializeMessage;
class ISerializeMessageList;
class ISerializeDispatcher;
class Request;
class MessageId;
class MessageTag;
class MessageRank;
class MessageSourceInfo;
class PointToPointMessageInfo;
class IStat;
class Stat;
class OneStat;
class StatData;
class IMessagePassingMng;
class MessagePassingMng;
class IDispatchers;
class Dispatchers;
class IProfiler;
class ISubRequest;
class IControlDispatcher;
template<typename DataType> class ITypeDispatcher;
class GatherMessageInfoBase;
template<typename DataType> class GatherMessageInfo;
class PointToPointSerializerMng;
class ISerializeMessage;
class ISerializeMessageList;
class IMachineMemoryWindowBase;

/*!
 * \brief Types des réductions supportées.
 */
enum eReduceType
{
  ReduceMin, //!< Minimum des valeurs
  ReduceMax, //!< Maximum des valeurs
  ReduceSum  //!< Somme des valeurs
};

/*!
 * \brief Type d'attente.
 */
enum eWaitType
{
  WaitAll = 0, //! Attend que tous les messages de la liste soient traités
  WaitSome = 1, //! Attend que au moins un message de la liste soit traité
  TestSome = 2, //! Traite uniquement les messages qui peuvent l'être sans attendre.
  //! \deprecated Utiliser TestSome à la place
  WaitSomeNonBlocking = 2
};
/*!
 * \brief Type indiquant si un message est bloquant ou non.
 */
enum eBlockingType
{
  Blocking = 0,
  NonBlocking
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type de message point à point.
 */
enum ePointToPointMessageType
{
  MsgSend = 0,
  MsgReceive
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::internal
{
class BasicSerializeMessage;
class SerializeMessageList;
}

namespace Arccore::MessagePassing::internal
{
using Arcane::MessagePassing::internal::BasicSerializeMessage;
using Arcane::MessagePassing::internal::SerializeMessageList;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::MessagePassing::IMessagePassingMng)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Arcane::MessagePassing::IControlDispatcher;
using Arcane::MessagePassing::IMessagePassingMng;
using Arcane::MessagePassing::ISerializeMessage;
using Arcane::MessagePassing::ISerializeMessageList;
using Arcane::MessagePassing::ITypeDispatcher;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
using Arcane::MessagePassing::eReduceType;
using Arcane::MessagePassing::ReduceMax;
using Arcane::MessagePassing::ReduceMin;
using Arcane::MessagePassing::ReduceSum;

using Arcane::MessagePassing::eWaitType;
using Arcane::MessagePassing::TestSome;
using Arcane::MessagePassing::WaitAll;
using Arcane::MessagePassing::WaitSome;
using Arcane::MessagePassing::WaitSomeNonBlocking;

using Arcane::MessagePassing::Blocking;
using Arcane::MessagePassing::eBlockingType;
using Arcane::MessagePassing::NonBlocking;

using Arcane::MessagePassing::ePointToPointMessageType;
using Arcane::MessagePassing::MsgReceive;
using Arcane::MessagePassing::MsgSend;

using Arcane::MessagePassing::IRequestCreator;
using Arcane::MessagePassing::IRequestList;
using Arcane::MessagePassing::ISubRequest;
using Arcane::MessagePassing::MessageId;
using Arcane::MessagePassing::MessageRank;
using Arcane::MessagePassing::MessageSourceInfo;
using Arcane::MessagePassing::MessageTag;
using Arcane::MessagePassing::PointToPointMessageInfo;
using Arcane::MessagePassing::Request;
using Arcane::MessagePassing::MessagePassingMng;
using Arcane::MessagePassing::SubRequestCompletionInfo;

using Arcane::MessagePassing::Communicator;
using Arcane::MessagePassing::IProfiler;
using Arcane::MessagePassing::IStat;
using Arcane::MessagePassing::Stat;
using Arcane::MessagePassing::ISerializeDispatcher;
using Arcane::MessagePassing::IDispatchers;
using Arcane::MessagePassing::Dispatchers;

using Arcane::MessagePassing::A_NULL_RANK;
using Arcane::MessagePassing::A_NULL_TAG_VALUE;
using Arcane::MessagePassing::A_ANY_SOURCE_RANK;
using Arcane::MessagePassing::A_PROC_NULL_RANK;

using Arcane::MessagePassing::GatherMessageInfoBase;
using Arcane::MessagePassing::GatherMessageInfo;

using Arcane::MessagePassing::OneStat;
using Arcane::MessagePassing::StatData;
using Arcane::MessagePassing::IControlDispatcher;
using Arcane::MessagePassing::IMessagePassingMng;
using Arcane::MessagePassing::PointToPointSerializerMng;
using Arcane::MessagePassing::ISerializeMessage;
using Arcane::MessagePassing::ISerializeMessageList;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète et utiliser Arcane::MessagePassing à la place
namespace Arcane::Parallel
{
using Arcane::MessagePassing::eReduceType;
using Arcane::MessagePassing::ReduceMax;
using Arcane::MessagePassing::ReduceMin;
using Arcane::MessagePassing::ReduceSum;

using Arcane::MessagePassing::eWaitType;
using Arcane::MessagePassing::TestSome;
using Arcane::MessagePassing::WaitAll;
using Arcane::MessagePassing::WaitSome;
using Arcane::MessagePassing::WaitSomeNonBlocking;

using Arcane::MessagePassing::Blocking;
using Arcane::MessagePassing::eBlockingType;
using Arcane::MessagePassing::NonBlocking;

using Arcane::MessagePassing::ePointToPointMessageType;
using Arcane::MessagePassing::MsgReceive;
using Arcane::MessagePassing::MsgSend;

using Arcane::MessagePassing::IRequestCreator;
using Arcane::MessagePassing::IRequestList;
using Arcane::MessagePassing::ISubRequest;
using Arcane::MessagePassing::MessageId;
using Arcane::MessagePassing::MessageRank;
using Arcane::MessagePassing::MessageSourceInfo;
using Arcane::MessagePassing::MessageTag;
using Arcane::MessagePassing::PointToPointMessageInfo;
using Arcane::MessagePassing::Request;

using Arcane::MessagePassing::Communicator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

