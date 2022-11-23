// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingGlobal.h                                      (C) 2000-2022 */
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

namespace Arccore
{
class ISerializer;
class ITimeMetricCollector;
}

namespace Arccore::MessagePassing
{
//! Numéro correspondant à un rang nul
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

//! Numéro correspondant à un rang nul
static const Int32 A_NULL_TAG_VALUE = static_cast<Int32>(-1);

class Communicator;
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
class IMessagePassingMng;
class MessagePassingMng;
class IDispatchers;
class Dispatchers;
class IProfiler;
class IControlDispatcher;
template<class DataType>
class ITypeDispatcher;

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

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::internal
{
class BasicSerializeMessage;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arccore::MessagePassing::IMessagePassingMng)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

