// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MessagePassingGlobal.h                                      (C) 2000-2020 */
/*                                                                           */
/* Définitions globales de la composante 'MessagePassing' de 'Arccore'.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGGLOBAL_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

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

namespace Arccore::MessagePassing
{
//! Numéro correspondant à un rang nul
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

class Request;
class MessageId;
class PointToPointMessageInfo;
class IStat;
class IMessagePassingMng;
class MessagePassingMng;
class IDispatchers;
class Dispatchers;
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
  WaitAll, //! Attend que tous les messages de la liste soient traités
  WaitSome,//! Attend que au moins un message de la liste soit traité
  WaitSomeNonBlocking //! Traite uniquement les messages qui peuvent l'être sans attendre.
};
/*!
 * \brief Type indiquant si un message est bloquant ou non.
 */
enum eBlockingType
{
  Blocking = 0,
  NonBlocking
};
} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

