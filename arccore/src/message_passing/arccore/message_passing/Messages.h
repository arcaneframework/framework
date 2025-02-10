// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Messages.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGES_H
#define ARCCORE_MESSAGEPASSING_MESSAGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IDispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Request.h"

#include "arccore/base/RefDeclarations.h"
#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(type) \
  /*! AllGather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf); \
  /*! gather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank); \
  /*! AllGather non bloquant */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf); \
  /*! Gather non bloquant */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingGather(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 rank); \
  /*! AllGatherVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf); \
  /*! GatherVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpGatherVariable(IMessagePassingMng* pm, Span<const type> send_buf, Array<type>& recv_buf, Int32 rank); \
  /*! Generic Gather */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpGather(IMessagePassingMng* pm, GatherMessageInfo<type>& gather_info); \
  /*! ScatterVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpScatterVariable(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 root); \
  /*! AllReduce */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT type \
  mpAllReduce(IMessagePassingMng* pm, eReduceType rt, type v); \
  /*! AllReduce */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<type> buf); \
  /*! AllReduce non bloquant */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllReduce(IMessagePassingMng* pm, eReduceType rt, Span<const type> send_buf, Span<type> recv_buf); \
  /*! Broadcast */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank); \
  /*! Broadcast non bloquant */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingBroadcast(IMessagePassingMng* pm, Span<type> send_buf, Int32 rank); \
  /*! Send */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank); \
  /*! Receive */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank); \
  /*! Send */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpSend(IMessagePassingMng* pm, Span<const type> values, Int32 rank, bool is_blocked); \
  /*! Send */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpSend(IMessagePassingMng* pm, Span<const type> values, const PointToPointMessageInfo& message); \
  /*! Receive */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpReceive(IMessagePassingMng* pm, Span<type> values, Int32 rank, bool is_blocked); \
  /*! Receive */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpReceive(IMessagePassingMng* pm, Span<type> values, const PointToPointMessageInfo& message); \
  /*! AllToAll */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count); \
  /*! AllToAll non bloquant */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllToAll(IMessagePassingMng* pm, Span<const type> send_buf, Span<type> recv_buf, Int32 count); \
  /*! AllToAllVariable */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void \
  mpAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                     ConstArrayView<Int32> send_index, Span<type> recv_buf, \
                     ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index); \
  /*! AllToAllVariable non bloquant */ \
  extern "C++" ARCCORE_MESSAGEPASSING_EXPORT Request \
  mpNonBlockingAllToAllVariable(IMessagePassingMng* pm, Span<const type> send_buf, ConstArrayView<Int32> send_count, \
                                ConstArrayView<Int32> send_index, Span<type> recv_buf, \
                                ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une liste de requêtes.
 *
 * \sa IRequestList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<IRequestList>
mpCreateRequestListRef(IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Bloque tant que les requêtes de \a requests ne sont pas terminées.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWaitAll(IMessagePassingMng* pm, ArrayView<Request> requests);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Bloque jusqu'à ce que la requête \a request soit terminée.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWait(IMessagePassingMng* pm, Request request);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Bloque jusqu'à ce qu'au moins une des requêtes de \a request soit terminée.
 *
 * En retour, le tableaux \a indexes contient la valeur \a true pour indiquer
 * qu'une requête est terminée.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWaitSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste si des requêtes de \a request sont terminées.
 *
 * En retour, le tableaux \a indexes contient la valeur \a true pour indiquer
 * qu'une requête est terminée.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpTestSome(IMessagePassingMng* pm, ArrayView<Request> requests, ArrayView<bool> indexes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction générale d'attente de terminaison de requête.
 *
 * En fonction de la valeur de \a wait_type, appelle mpWait(), mpWaitSome(), ou
 * mpTestSome().
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpWait(IMessagePassingMng* pm, ArrayView<Request> requests,
       ArrayView<bool> indexes, eWaitType wait_type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste si un message est disponible.
 *
 * Cette fonction permet de savoir si un message issu du couple (rang,tag)
 * est disponible. \a message doit avoir été initialisé avec un couple (rang,tag)
 * (message.isRankTag() doit être vrai).
 *
 * Retourne une instance de \a MessageId.
 *
 * En mode non bloquant, si aucun message n'est disponible, alors
 * MessageId::isValid() vaut \a false pour l'instance retournée.
 *
 * La sémantique est identique à celle de MPI_Mprobe. Le message retourné est enlevé
 * de la liste des messages et donc un appel ultérieur à cette méthode avec les mêmes
 * paramètres retournera un autre message ou un message nul. Si on souhaite un
 * comportement identique à MPI_Iprobe()/MPI_Probe() alors il faut utiliser mpLegacyProbe().
 */
ARCCORE_MESSAGEPASSING_EXPORT MessageId
mpProbe(IMessagePassingMng* pm, const PointToPointMessageInfo& message);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste si un message est disponible.
 *
 * Cette fonction permet de savoir si un message issu du couple (rang,tag)
 * est disponible. \a message doit avoir été initialisé avec un couple (rang,tag)
 * (message.isRankTag() doit être vrai).
 *
 * Retourne une instance de \a MessageSourceInfo. En mode non bloquant, si aucun message
 * n'est disponible, alors MessageSourceInfo::isValid() vaut \a false pour
 * l'instance retournée.
 *
 * La sémantique est identique à celle de MPI_Probe. Il est donc possible
 * si on appelle plusieurs fois cette fonction de retourner le même message.
 * Il n'est pas garanti non plus si on fait un mpReceive() avec l'instance retournée
 * d'avoir le même message. Pour toutes ces raisons il est préférable d'utiliser
 * la fonction mpProbe().
 */
ARCCORE_MESSAGEPASSING_EXPORT MessageSourceInfo
mpLegacyProbe(IMessagePassingMng* pm, const PointToPointMessageInfo& message);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une nouvelle instance de \a IMessagePassingMng.
 *
 * \a keep est vrai si ce rang est présent dans le nouveau communicateur.
 *
 * L'instance retournée doit être détruite par l'appel à l'opérateur
 * operator delele().
 */
ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng*
mpSplit(IMessagePassingMng* pm, bool keep);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue une barrière
 *
 * Bloque tant que tous les rangs n'ont pas atteint cette appel.
 */
ARCCORE_MESSAGEPASSING_EXPORT void
mpBarrier(IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue une barrière non bloquante.
 */
ARCCORE_MESSAGEPASSING_EXPORT Request
mpNonBlockingBarrier(IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une liste de messages de sérialisation.
 *
 * \sa ISerializeMessageList
 */
ARCCORE_MESSAGEPASSING_EXPORT Ref<ISerializeMessageList>
mpCreateSerializeMessageListRef(IMessagePassingMng* pm);

//! Message d'envoi utilisant un ISerializer.
ARCCORE_MESSAGEPASSING_EXPORT Request
mpSend(IMessagePassingMng* pm, const ISerializer* values, const PointToPointMessageInfo& message);

//! Message de réception utilisant un ISerializer.
ARCCORE_MESSAGEPASSING_EXPORT Request
mpReceive(IMessagePassingMng* pm, ISerializer* values, const PointToPointMessageInfo& message);

//! Message allGather() pour une sérialisation
ARCCORE_MESSAGEPASSING_EXPORT void
mpAllGather(IMessagePassingMng* pm, const ISerializer* send_serializer, ISerializer* recv_serializer);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(char)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(signed char)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned char)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(short)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned short)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(int)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned int)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long long)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(float)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(double)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long double)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(BFloat16)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(Float16)

#undef ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
using Arcane::MessagePassing::mpSend;
using Arcane::MessagePassing::mpReceive;
using Arcane::MessagePassing::mpAllGather;
using Arcane::MessagePassing::mpGather;
using Arcane::MessagePassing::mpNonBlockingAllGather;
using Arcane::MessagePassing::mpNonBlockingGather;
using Arcane::MessagePassing::mpAllGatherVariable;
using Arcane::MessagePassing::mpGatherVariable;
using Arcane::MessagePassing::mpGather;
using Arcane::MessagePassing::mpScatterVariable;
using Arcane::MessagePassing::mpAllReduce;
using Arcane::MessagePassing::mpNonBlockingAllReduce;
using Arcane::MessagePassing::mpBroadcast;
using Arcane::MessagePassing::mpNonBlockingBroadcast;
using Arcane::MessagePassing::mpAllToAll;
using Arcane::MessagePassing::mpNonBlockingAllToAll;
using Arcane::MessagePassing::mpAllToAllVariable;
using Arcane::MessagePassing::mpNonBlockingAllToAllVariable;
using Arcane::MessagePassing::mpCreateRequestListRef;
using Arcane::MessagePassing::mpWaitAll;
using Arcane::MessagePassing::mpWait;
using Arcane::MessagePassing::mpWaitSome;
using Arcane::MessagePassing::mpTestSome;
using Arcane::MessagePassing::mpProbe;
using Arcane::MessagePassing::mpLegacyProbe;
using Arcane::MessagePassing::mpSplit;
using Arcane::MessagePassing::mpBarrier;
using Arcane::MessagePassing::mpNonBlockingBarrier;
using Arcane::MessagePassing::mpCreateSerializeMessageListRef;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
