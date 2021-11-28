// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Parallel.h                                                  (C) 2000-2020 */
/*                                                                           */
/* Espace de nom des types gérant le parallélisme.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_H
#define ARCANE_PARALLEL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arccore/base/RefDeclarations.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/Communicator.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/message_passing/IControlDispatcher.h"

#define ARCANE_BEGIN_NAMESPACE_PARALLEL namespace Parallel {
#define ARCANE_END_NAMESPACE_PARALLEL   }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
class IParallelMngContainer;
class IParallelMngContainerFactory;
class IGetVariablesValuesParallelOperation;
class ITransferValuesParallelOperation;
class IParallelExchanger;
class IVariableSynchronizer;
class IParallelTopology;
}

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IParallelMng)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IParallelMngContainer)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
using Arccore::MessagePassing::IMessagePassingMng;
using Arccore::MessagePassing::ITypeDispatcher;
using Arccore::MessagePassing::IControlDispatcher;
using Arccore::MessagePassing::ISerializeMessage;
using Arccore::MessagePassing::ISerializeMessageList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Types des classes du parallélisme.
 */
namespace Parallel
{

using Arccore::MessagePassing::eReduceType;
using Arccore::MessagePassing::ReduceMin;
using Arccore::MessagePassing::ReduceMax;
using Arccore::MessagePassing::ReduceSum;

using Arccore::MessagePassing::eWaitType;
using Arccore::MessagePassing::WaitAll;
using Arccore::MessagePassing::WaitSome;
using Arccore::MessagePassing::TestSome;
using Arccore::MessagePassing::WaitSomeNonBlocking;

using Arccore::MessagePassing::eBlockingType;
using Arccore::MessagePassing::Blocking;
using Arccore::MessagePassing::NonBlocking;

using Arccore::MessagePassing::ePointToPointMessageType;
using Arccore::MessagePassing::MsgSend;
using Arccore::MessagePassing::MsgReceive;

using Arccore::MessagePassing::Request;
using Arccore::MessagePassing::IRequestCreator;
using Arccore::MessagePassing::IRequestList;
using Arccore::MessagePassing::ISubRequest;
using Arccore::MessagePassing::MessageId;
using Arccore::MessagePassing::MessageTag;
using Arccore::MessagePassing::MessageRank;
using Arccore::MessagePassing::PointToPointMessageInfo;

using Arccore::MessagePassing::Communicator;

class IStat;
} // End namespace Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Déclarations des types et méthodes utilisés par les mécanismes
 * d'échange de messages.
 */
namespace MessagePassing
{
/*!
 * \brief Effectue une barrière nommée de nom \a name
 *
 * Effectue une barrière de nom \a name en utilisant le gestionnaire
 * \a pm.
 *
 * Tous les rangs de \a pm bloquent dans cette barrière et vérifient
 * que tous les rangs utilisent le même nom de barrière. Si un des rangs
 * utilise un nom différent une exception est levée.
 *
 * Cette opération permet de vérifier que tous les rangs utilisent
 * la même barrière contrairement à l'opération IParallelMng::barrier().
 *
 * \note Seuls les 1024 premiers caractères de \a name sont utilisés.
 */
extern "C++" ARCANE_CORE_EXPORT void
namedBarrier(IParallelMng* pm,const String& name);

} // End namespace MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const Parallel::Request prequest)
{
  prequest.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

