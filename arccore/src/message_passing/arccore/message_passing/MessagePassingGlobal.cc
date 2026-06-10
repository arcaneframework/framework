// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingGlobal.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Global definitions for the 'message_passing' component of 'Arccore'.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// It is necessary to include all header files that are not
// directly included by '.cc' files to ensure that
// external symbols are properly defined.
#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing/ISerializeMessageList.h"
#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/IProfiler.h"
#include "arccore/message_passing/internal/SubRequestCompletionInfo.h"
#include "arccore/message_passing/internal/IContigMachineShMemWinBaseInternal.h"
#include "arccore/message_passing/internal/IMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file MessagePassingGlobal.h
 *
 * \brief General declarations for the 'message_passing' component
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Namespace containing the types and declarations that manage
 * the message-passing parallelism mechanism.
 */
namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
