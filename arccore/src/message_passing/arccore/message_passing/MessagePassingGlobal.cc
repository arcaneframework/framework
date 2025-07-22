// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingGlobal.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'message_passing' de 'Arccore'.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Il est nécessaire d'inclure tous les fichiers d'en-tête qui ne sont
// pas directement inclus par des fichiers '.cc' pour s'assurer que
// les symboles externes sont bien définis.
#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing/ISerializeMessageList.h"
#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/IProfiler.h"
#include "arccore/message_passing/internal/SubRequestCompletionInfo.h"
#include "arccore/message_passing/internal/IMachineMemoryWindowBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MessagePassingGlobal.h
 *
 * \brief Déclarations générales de la composante 'message_passing'
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Espace de nommage contenant les types et déclarations qui gèrent
 * le mécanisme de parallélisme par échange de message.
 */
namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
