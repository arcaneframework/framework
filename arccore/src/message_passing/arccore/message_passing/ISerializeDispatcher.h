// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializeDispatcher.h                                      (C) 2000-2025 */
/*                                                                           */
/* Interface for serialization messages.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISERIALIZEDISPATCHER_H
#define ARCCORE_MESSAGEPASSING_ISERIALIZEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for serialization messages
 */
class ARCCORE_MESSAGEPASSING_EXPORT ISerializeDispatcher
{
 public:

  virtual ~ISerializeDispatcher() = default;

 public:

  //! Create a list of serialization messages
  virtual Ref<ISerializeMessageList> createSerializeMessageListRef() =0;

  //! Sending message
  virtual Request
  sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message) =0;

  //! Receiving message
  virtual Request
  receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
