// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* INodeWindowBase.h                                              (C) 2000-2025 */
/*                                                                           */
/* TODO.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_INODEWINDOW_H
#define ARCCORE_MESSAGEPASSING_INODEWINDOW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSING_EXPORT INodeWindowBase
{
 public:

  virtual ~INodeWindowBase() = default;

 public:

  virtual Integer sizeofOneElem() const = 0;

  virtual Integer sizeLocalSegment() const = 0;
  virtual Integer sizeOtherRankSegment(int rank) const = 0;

  virtual void* data() = 0;
  virtual void* dataOtherRank(int rank) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

