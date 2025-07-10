// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* INodeWindow.h                                              (C) 2000-2025 */
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

template<class Type>
class ARCCORE_MESSAGEPASSING_EXPORT INodeWindow
{
 public:

  virtual ~INodeWindow() = default;

 public:

  virtual Int64 sizeLocalSegment() const = 0;
  virtual Int64 sizeOtherRankSegment(int rank) const = 0;

  virtual ArrayView<Type> localSegmentView() = 0;
  virtual ArrayView<Type> otherRankSegmentView(int rank) = 0;

  virtual ConstArrayView<Type> localSegmentConstView() const = 0;
  virtual ConstArrayView<Type> otherRankSegmentConstView(int rank) const = 0;

  virtual Type* data() = 0;
  virtual Type* dataOtherRank(int rank) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

