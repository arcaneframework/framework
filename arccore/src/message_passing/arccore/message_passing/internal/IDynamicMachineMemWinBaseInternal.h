// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDynamicMachineMemWinBaseInternal.h                          (C) 2000-2025 */
/*                                                                           */
/* TODO.                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_IDYNAMICMACHINEMEMWINBASEINTERNAL_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_IDYNAMICMACHINEMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSING_EXPORT IDynamicMachineMemWinBaseInternal
{
 public:

  virtual ~IDynamicMachineMemWinBaseInternal() = default;

 public:

  virtual Int32 sizeofOneElem() const = 0;

  virtual Span<std::byte> segment() const = 0;
  virtual Span<std::byte> segment(Int32 rank) const = 0;

  virtual void add(Span<const std::byte> elem) = 0;

  virtual void exchangeSegmentWith(Int32 rank) = 0;

  virtual ConstArrayView<Int32> machineRanks() const = 0;
  virtual void syncAdd() = 0;
  virtual void barrier() = 0;

  virtual void reserve(Int64 new_capacity) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

