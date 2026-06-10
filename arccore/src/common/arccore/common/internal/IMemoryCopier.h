// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryCopier.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface for memory copies.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_IMEMORYCOPIER_H
#define ARCCORE_COMMON_INTERNAL_IMEMORYCOPIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/IMemoryResourceMng.h"
#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for memory copies with accelerator support.
 */
class ARCCORE_COMMON_EXPORT IMemoryCopier
{
 public:

  virtual ~IMemoryCopier() = default;

 public:

  /*!
   * \brief Copies the data from \a from to \a to with the queue \a queue.
   *
   * \a queue may be null.
   */
  virtual void copy(ConstMemoryView from, eMemoryResource from_mem,
                    MutableMemoryView to, eMemoryResource to_mem,
                    const RunQueue* queue) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
