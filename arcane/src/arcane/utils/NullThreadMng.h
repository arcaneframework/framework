// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullThreadMng.h                                             (C) 2000-2010 */
/*                                                                           */
/* Single-threaded mode thread manager.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NULLTHREADMNG_H
#define ARCANE_UTILS_NULLTHREADMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/NullThreadImplementation.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/IThreadMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Thread manager in single-threaded mode.
 */
class ARCANE_UTILS_EXPORT NullThreadMng
: public IThreadMng
{
 public:

  virtual ~NullThreadMng() {}

 public:

  virtual void beginCriticalSection() {}
  virtual void endCriticalSection() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
