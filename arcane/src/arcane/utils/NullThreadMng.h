// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "arccore/concurrency/NullThreadImplementation.h"
#include "arcane/utils/UtilsTypes.h"
/*---------------------------------------------------------------------------*/
/* NullThreadMng.h                                             (C) 2000-2010 */
/*                                                                           */
/* Gestionnaire de thread en mode mono-thread.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NULLTHREADMNG_H
#define ARCANE_UTILS_NULLTHREADMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IThreadMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de thread en mode mono-thread.
 */
class ARCANE_UTILS_EXPORT NullThreadMng
: public IThreadMng
{
 public:
  virtual ~NullThreadMng(){}
 public:
  virtual void beginCriticalSection() {}
  virtual void endCriticalSection() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
