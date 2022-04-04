// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CriticalSection.h                                           (C) 2000-2008 */
/*                                                                           */
/* Section critique en multi-thread.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CRITICALSECTION_H
#define ARCANE_UTILS_CRITICALSECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IThreadMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Section critique en multi-thread.
 */
class ARCANE_UTILS_EXPORT CriticalSection
{
 public:
  CriticalSection(IThreadMng* tm)
  : m_thread_mng(tm)
  {
    m_thread_mng->beginCriticalSection();
  }
  ~CriticalSection()
  {
    m_thread_mng->endCriticalSection();
  }
 private:
  IThreadMng* m_thread_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
