// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadBindingMng.h                                          (C) 2000-2021 */
/*                                                                           */
/* Gestionnaire pour punaiser les threads.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_THREADBINDERMNG_H
#define ARCANE_IMPL_INTERNAL_THREADBINDERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/ObserverPool.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT ThreadBindingMng
{
 public:

  void initialize(ITraceMng* tm,const String& strategy);

 private:

  ObserverPool m_observer_pool;
  ITraceMng* m_trace_mng = nullptr;
  String m_bind_strategy;
  Int32 m_current_thread_index = 0;
  Int32 m_max_thread = 0;

 private:

  void _createThreadCallback();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
