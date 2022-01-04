// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemEnumeratorTracer.h                                     (C) 2000-2018 */
/*                                                                           */
/* Interface de trace des appels aux énumérateur sur les entités.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMENUMERATORTRACER_H
#define ARCANE_IITEMENUMERATORTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemEnumerator;
class EnumeratorTraceInfo;
class SimdItemEnumeratorBase;
class IPerformanceCounterService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un traceur d'énumérateur sur les entités.
 */
class ARCANE_CORE_EXPORT IItemEnumeratorTracer
{
 private:
  static IItemEnumeratorTracer* m_singleton;
 public:
  static IItemEnumeratorTracer* singleton() { return m_singleton; }
  //! Internal
  static void _setSingleton(IItemEnumeratorTracer* tracer);
 public:
  virtual ~IItemEnumeratorTracer(){}
 public:
  virtual void enterEnumerator(const ItemEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const ItemEnumerator& e,EnumeratorTraceInfo& eti) =0;
  virtual void enterEnumerator(const SimdItemEnumeratorBase& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const SimdItemEnumeratorBase& e,EnumeratorTraceInfo& eti) =0;
 public:
  virtual void dumpStats() =0;
  virtual IPerformanceCounterService* perfCounter() =0;
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
