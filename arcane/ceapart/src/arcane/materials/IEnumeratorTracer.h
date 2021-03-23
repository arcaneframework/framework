// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IEnumeratorTracer.h                                         (C) 2000-2018 */
/*                                                                           */
/* Interface du tracage des énumérateurs sur les composants.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IENUMERATORTRACER_H
#define ARCANE_MATERIALS_IENUMERATORTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class EnumeratorTraceInfo;
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ComponentEnumerator;
class MatEnumerator;
class EnvEnumerator;
class ComponentCellEnumerator;
class AllEnvCellEnumerator;
class CellComponentCellEnumerator;
class ComponentPartSimdCellEnumerator;
class ComponentPartCellEnumerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT IEnumeratorTracer
{
 private:
  static IEnumeratorTracer* m_singleton;
 public:
  static IEnumeratorTracer* singleton() { return m_singleton; }
  //! Internal
  static void _setSingleton(IEnumeratorTracer* tracer);
 public:
  virtual ~IEnumeratorTracer(){}
 public:
  virtual void enterEnumerator(const ComponentEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const ComponentEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const MatEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const MatEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const EnvEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const EnvEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const ComponentCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const ComponentCellEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const AllEnvCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const AllEnvCellEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const CellComponentCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const CellComponentCellEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const ComponentPartSimdCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const ComponentPartSimdCellEnumerator& e,EnumeratorTraceInfo& eti) =0;

  virtual void enterEnumerator(const ComponentPartCellEnumerator& e,EnumeratorTraceInfo& eti,const TraceInfo* ti) =0;
  virtual void exitEnumerator(const ComponentPartCellEnumerator& e,EnumeratorTraceInfo& eti) =0;

 public:

  virtual void dumpStats() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

