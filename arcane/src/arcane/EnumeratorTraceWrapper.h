// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnumeratorTraceWrapper.h                                    (C) 2000-2016 */
/*                                                                           */
/* Enumérateur sur des groupes d'entités du maillage.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ENUMERATORTRACEWRAPPER_H
#define ARCANE_ENUMERATORTRACEWRAPPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_TRACE_ENUMERATOR
// Décommenter si on souhaite toujours activer les traces des énumérateurs
//#define ARCANE_TRACE_ENUMERATOR
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour les traces d'un énumérator.
 */
class EnumeratorTraceInfo
{
 public:
  Int64ArrayView counters() { return Int64ArrayView(8,m_counters); }
 private:
  Int64 m_counters[8];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper autour d'un énumérator pour les traces.
 *
 * \a TrueEnumerator est le type du véritable énumérateur et
 * \a TraceInterface celui de l'interface de gestion. Le type \a TraceInterface
 * doit posséder les propriétés suivantes:
 * - une méthode singleton() renvoyant une instance.
 * - une méthode enterEnumerator() et une méthode exitEnumerator() pour
 * chaque type d'énumérateur supporté.
 */
template<typename TrueEnumerator,typename TracerInterface>
class EnumeratorTraceWrapper
: public TrueEnumerator
{
 public:
  EnumeratorTraceWrapper(TrueEnumerator&& tenum)
  : TrueEnumerator(tenum), m_tracer(TracerInterface::singleton())
  {
    if (m_tracer)
      m_tracer->enterEnumerator(*this,m_infos,nullptr);
  }
  EnumeratorTraceWrapper(TrueEnumerator&& tenum,const TraceInfo& ti)
  : TrueEnumerator(tenum), m_tracer(TracerInterface::singleton())
  {
    if (m_tracer)
      m_tracer->enterEnumerator(*this,m_infos,&ti);
  }
  ~EnumeratorTraceWrapper() ARCANE_NOEXCEPT_FALSE
  {
    if (m_tracer)
      m_tracer->exitEnumerator(*this,m_infos);
  }
 private:
  TracerInterface* m_tracer;
  EnumeratorTraceInfo m_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_TRACE_ENUMERATOR)
#define A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) \
  ::Arcane::EnumeratorTraceWrapper< ::Arcane::_EnumeratorClassName, ::Arcane::IItemEnumeratorTracer >
#define A_TRACE_ENUMERATOR_WHERE   ,A_FUNCINFO
#else
#define A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) \
  ::Arcane::_EnumeratorClassName
#define A_TRACE_ENUMERATOR_WHERE
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
