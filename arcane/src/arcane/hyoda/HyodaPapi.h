// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaPapi.h                                                 (C) 2000-2013 *
 *****************************************************************************/
#ifndef ARCANE_HYODA_PAPI_H
#define ARCANE_HYODA_PAPI_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class Hyoda;
class HyodaTcp;

/******************************************************************************
 * HyodaMix CLASS
 *****************************************************************************/
class HyodaPapi: public TraceAccessor{
public:
  HyodaPapi(Hyoda*, IApplication*, ITraceMng*);
  ~HyodaPapi();
public:
  void initialize(ISubDomain*,HyodaTcp*);
  void start(void);
  void stop(void);
  void dump(void);
private:
  Hyoda *m_hyoda;
  IApplication *m_app;
  ISubDomain *m_sub_domain;
  IProfilingService *m_papi;
  HyodaTcp *m_tcp;
  Int64UniqueArray pkt;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  // ARCANE_HYODA_PAPI_H
