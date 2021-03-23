// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAlephFactory.h                                             (C) 2000-2015 */
/*                                                                           */
/* Interface des fabriques pour Aleph.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_IALEPHFACTORY_H
#define ARCANE_ALEPH_IALEPHFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/aleph/AlephGlobal.h"
#include "arcane/aleph/AlephInterface.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/******************************************************************************
 * IAlephFactory::IAlephFactory
 *****************************************************************************/
class ARCANE_ALEPH_EXPORT AlephFactory
: public IAlephFactory
{
 private:
  class FactoryImpl;

 public:
  AlephFactory(IApplication* app, ITraceMng* tm);
  ~AlephFactory();

 public:
  IAlephTopology* GetTopology(AlephKernel* kernel, Integer index, Integer nb_row_size);
  IAlephVector* GetVector(AlephKernel* kernel, Integer index);
  IAlephMatrix* GetMatrix(AlephKernel* kernel, Integer index);
  virtual bool hasSolverImplementation(Integer id);

 private:
  typedef std::map<Integer, FactoryImpl*> FactoryImplMap;
  FactoryImplMap m_impl_map;
  IAlephFactoryImpl* _getFactory(Integer solver_index);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_IALEPH_FACTORY_H
