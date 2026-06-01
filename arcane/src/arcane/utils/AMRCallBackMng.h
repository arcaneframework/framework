// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CallBackDefinition.h                                        (C) 2000-2010 */
/*                                                                           */
/* Callback function manager.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_AMRCALLBACKMNG_H
#define ARCANE_UTILS_AMRCALLBACKMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/List.h"
#include "arcane/utils/AMRComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT AMRCallBackMng
{
  typedef List<IAMRTransportFunctor*> IAMRTransportFunctorList;

 public:

  AMRCallBackMng() {};
  ~AMRCallBackMng() {};

 public:

  void initialize();
  void finalize();

  void registerCallBack(IAMRTransportFunctor*);

  void unregisterCallBack(IAMRTransportFunctor*);

  void callCallBacks(Array<ItemInternal*>& old, AMROperationType op);

  void callCallBacks(Array<Cell>& old, AMROperationType op);

 private:

  IAMRTransportFunctorList m_amr_transport_functors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
