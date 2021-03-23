// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CallBackDefinition.h                                        (C) 2000-2010 */
/*                                                                           */
/* Gestionnaire des fonctions callback.                                        */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// -------------------------------------------------------------------
class ARCANE_UTILS_EXPORT AMRCallBackMng
{
	typedef List<IAMRTransportFunctor *>  IAMRTransportFunctorList;
public:

public:

  AMRCallBackMng() {};
  ~AMRCallBackMng() {};

  void initialize();
  void finalize();

  void registerCallBack(IAMRTransportFunctor *);


  void unregisterCallBack(IAMRTransportFunctor *);


  void callCallBacks(Array<ItemInternal*>& old,AMROperationType op);

  void callCallBacks(Array<Cell>& old,AMROperationType op);

private:

  IAMRTransportFunctorList m_amr_transport_functors;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_UTILS_AMRCALLBACKMNG_H_ */

