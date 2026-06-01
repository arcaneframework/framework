// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFunctorWithArgument.h                                      (C) 2000-2005 */
/*                                                                           */
/* Interface of a functor with argument.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IAMRTRANSPORTFUNCTOR_H
#define ARCANE_UTILS_IAMRTRANSPORTFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/ItemTypes.h"

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/UtilsTypes.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum AMROperationType {
	Restriction= 0,
	Prolongation= 1
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a functor with argument.
 */
class IAMRTransportFunctor
{
 public:
	
  //! Frees resources
  virtual ~IAMRTransportFunctor(){}

 public:

  //! Executes the associated method
  virtual void executeFunctor(Array<ItemInternal*>& old_items,AMROperationType op) =0;
  //! Executes the associated method
  virtual void executeFunctor(Array<Cell>& old_items,AMROperationType op) =0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
