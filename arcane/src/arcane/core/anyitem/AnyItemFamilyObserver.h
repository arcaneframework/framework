// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemFamilyObserver.h                                     (C) 2000-2025 */
/*                                                                           */
/* Observer Interfaces for family and link family                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMFAMILYOBSERVER_H
#define ARCANE_CORE_ANYITEM_ANYITEMFAMILYOBSERVER_H 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
  
/*!
 * \brief AnyItem family observer interface
 */
class IFamilyObserver
{ 
public:
  
  virtual ~IFamilyObserver() {}
  
  //! Notifies the observer that the family is invalidated
  virtual void notifyFamilyIsInvalidate() = 0; 

  //! Notifies the observer that the family has been increased
  virtual void notifyFamilyIsIncreased() = 0;
};

/*---------------------------------------------------------------------------*/

/*!
 * \brief AnyItem link family observer interface
 */
class ILinkFamilyObserver
{ 
public:
  
  virtual ~ILinkFamilyObserver() {}

  //! Notifies the observer that the family is invalidated
  virtual void notifyFamilyIsInvalidate() = 0; 

  //! Notifies the observer that the family is reserved
  virtual void notifyFamilyIsReserved() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMFAMILY_H */
