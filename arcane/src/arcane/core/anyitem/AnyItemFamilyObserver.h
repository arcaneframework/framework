// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemFamilyObserver.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interfaces Observeur pour la famille et famille de liens                  */
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
 * \brief Interface d'observeurs de famille AnyItem
 */
class IFamilyObserver
{ 
public:
  
  virtual ~IFamilyObserver() {}
  
  //! Notifie à l'observeur que la famille est invalidée
  virtual void notifyFamilyIsInvalidate() = 0; 

  //! Notifie à l'observeur que la famille est agrandie
  virtual void notifyFamilyIsIncreased() = 0;
};

/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface d'observeurs de famille de liaisons AnyItem
 */
class ILinkFamilyObserver
{ 
public:
  
  virtual ~ILinkFamilyObserver() {}

  //! Notifie à l'observeur que la famille est invalidée
  virtual void notifyFamilyIsInvalidate() = 0; 

  //! Notifie à l'observeur que la famille est reservée
  virtual void notifyFamilyIsReserved() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMFAMILY_H */
