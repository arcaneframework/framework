// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ActionWrapperService.h                                      (C) 2000-2006 */
/*                                                                           */
/* Service faisant un wrapper autour d'une action.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACTIONWRAPPER_H
#define ARCANE_ACTIONWRAPPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IActionWrapperService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceBuildInfo;
class IApplication;
class IServiceInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Wrapper autour d'une action.
 *
 */ 
class ActionWrapperService
: public IActionWrapperService
{
 public:

  ActionWrapperService(const ServiceBuildInfo& sbi);
  virtual ~ActionWrapperService();

 public:
  
  //! Parent de ce service
  virtual IBase* serviceParent() const;
    
  //! Informations du service
  virtual IServiceInfo* serviceInfo() const
    { return m_service_info; }

 private:

  IApplication* m_application;
  IServiceInfo* m_service_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

