﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicService.h                                              (C) 2000-2021 */
/*                                                                           */
/* Classe de base d'un service lié à un sous-domaine.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_BASICSERVICE_H
#define ARCANE_BASICSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshAccessor.h"
#include "arcane/AbstractService.h"
#include "arcane/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceBuildInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de service lié à un sous-domaine.
 *
 * \ingroup Service
 */
class ARCANE_CORE_EXPORT BasicService
: public AbstractService
, public MeshAccessor
, public CommonVariables
{
 protected:

  explicit BasicService(const ServiceBuildInfo&);

 public:
	
  ~BasicService() override; //!< Libère les ressources.

 public:

  virtual ISubDomain* subDomain() { return m_sub_domain; }

 private:

  ISubDomain* m_sub_domain;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

