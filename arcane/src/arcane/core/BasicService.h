// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicService.h                                              (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un service lié à un sous-domaine.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_BASICSERVICE_H
#define ARCANE_CORE_BASICSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshAccessor.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

  ISubDomain* m_sub_domain = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

