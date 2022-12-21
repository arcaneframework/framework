// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicService.h                                              (C) 2000-2022 */
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

// Pour indiquer qu'on a une classe de base spécifique pour chaque type de
// service. Utilisé seulement par 'axlstar' pour la version 3.8 de Arcane.
// Par la suite on pourra supprimer cela
#define ARCANE_HAS_SPECIFIC_BASIC_SERVICE

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

  ISubDomain* m_sub_domain;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de service lié à une option du jeu de données.
 *
 * \ingroup Service.
 *
 * Pour l'instant (décembre 2022) cette classe dérive de 'BasicService' pour
 * des raisons de compatibilité avec l'existant mais à terme ce ne sera plus
 * le cas.
 */
class ARCANE_CORE_EXPORT BasicCaseOptionService
: public BasicService
{
 protected:

  explicit BasicCaseOptionService(const ServiceBuildInfo& s)
  : BasicService(s)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de service lié à un maillage.
 *
 * \ingroup Service.
 *
 * Pour l'instant (décembre 2022) cette classe dérive de 'BasicService' pour
 * des raisons de compatibilité avec l'existant mais à terme ce ne sera plus
 * le cas.
 */
class ARCANE_CORE_EXPORT BasicMeshService
: public BasicService
{
 protected:

  explicit BasicMeshService(const ServiceBuildInfo& s)
  : BasicService(s)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de service lié à un sous-domaine.
 *
 * \ingroup Service.
 */
class ARCANE_CORE_EXPORT BasicSubDomainService
: public BasicService
{
 protected:

  explicit BasicSubDomainService(const ServiceBuildInfo& s)
  : BasicService(s)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
