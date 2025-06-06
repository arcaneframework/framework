// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un service.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTSERVICE_H
#define ARCANE_CORE_ABSTRACTSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un service.
 *
 * Cette classe est LA classe d'implémentation bas niveau de l'interface \a IService.
 *
 * \ingroup Service
 */
class ARCANE_CORE_EXPORT AbstractService
: public TraceAccessor
, public IService
{
 protected:

  //! Constructeur à partir d'un \a ServiceBuildInfo
  explicit AbstractService(const ServiceBuildInfo&);

 public:

  //! Destructeur
  ~AbstractService() override;

 public:

  /*!
   * \brief Construction de niveau \a build du service.
   *
   * Cette méthode est appelé juste après le constructeur.
   */
  virtual void build() {}

 public:

  //! Accès aux informations du service.  Voir \a IServiceInfo pour les détails
  IServiceInfo* serviceInfo() const override { return m_service_info; }

  //! Accès à l'interface de base des principaux objets Arcane
  IBase* serviceParent() const override { return m_parent; }

  //! Retourne l'interface bas niveau \a IService du service
  IService* serviceInterface() override { return this; }

 private:

  IServiceInfo* m_service_info = nullptr;
  IBase* m_parent = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
