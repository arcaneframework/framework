// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IService.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICE_H
#define ARCANE_CORE_ISERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/ExternalRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service.
 *
 * Les instances retournées par serviceInfo() et serviceParent() sont la
 * propriété de l'application (interface IApplication) et ne doivent jamais
 * être modifiée ni détruite.
 *
 * \deprecated
 */
class ARCANE_CORE_EXPORT IService
{
 protected:

  //! Constructeur
  IService() {}

 public:

  virtual ~IService() {} //!< Libère les ressources

 public:

  //! Parent de ce service
  virtual IBase* serviceParent() const = 0;

  //! Interface de ce service (normalement this)
  virtual IService* serviceInterface() = 0;

  //! Informations du service
  virtual IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une instance d'un service.
 */
class ARCANE_CORE_EXPORT IServiceInstance
{
  friend class Ref<IServiceInstance>;

 protected:

  virtual ~IServiceInstance() = default;

 public:

  //! Ajoute une référence.
  virtual void addReference() = 0;
  //! Supprime une référence.
  virtual void removeReference() = 0;
  virtual IServiceInfo* serviceInfo() const = 0;
  //! \internal
  virtual Internal::ExternalRef _internalDotNetHandle() const { return {}; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une instance de service singleton.
 */
class ARCANE_CORE_EXPORT ISingletonServiceInstance
: public IServiceInstance
{
 public:

  //! Liste des instances des interfaces implémentées par le singleton
  virtual ServiceInstanceCollection interfaceInstances() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant l'instance d'un service.
 */
template <typename InterfaceType>
class IServiceInstanceT
: public IServiceInstance
{
 public:

  virtual Ref<InterfaceType> instance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
