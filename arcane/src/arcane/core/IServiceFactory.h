// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceFactory.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface de la manufacture des services.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEFACTORY_H
#define ARCANE_CORE_ISERVICEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/core/ArcaneTypes.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la fabrique d'un service.
 *
 * Cette interface contient les informations nécessaire sur une fabrique
 * d'un service.
 *
 * En général les instances de cette classe sont soit créées par Arcane
 * à partir d'un fichier axl, soit  ou en utilisant une des macros
 * de fabrique de service (définies dans le fichier ServiceFactory.h).
 *
 * La liste des interfaces supportées par le service et la
 * fabrique associées sont décrites dans serviceInfo().
 */
class ARCANE_CORE_EXPORT IServiceFactoryInfo
{
 public:

  //! Libère les ressources
  virtual ~IServiceFactoryInfo() {}

 public:

  //! vrai si le service est un module et doit être chargé automatiquement
  //TODO: regarder si autoload est encore utile pour ces services.
  virtual bool isAutoload() const =0;
  //! vrai si le service est un service singleton (une seule instance)
  virtual bool isSingleton() const =0;

 public:

  /*! \brief Informations sur le service pouvant être créé par cette fabrique.
   *
   * L'instance retournée reste la propriété de l'application l'ayant créée
   * et ne doit ni être modifiée, ni être détruite.
   */  
  virtual IServiceInfo* serviceInfo() const =0;
};

namespace Internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief internal.
 * \brief Interface d'une fabrique pour un service (nouvelle version).
 *
 * Cette classe s'utiliser via un ReferenceCounter pour gérer sa destruction.
 */
class ARCANE_CORE_EXPORT IServiceFactory2
{
 protected:
  virtual ~IServiceFactory2() = default;
 public:
  //! Ajoute une référence.
  virtual void addReference() =0;
  //! Supprime une référence.
  virtual void removeReference() =0;
 public:
  //! Créé une instance du service à partir des infos de \a sbi.
  virtual ServiceInstanceRef createServiceInstance(const ServiceBuildInfoBase& sbi) =0;
  
  //! Retourne le IServiceInfo associé à cette fabrique.
  virtual IServiceInfo* serviceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief internal.
 * \brief Classe de base pour une fabrique pour un service.
 *
 * Cette classe s'utiliser via un ReferenceCounter pour gérer sa destruction.
 */
class ARCANE_CORE_EXPORT AbstractServiceFactory
: public IServiceFactory2
{
 protected:
  AbstractServiceFactory() : m_nb_ref(0){}
 public:
  void addReference() override;
  void removeReference() override;
 private:
  std::atomic<Int32> m_nb_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique pour un service implémentant l'interface \a InterfaceType.
 */
template<typename InterfaceType>
class IServiceFactory2T
: public AbstractServiceFactory
{
 public:
  virtual Ref<InterfaceType> createServiceReference(const ServiceBuildInfoBase& sbi) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief internal.
 * \brief Fabrique pour un service singleton.
 * Un service singleton n'est créé qu'une seule fois mais peut avoir plusieurs
 * interfaces et il y a donc autant de IServiceInstance que d'interfaces
 * implémentées par le service.
 *
 * La méthode \a createSingletonServiceInstance() permet de créér l'instance
 * du service singleton ainsi que les IServiceInstance pour chaque interface
 * implémentée.
 */
class ARCANE_CORE_EXPORT ISingletonServiceFactory
{
 public:
  virtual ~ISingletonServiceFactory() = default;

  //! Créé une instance d'un service singleton.
  virtual Ref<ISingletonServiceInstance>
  createSingletonServiceInstance(const ServiceBuildInfoBase& sbi) =0;

  //! Retourne le IServiceInfo associé à cette fabrique.
  virtual IServiceInfo* serviceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

