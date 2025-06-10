// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceRegisterer.h                                         (C) 2000-2025 */
/*                                                                           */
/* Singleton permettant d'enregistrer un service.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERCVICEREGISTERER_H
#define ARCANE_CORE_SERCVICEREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ServiceProperty.h"
#include "arcane/core/ModuleProperty.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistreur de service et modules
 *
 * Cette classe implémente le pattern Singleton pour un service donnée.
 * 
 * Elle permet de déclarer une variable globale qui enregistre automatiquement
 * le IServiceFactory du service souhaité. Cette classe ne s'utilise pas
 * directement mais par l'intermédiaire de la macro ARCANE_DEFINE_SERVICE(name).
 * 
 * Comme cette classe est utilisé avec des instances globales, elles sont
 * construites avant que le code ne rentre dans le main(). Il faut donc
 * faire très attention à n'utiliser aucun objet ni faire d'allocation( En
 * particulier, le nom du service doit être un const char* standard).
 * Pour cela, la liste des services enregistré est maintenu dans une liste
 * chaînée et chaque instance contient le pointeur vers le membre suivant et
 * précédent de la liste. Le premier élément de la liste est obtenu par
 * l'appel à ServiceRegisterer::firstService().
 */
class ARCANE_CORE_EXPORT ServiceRegisterer
{
 public:

  typedef IModuleFactoryInfo* (*ModuleFactoryWithPropertyFunc)(const ModuleProperty& properties);
  typedef IServiceInfo* (*ServiceInfoWithPropertyCreateFunc)(const ServiceProperty& properties);

 public:

  /*!
   * \brief Crée en enregistreur pour le service \a name et la fonction \a func.
   *
   * Ce constructeur est utilisé pour enregistrer un service.
   */
  ServiceRegisterer(ServiceInfoWithPropertyCreateFunc func, const ServiceProperty& properties) ARCANE_NOEXCEPT;

  /*!
   * \brief Crée en enregistreur pour le module \a name avec les propriétés \a properties.
   *
   * Ce constructeur est utilisé pour enregistrer un module.
   */
  ServiceRegisterer(ModuleFactoryWithPropertyFunc func, const ModuleProperty& properties) ARCANE_NOEXCEPT;

 public:

  /*!
   * \brief Fonction de création de l'instance 'ServiceInfo' si on est un service.
   *
   * Ce pointeur peut-être nul si on n'est pas un service, auquel cas
   * il faut utiliser infoCreatorFunction().
   */
  ServiceInfoWithPropertyCreateFunc infoCreatorWithPropertyFunction() { return m_info_function_with_property; }

  /*!
   * \brief Fonction de création de la factory si on est un module.
   *
   * Ce pointeur peut-être nul si on n'est pas un module, auquel cas
   * il faut utiliser infoCreatorFunction().
   */
  ModuleFactoryWithPropertyFunc moduleFactoryWithPropertyFunction() { return m_module_factory_with_property_functor; }

  //! Nom du service
  const char* name() { return m_name; }

  /*!
   * \brief Propriétés du service.
   *
   * \deprecated Utiliser \a serviceProperty() à la place
   */
  ARCANE_DEPRECATED_260 const ServiceProperty& property() const { return m_service_property; }

  //! Propriétés dans le cas d'un service
  const ServiceProperty& serviceProperty() const { return m_service_property; }

  //! Propriétés dans le cas d'un module
  const ModuleProperty& moduleProperty() const { return m_module_property; }

  //! Service précédent (0 si le premier)
  ServiceRegisterer* previousService() const { return m_previous; }

  //! Service suivant (0 si le dernier)
  ServiceRegisterer* nextService() const { return m_next; }

 private:

  //! Positionne le service précédent
  /*! Utilisé en interne pour construire la chaine de service */
  void setPreviousService(ServiceRegisterer* s) { m_previous = s; }

  //! Positionne le service suivant
  /*! Utilisé en interne pour construire la chaine de service */
  void setNextService(ServiceRegisterer* s) { m_next = s; }

 public:

  //! Accès au premier élément de la chaine d'enregistreur de service
  static ServiceRegisterer* firstService();

  //! Nombre d'enregistreurs de service dans la chaine
  static Integer nbService();

 private:

  //! Fonction de création du IModuleFactory
  ModuleFactoryWithPropertyFunc m_module_factory_with_property_functor = nullptr;
  //! Fonction de création du IServiceInfo
  ServiceInfoWithPropertyCreateFunc m_info_function_with_property = nullptr;
  //! Nom du service
  const char* m_name = nullptr;
  //! Propriétés du service
  ServiceProperty m_service_property;
  //! Propriétés du module
  ModuleProperty m_module_property;
  //! Service précédent
  ServiceRegisterer* m_previous = nullptr;
  //! Service suivant
  ServiceRegisterer* m_next = nullptr;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
