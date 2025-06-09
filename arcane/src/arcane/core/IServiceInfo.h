// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceInfo.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface des informations d'un service.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEINFO_H
#define ARCANE_CORE_ISERVICEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Service
 * \brief Interface des informations d'un service ou d'un module.
 */
class ARCANE_CORE_EXPORT IServiceInfo
{
 public:

  static const Integer Dim1 = 1;
  static const Integer Dim2 = 2;
  static const Integer Dim3 = 4;

 public:

  virtual ~IServiceInfo() = default; //!< Libère les ressources

 public:

  //! Partie locale du nom du service
  virtual String localName() const = 0;

  //! Namespace du nom du service
  virtual String namespaceURI() const = 0;

  //! Version du service
  virtual VersionInfo version() const = 0;

  //! Version du fichier axl décrivant ce service
  virtual Real axlVersion() const = 0;

  //! Indique si le service est utilisable en dimension \a n.
  virtual bool allowDimension(Integer n) const = 0;

  /*! \brief Ajoute l'interface de nom \a name aux interfaces
   * implémentées par ce service.
   */
  virtual void addImplementedInterface(const String& name) = 0;

  //! Liste des noms des classes implémentées par ce service
  virtual StringCollection implementedInterfaces() const = 0;

  //! Nom du fichier contenant le jeu de données (nul si aucun)
  virtual const String& caseOptionsFileName() const = 0;

  //! Liste des fabriques du service
  virtual ServiceFactory2Collection factories() const = 0;

  //! Fabrique pour les service singleton (nullptr si non supporté)
  virtual Internal::ISingletonServiceFactory* singletonFactory() const = 0;

  /*! \brief Nom de l'élément XML du service pour le langage \a lang.
   * Si \a lang est nul, retourne le nom par défaut.
   */
  virtual String tagName(const String& lang) const = 0;

  //! Infos sur les fabriques disponibles pour ce service
  virtual IServiceFactoryInfo* factoryInfo() const = 0;

  /*!
   * \brief Indique où peut être utilisé le service.
   *
   * Il s'agit d'une combinaison de valeurs de eServiceType.
   */
  virtual int usageType() const = 0;

  //! Contenu du fichier AXL associé à ce service ou module
  virtual const FileContent& axlContent() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

