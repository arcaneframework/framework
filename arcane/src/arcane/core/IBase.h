// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBase.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Interface d'un objet de base.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IBASE_H
#define ARCANE_CORE_IBASE_H
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
 * \brief Interface de la classe de base des objets principaux arcane
 */
class ARCANE_CORE_EXPORT IBase
{
 public:

  virtual ~IBase() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Construit les membres de l'instance.
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée. Cette méthode doit être appelée avant initialize().
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void build() = 0;

  /*!
   * \brief Initialise l'instance.
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée.
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void initialize() = 0;

 public:

  //! Parent de cet objet
  virtual IBase* objectParent() const = 0;

  //! Namespace de l'objet.
  virtual String objectNamespaceURI() const = 0;

  //! Nom local de l'objet.
  virtual String objectLocalName() const = 0;

  //! Numéro de version du service.
  virtual VersionInfo objectVersion() const = 0;

 public:

  //! Gestionnaire de traces
  virtual ITraceMng* traceMng() const = 0;

  //! Gestionnaire de ressources
  virtual IRessourceMng* ressourceMng() const = 0;

  //! Gestionnaire de services
  virtual IServiceMng* serviceMng() const = 0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT String
arcaneNamespaceURI();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

