// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRessourceMng.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface d'un gestionnaire de ressources.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IRESSOURCEMNG_H
#define ARCANE_CORE_IRESSOURCEMNG_H
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
 * \brief Interface d'un gestionnaire de ressource.
 *
 */
class ARCANE_CORE_EXPORT IRessourceMng
{
  // TODO: supprimer cette classe qui n'est plus utile.
  // Il est possible de créer directement une instance
  // de IXmlDocumentHolder

 public:

  //! Création d'un gestionnaire d'historique par défaut.
  static IRessourceMng* createDefault(IApplication*);

 public:

  virtual ~IRessourceMng() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Créé un noeud document XML.
   *
   * Crée et retourne un document XML utilisant une implémentation par défaut.
   * La destruction de ce document invalide tous les noeuds qui en dépendent.
   */
  virtual IXmlDocumentHolder* createXmlDocument() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
