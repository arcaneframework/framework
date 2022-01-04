// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRessourceMng.h                                             (C) 2000-2008 */
/*                                                                           */
/* Interface d'un gestionnaire de ressources.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IRESSOURCEMNG_H
#define ARCANE_IRESSOURCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IXmlDocumentHolder;
class IApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un gestionnaire de ressource.
 *
 */
class ARCANE_CORE_EXPORT IRessourceMng
{
 public:
	
  //! Création d'un gestionnaire d'historique par défaut.
  static IRessourceMng* createDefault(IApplication*);

 public:

  virtual ~IRessourceMng(){} //!< Libère les ressources

 public:

  /*! \brief Créé un noeud document XML.
   * Crée et retourne un document XML utilisant une implémentation par défaut.
   * La destruction de ce document invalide tous les noeuds qui en dépendent.
   */
  virtual IXmlDocumentHolder* createXmlDocument() =0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

