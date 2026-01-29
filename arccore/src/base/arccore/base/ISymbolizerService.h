// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISymbolizerService.h                                        (C) 2000-2026 */
/*                                                                           */
/* Interface d'un service de récupération des symboles du code source.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ISYMBOLIZERSERVICE_H
#define ARCCORE_BASE_ISYMBOLIZERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service de récupération des symboles du code source.
 *
 * Ce service permet de récupérer certaines informations du code source
 * à partir d'une adresse mémoire. Parmi les informations récupérables
 * il y a le nom du fichier, le nom de la méthode et les numéros de ligne.
 *
 * \warning UNSTABLE API
 */
class ARCCORE_BASE_EXPORT ISymbolizerService
{
 public:

  virtual ~ISymbolizerService() {} //<! Libère les ressources

 public:

  //! Informations pour la pile d'appel \a frames.
  // TODO TODO RENOMMER CETTE METHODE
  virtual String stackTrace(ConstArrayView<StackFrame> frames) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
