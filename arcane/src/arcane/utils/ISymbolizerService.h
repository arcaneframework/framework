// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISymbolizerService.h                                        (C) 2000-2018 */
/*                                                                           */
/* Interface d'un service de récupération des symboles du code source.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ISYMBOLIZERSERVICE_H
#define ARCANE_UTILS_ISYMBOLIZERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
class ARCANE_UTILS_EXPORT ISymbolizerService
{
 public:

  virtual ~ISymbolizerService() {} //<! Libère les ressources

 public:

  //! Informations pour la pile d'appel \a frames.
  // TODO TODO RENOMMER CETTE METHODE
  virtual String stackTrace(ConstArrayView<StackFrame> frames) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

