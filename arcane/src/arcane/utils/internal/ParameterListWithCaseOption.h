// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterListWithCaseOption.h                               (C) 2000-2025 */
/*                                                                           */
/* Liste de paramètres avec support pour les options du jeu de données.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_PARAMETERLISTWITHCASEOPTION_H
#define ARCANE_UTILS_INTERNAL_PARAMETERLISTWITHCASEOPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ParameterCaseOption.h"
#include "arcane/utils/ParameterList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de paramètres avec informations pour surcharger les options
 * du jeu de données.
 */
class ARCANE_UTILS_EXPORT ParameterListWithCaseOption
: private ParameterList
{
 public:

  using ParameterList::addParameterLine;
  using ParameterList::getParameterOrNull;

  /*!
   * \brief Méthode permettant de récupérer un objet de type ParameterCaseOption.
   *
   * Cet objet peut être détruit après utilisation.
   *
   * \param language Le langage dans lequel est écrit le jeu de données.
   * \return Un objet de type ParameterCaseOption.
   */
  ParameterCaseOption getParameterCaseOption(const String& language) const
  {
    return _getParameterCaseOption(language);
  }

  //! Ajoute les paramètres de \a parameters aux paramètres de l'instance
  void addParameters(const ParameterList& parameters);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
