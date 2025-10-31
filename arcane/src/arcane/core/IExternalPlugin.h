// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExternalPlugin.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface du service de plugin externes.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IEXTERNALPLUGIN_H
#define ARCANE_CORE_IEXTERNALPLUGIN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// WARNING: Experimental API. Do not use outside of Arcane

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du service de chargement de services externes.
 * \warning Cette interface est expérimentale.
 *
 * Il faut appeler loadFile() (éventuellement avec une chaîne de caractères vide)
 * pour initialiser l'instance.
 */
class ARCANE_CORE_EXPORT IExternalPlugin
{
 public:

  //! Libère les ressources
  virtual ~IExternalPlugin() = default;

 public:

 /*!
  * \brief Charge et exécute un fichier contenant un script externe.
  *
  * \a filename peut-être nul, auquel cas on ne fait que initialiser l'instance.
  */
  virtual void loadFile(const String& filename) = 0;

  /*!
   * \brief Exécute la fonction \a function_name.
   *
   * Il faut avoir chargé un script contenant cette fonction (via loadFile())
   * avant d'appeler cette méthode. La méthode \a function_name ne doit pas
   * avoir d'arguments.
   */
  virtual void executeFunction(const String& function_name) = 0;

  /*!
   * \brief Exécute la fonction \a function_name avec un contexte
   *
   * Il faut avoir chargé un script contenant cette fonction (via loadFile())
   * avant d'appeler cette méthode. La méthode spécifiée doit prendre en
   * argument une instance de PythonSubDomainContext.
   */
  virtual void executeContextFunction(const String& function_name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
