// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExternalPlugin.h                                           (C) 2000-2024 */
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
 */
class ARCANE_CORE_EXPORT IExternalPlugin
{
 public:

  //! Libère les ressources
  virtual ~IExternalPlugin() = default;

 public:

  //! Charge et exécute un fichier contenant un script externe
  virtual void loadFile(const String& filename) = 0;

  /*!
   * \brief Exécute la fonction \a function_name.
   *
   * Il faut avoir chargé un script contenant cette fonction (via loadFile())
   * avant d'appeler cette méthode.
   */
  virtual void executeFunction(const String& function_name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
