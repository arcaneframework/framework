// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExternalPlugin.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface for external plugin service.                                    */
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
 * \brief Interface for external service loading.
 * \warning This interface is experimental.
 *
 * You must call loadFile() (possibly with an empty string)
 * to initialize the instance.
 */
class ARCANE_CORE_EXPORT IExternalPlugin
{
 public:

  //! Releases resources
  virtual ~IExternalPlugin() = default;

 public:

 /*!
  * \brief Loads and executes a file containing an external script.
  *
  * \a filename may be null, in which case only the instance is initialized.
  */
  virtual void loadFile(const String& filename) = 0;

  /*!
   * \brief Executes the function \a function_name.
   *
   * You must have loaded a script containing this function (via loadFile())
   * before calling this method. The method \a function_name must not
   * have arguments.
   */
  virtual void executeFunction(const String& function_name) = 0;

  /*!
   * \brief Executes the function \a function_name with a context
   *
   * You must have loaded a script containing this function (via loadFile())
   * before calling this method. The specified method must take an
   * instance of PythonSubDomainContext as an argument.
   */
  virtual void executeContextFunction(const String& function_name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
