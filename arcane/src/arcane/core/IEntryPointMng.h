// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IEntryPointMng.h                                            (C) 2000-2023 */
/*                                                                           */
/* Interface for the entry point manager.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IENTRYPOINTMNG_H
#define ARCANE_CORE_IENTRYPOINTMNG_H
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
 * \internal
 * \brief Interface for the entry point manager.
 */
class IEntryPointMng
{
 public:

  virtual ~IEntryPointMng() = default; //!< Frees resources.

 public:

  //! Adds an entry point to the manager
  virtual void addEntryPoint(IEntryPoint*) = 0;

  /*!
   * \brief Entry point by name \a s.
   *
   * Returns \a nullptr if the entry point is not found
   */
  virtual IEntryPoint* findEntryPoint(const String& s) = 0;

  /*!
   * \brief Entry point by name \a s from module name \a module_name.
   *
   * Returns \a nullptr if the entry point is not found
   */
  virtual IEntryPoint* findEntryPoint(const String& module_name, const String& s) = 0;

  //! Displays the list of entry points of the manager in \o
  virtual void dumpList(std::ostream& o) = 0;

  //! List of entry points
  virtual EntryPointCollection entryPoints() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
