// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleProperty.h                                            (C) 2000-2018 */
/*                                                                           */
/* Module properties.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MODULEPROPERTY_H
#define ARCANE_MODULEPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Module creation properties.
 *
 * This class is used in module registration macros
 * and can therefore be instantiated as a global variable before entering
 * the code's main(). It must therefore only contain Plain Object Data (POD) fields.
 *
 * Generally, instances of this class are used when
 * registering a service via the ARCANE_REGISTER_MODULES() macro.
 */
class ARCANE_CORE_EXPORT ModuleProperty
{
 public:

  /*!
   * \brief Constructs an instance for a module named \a aname.
   */
  ModuleProperty(const char* aname, bool is_autoload) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_is_autoload(is_autoload)
  {
  }

  /*!
   * \brief Constructs an instance for a module named \a aname.
   */
  explicit ModuleProperty(const char* aname) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_is_autoload(false)
  {
  }

 public:

  //! Module name.
  const char* name() const { return m_name; }

  //! Indicates if the module is automatically loaded.
  bool isAutoload() const { return m_is_autoload; }

 private:

  const char* m_name;
  bool m_is_autoload;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
