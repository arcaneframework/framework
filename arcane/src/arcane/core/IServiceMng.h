// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceMng.h                                               (C) 2000-2025 */
/*                                                                           */
/* Service manager interface.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEMNG_H
#define ARCANE_CORE_ISERVICEMNG_H
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
 * \brief Service manager interface.
 */
class IServiceMng
{
 public:

  virtual ~IServiceMng() = default; //!< Releases resources.

 public:

  //! Associated trace manager
  virtual ITraceMng* traceMng() const = 0;

  //! Adds a reference to the service \a sv
  virtual void addSingletonInstance(SingletonServiceInstanceRef sv) = 0;

  //! Returns the list of singleton services
  virtual SingletonServiceInstanceCollection singletonServices() const = 0;

  /*!
   * Singleton service named \a name.
   *
   * Returns a null reference if no instance of name \a name exists.
   */
  virtual SingletonServiceInstanceRef singletonServiceReference(const String& name) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
