// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModuleMng.h                                                (C) 2000-2025 */
/*                                                                           */
/* Module manager interface.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMODULEMNG_H
#define ARCANE_CORE_IMODULEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IModule;
class Msg;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Module manager interface.
 * \ingroup Module
 */
class IModuleMng
{
 public:

  //! Destructor
  /*! Frees resources */
  virtual ~IModuleMng() {}

 public:

  //! Adds module \a m to the manager
  virtual void addModule(Ref<IModule> m) = 0;

  //! Removes module \a m
  virtual void removeModule(Ref<IModule> m) = 0;

  //! Prints the list of modules in the manager to a stream \a o
  virtual void dumpList(std::ostream& o) = 0;

  //! List of modules
  virtual ModuleCollection modules() const = 0;

  //! Removes and destroys modules managed by this manager
  virtual void removeAllModules() = 0;

  //! Indicates if the module named \a name is active
  /*!
   * If no module named \a name exists, returns false.
   */
  virtual bool isModuleActive(const String& name) = 0;

  //! Returns the instance of the module named \a name.
  /*!
   * If no module named \a name exists, returns 0.
   */
  virtual IModule* findModule(const String& name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
