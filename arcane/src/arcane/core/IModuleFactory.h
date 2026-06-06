// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModuleFactory.h                                            (C) 2000-2019 */
/*                                                                           */
/* Module manufacturing interface.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMODULEFACTORY_H
#define ARCANE_IMODULEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"
#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class MeshHandle;
class IModuleFactory2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about a module factory.
 *
 * This interface contains the necessary information about a module
 * factory.
 *
 * The module can be created directly via the createModule() method.
 *
 * This class uses a reference counter to manage its lifetime
 * (see the ReferenceCounter class).
 */
class ARCANE_CORE_EXPORT IModuleFactoryInfo
{
 protected:

  //! Releases resources
  virtual ~IModuleFactoryInfo() {}

 public:

  virtual void addReference() = 0;
  virtual void removeReference() = 0;
  /*!
   * \brief Indicates if the module should be loaded automatically.
   *
   * If this property is true, the module will always be loaded even
   * if it does not appear in the time loop.
   */
  virtual bool isAutoload() const = 0;

  /*!
   * \brief If the factory is a one-to-one module,
   * initializes it on the sub-domain \a sub_domain.
   *
   * This method is called when the sub-domain is created, to
   * perform specific module initializations before
   * it is manufactured. For example, to add time loops
   * specific to the module.
   */
  virtual void initializeModuleFactory(ISubDomain* sub_domain) = 0;

  /*!
   * \brief Creates a module.
   *
   * The implementation must call parent->moduleMng()->addModule()
   * for the created module.
   *
   * \param parent Parent of this module. 
   * \param mesh mesh associated with the module.
   * \return the created module
   */
  virtual Ref<IModule> createModule(ISubDomain* parent, const MeshHandle& mesh_handle) = 0;

  //! Name of the module created by this factory.
  virtual String moduleName() const = 0;

  /*!
   * \brief Information about the module that can be created by this
   * factory.
   *
   * The returned instance remains the property of the application that
   * created it and must neither be modified nor destroyed.
   */
  virtual const IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Module factory interface (V2).
 *
 * This interface is reserved for IModuleFactoryInfo and should not
 * be used directly.
 */
class ARCANE_CORE_EXPORT IModuleFactory2
{
 public:

  virtual ~IModuleFactory2() {}

 public:

  /*!
   * \brief Creates a module instance.
   *
   * \param sd associated sub-domain.
   * \param mesh mesh associated with the module.
   * \return the created module
   */
  virtual Ref<IModule> createModuleInstance(ISubDomain* sd, const MeshHandle& mesh_handle) = 0;

  /*!
   * \brief Static initialization of the module.
   *
   * This method is called when the sub-domain is created, to
   * perform specific module initializations before
   * it is manufactured. For example, to add time loops
   * specific to the module.
   */
  virtual void initializeModuleFactory(ISubDomain* sd) = 0;

  //! Name of the module created by this factory.
  virtual String moduleName() const = 0;

  //! Information about the module that can be created by this factory.
  virtual const IServiceInfo* serviceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reference counter for a module factory.
 */
class ARCANE_CORE_EXPORT ModuleFactoryReference
: ReferenceCounter<IModuleFactoryInfo>
{
 public:

  typedef ReferenceCounter<IModuleFactoryInfo> Base;

 public:

  explicit ModuleFactoryReference(IModuleFactoryInfo* f)
  : Base(f)
  {}
  ModuleFactoryReference(Ref<IModuleFactory2> factory, bool is_autoload);

 public:

  IModuleFactoryInfo* factory() const { return get(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
