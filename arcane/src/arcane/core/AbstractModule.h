// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractModule.h                                            (C) 2000-2025 */
/*                                                                           */
/* Abstract base class of a module.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTMODULE_H
#define ARCANE_CORE_ABSTRACTMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IModule.h"
#include "arcane/core/ModuleBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ModuleBuildInfo;
typedef ModuleBuildInfo ModuleBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class representing a module.
 *
 * This class is THE low-level implementation class of the \a IModule interface.
 *
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT AbstractModule
: public TraceAccessor
, public IModule
{
 public:

  //! Constructor from a \a ModuleBuildInfo
  AbstractModule(const ModuleBuildInfo&);

 public:

  //! Destructor
  virtual ~AbstractModule();

 public:

  //! Module version
  VersionInfo versionInfo() const override { return m_version_info; }

 public:

  /*! \brief Initialization of the module for the sub-domain \a sd.
   *
   * This static method can be redefined in a derived class
   * to perform initializations for the sub-domain \a sd
   * even if the module is not used.
   *
   * A common use is registering entry points
   * for modules without .axl
   *
   * This method will be called during the sub-domain creation phase
   * on all Modules (even unused ones).
   */
  static void staticInitialize(ISubDomain* sd) { ARCANE_UNUSED(sd); }

 public:

  //! Module name
  String name() const override { return m_name; }
  //! Session associated with the module
  ISession* session() const override { return m_session; }
  //! Sub-domain associated with the module
  ISubDomain* subDomain() const override { return m_sub_domain; }
  //! Default mesh for this module
  IMesh* defaultMesh() const override { return m_default_mesh_handle.mesh(); }
  //! Default mesh for this module
  MeshHandle defaultMeshHandle() const override { return m_default_mesh_handle; }
  //! Message passing parallelism manager
  IParallelMng* parallelMng() const override;
  //! Accelerator manager.
  IAcceleratorMng* acceleratorMng() const override;
  //! Trace manager
  ITraceMng* traceMng() const override;
  //! Sets the module usage flag
  void setUsed(bool v) override { m_used = v; }
  //! Returns the module usage status
  bool used() const override { return m_used; }
  //! Sets the module activation flag
  void setDisabled(bool v) override { m_disabled = v; }
  //! Returns the module activation status
  bool disabled() const override { return m_disabled; }
  //! Indicates if the module uses a Garbage collection system
  /*! 
   *  <ul >
   *  <li >if \a true, indicates destruction by a Garbage collector and not explicit destruction</li>
   *  <li >if \a false, this module will be destroyed explicitly by calling its destructor</li>
   *  </ul >
   *
   * The Garbage collection system is usually activated for
   * modules resulting from a C# implementation. Classic modules
   * in C++ do not have this mechanism.
   *
   * \todo Check in ModuleMng::removeModule the use of
   * this indication. A call to the Deleter as in
   * ModuleMng::removeAllModules might be necessary.
   */
  bool isGarbageCollected() const override { return false; }

 protected:

  void _setVersionInfo(const VersionInfo& vi)
  {
    m_version_info = vi;
  }

 private:

  ISession* m_session; //!< Session
  ISubDomain* m_sub_domain; //!< sub-domain
  MeshHandle m_default_mesh_handle; //!< Default mesh of the module
  String m_name; //!< Module name
  bool m_used; //!< \a true if the module is used
  bool m_disabled; //!< Module activation status
  VersionInfo m_version_info; //!< Module version
  IAcceleratorMng* m_accelerator_mng; //!< Accelerator manager
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
