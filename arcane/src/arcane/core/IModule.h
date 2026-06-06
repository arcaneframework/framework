// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModule.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Interface of the Module class.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMODULE_H
#define ARCANE_CORE_IMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableRef;
class IParallelMng;
class CaseOptionsMain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a module.
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT IModule
{
 public:

  //! Destructor
  virtual ~IModule() {}

 public:

  //! Module name
  virtual String name() const =0;

  //! Module version
  virtual VersionInfo versionInfo() const =0;

 public:
	
  //! Module session
  virtual ISession* session() const =0;

  //! Sub-domain manager.
  virtual ISubDomain* subDomain() const =0;

  //! Associated mesh. Can be null. Use defaultMeshHandle() instead
  virtual IMesh* defaultMesh() const =0;

  //! Associated mesh
  virtual MeshHandle defaultMeshHandle() const =0;

  //! Message passing parallelism manager
  virtual IParallelMng* parallelMng() const =0;

 //! Accelerator manager
  virtual IAcceleratorMng* acceleratorMng() const =0;

  //! Trace manager.
  virtual ITraceMng* traceMng() const =0;

 public:
  
  /*!
   * \brief Indicates whether a module is used or not (internal).
   *
   * A module is used if and only if at least one of its
   * entry points is used in the time loop.
   */
  virtual void setUsed(bool v) =0;

  //! \a true if the module is used.
  virtual bool used() const =0;

  /*!
   * \brief Temporarily activates or deactivates the module (internal).
   *
   * When a module is disabled, its calculation loop entry points are
   * no longer called (but others like initialization or termination
   * ones still are).
   */
  virtual void setDisabled(bool v) =0;

  //! \a true if the module is disabled
  virtual bool disabled() const =0;

  /*! \internal
   * \brief Indicates if the module is managed by a garbage collector,
   * in which case the delete operator should not be called on it.
   */
  virtual bool isGarbageCollected() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
