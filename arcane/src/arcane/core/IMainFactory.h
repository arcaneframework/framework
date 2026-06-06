// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMainFactory.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface of Arcane's AbstractFactory.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMAINFACTORY_H
#define ARCANE_CORE_IMAINFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IBase;
class ISubDomain;
class ApplicationInfo;
class IArcaneMain;
class IParallelSuperMng;
class IApplication;
class IRegistry;
class IVariableMng;
class IModuleMng;
class IEntryPointMng;
class ITimeHistoryMng;
class ICaseMng;
class ICaseDocument;
class ITimerMng;
class ITimeLoopMng;
class ITimeLoop;
class IIOMng;
class IServiceMng;
class IServiceLoader;
class IXmlDocumentHolder;
class IMesh;
class IDataFactory;
class ITimeStats;
class IParallelMng;
class ItemGroup;
class IPrimaryMesh;
class ITraceMngPolicy;
class IModuleMaster;
class ILoadBalanceMng;
class ICheckpointMng;
class IPropertyMng;
class IDataFactoryMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Factory for Arcane classes.
 *
 It is a virtual class containing methods to manufacture
 the different instances of the architecture managers
 (Design Pattern: AbstractFactory).

 Arcane provides default factories for most managers
 (IApplication, IParallelSuperMng, ...). However, the class managing the code must
 be specified by implementing the createArcaneMain() method in a
 derived class.

 The general entry point of the code is achieved by calling the function
 arcaneMain().

 For example, if we define a class <tt>ConcreteMainFactory</tt> that
 derives from IMainFactory, we run the code as follows:
 
 * \code
 * int
 * main(int argc,char** argv)
 * {
 *   ApplicationInfo exe_info = ... // Creation of executable info.
 *   ConcreteMainFactory cmf; // Creation of the factory
 *   return IMainFactory::arcaneMain(exe_info,&cmf);
 * }
 * \endcode
 */
class IMainFactory
{
 public:

  virtual ~IMainFactory() {} //!< Releases resources.

 public:

 public:

  //! Creates an instance of IArcaneMain
  virtual IArcaneMain* createArcaneMain(const ApplicationInfo& app_info) =0;

 public:

  //! Creates an instance of a supervisor
  virtual IApplication* createApplication(IArcaneMain*) =0;

  //! Creates an instance of the variable manager
  virtual IVariableMng* createVariableMng(ISubDomain*) =0;

  //! Creates an instance of the module manager
  virtual IModuleMng* createModuleMng(ISubDomain*) =0;

  //! Creates an instance of the entry point manager
  virtual IEntryPointMng* createEntryPointMng(ISubDomain*) =0;

  //! Creates an instance of the time history manager
  virtual ITimeHistoryMng* createTimeHistoryMng(ISubDomain*) =0;

  //! Creates an instance of the case manager
  virtual ICaseMng* createCaseMng(ISubDomain*) =0;

  //! Creates an instance of a case document
  virtual ICaseDocument* createCaseDocument(IApplication*) =0;

  //! Creates an instance of a case document for a given language \a lang
  virtual ICaseDocument* createCaseDocument(IApplication*,const String& lang) =0;

  //! Creates an instance of a case document
  virtual ICaseDocument* createCaseDocument(IApplication*,IXmlDocumentHolder* doc) =0;

  /*!
   * \brief Creates an instance of execution time statistics.
   *
   * Use the overloaded createTimeStats(ITimerMng*,ITraceMng*,const String& name).
   */
  virtual ARCANE_DEPRECATED_116 ITimeStats* createTimeStats(ISubDomain*) =0;

  //! Creates an instance of execution time statistics
  virtual ITimeStats* createTimeStats(ITimerMng* tim,ITraceMng* trm,const String& name) =0;

  //! Creates an instance of the time loop manager
  virtual ITimeLoopMng* createTimeLoopMng(ISubDomain*) =0;

  //! Creates a time loop named \a name
  virtual ITimeLoop* createTimeLoop(IApplication* sm,const String& name) =0;

  //! Creates an instance of the I/O manager
  virtual IIOMng* createIOMng(IApplication*) =0;

  //! Creates an instance of the I/O manager for the parallelism manager \a pm
  virtual IIOMng* createIOMng(IParallelMng* pm) =0;

  //! Creates an instance of the service loader
  virtual IServiceLoader* createServiceLoader() =0;

  //! Creates an instance of the service manager
  virtual IServiceMng* createServiceMng(IBase*) =0;

  //! Creates an instance of the checkpoint manager
  virtual ICheckpointMng* createCheckpointMng(ISubDomain*) =0;

  //! Creates an instance of the property manager
  ARCCORE_DEPRECATED_2020("Use createPropertyMngReference() instead")
  virtual IPropertyMng* createPropertyMng(ISubDomain*) =0;

  //! Creates an instance of the property manager
  virtual Ref<IPropertyMng> createPropertyMngReference(ISubDomain*) =0;

  /*!
   * \brief Creates or retrieves a mesh.
   *
   * Creates or retrieves a mesh named \a name for the sub-domain \a sub_domain.
   *
   * If the sub-domain already has a mesh with the name \a name,
   * the latter is returned.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name) =0;

  /*!
   * \brief Creates or retrieves a mesh.
   *
   * Creates or retrieves a mesh named \a name for the sub-domain \a sub_domain.
   *
   * If the sub-domain already has a mesh with the name \a name,
   * the latter is returned.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use createMesh(..., eMeshAMRKind amr_type) instead")
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name,bool is_amr) =0;

  /*!
   * \brief Creates or retrieves a mesh.
   *
   * Creates or retrieves a mesh named \a name for the sub-domain \a sub_domain.
   *
   * If the sub-domain already has a mesh with the name \a name,
   * the latter is returned.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name,eMeshAMRKind amr_type) =0;

  /*!
   * \brief Creates or retrieves a mesh.
   *
   * Creates or retrieves a mesh named \a name for the sub-domain \a sub_domain
   * associated with the parallelism manager \a pm. If the sub-domain already has
   * a mesh with the name \a name, the latter is returned.
   *
   * The parallelism manager must be the same as that of the sub-domain
   * or derived from it.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain, IParallelMng* pm,
                                   const String& name) =0;

  /*!
   * \brief Creates or retrieves a mesh.
   *
   * Creates or retrieves a mesh named \a name for the sub-domain \a sub_domain
   * associated with the parallelism manager \a pm. If the sub-domain already has
   * a mesh with the name \a name, the latter is returned.
   *
   * The parallelism manager must be the same as that of the sub-domain
   * or derived from it.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use createMesh(..., eMeshAMRKind amr_type) instead")
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain, IParallelMng* pm,
                                   const String& name, bool is_amr) =0;

  /*!
   * \brief Creates or retrieves a mesh.
   *
   * Creates or retrieves a mesh named \a name for the sub-domain \a sub_domain
   * associated with the parallelism manager \a pm. If the sub-domain already has
   * a mesh with the name \a name, the latter is returned.
   *
   * The parallelism manager must be the same as that of the sub-domain
   * or derived from it.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain, IParallelMng* pm,
                                   const String& name, eMeshAMRKind amr_type) =0;

  /*!
   * \brief Creates a sub-mesh for the mesh \a mesh, named \a name.
   *
   * The sub-mesh is initialized with the items of the group \a group.
   * Currently, this group cannot be a complete group (isAllItems())
   * nor a calculated group (if not incremental).
   */
  virtual IMesh* createSubMesh(IMesh* mesh, const ItemGroup& group, const String& name) =0;

  //! Creates a factory for data
  ARCCORE_DEPRECATED_2020("Use createDataFactoryMngRef() instead")
  virtual IDataFactory* createDataFactory(IApplication*) =0;

  //! Creates a factory manager for data
  virtual Ref<IDataFactoryMng> createDataFactoryMngRef(IApplication*) =0;

  //! Creates a manager for accelerators
  virtual Ref<IAcceleratorMng> createAcceleratorMngRef(ITraceMng* tm) =0;

  /*!
   * \brief Creates a trace manager.
   *
   * The returned instance must be initialized via an ITraceMngPolicy.
   */
  virtual ITraceMng* createTraceMng() =0;

  /*!
   * \brief Creates a configuration manager for a trace manager.
   */
  virtual ITraceMngPolicy* createTraceMngPolicy(IApplication* app) =0;

  /*!
   * \brief Creates the master module for the sub-domain \a sd.
   */
  virtual IModuleMaster* createModuleMaster(ISubDomain* sd) =0;

  /*!
   * \brief Creates a description manager for load balancing.
   */
  virtual ILoadBalanceMng* createLoadBalanceMng(ISubDomain* sd) =0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
