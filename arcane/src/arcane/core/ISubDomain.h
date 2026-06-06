// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISubDomain.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a subdomain.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISUBDOMAIN_H
#define ARCANE_CORE_ISUBDOMAIN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/core/IBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableMng;
class IModuleMng;
class IServiceMng;
class IEntryPointMng;
class IModule;
class IMeshIOService;
class IMesh;
class IMeshMng;
class ApplicationInfo;
class IIOMng;
class ITimeLoopMng;
class CaseOptionsMain;
class IParallelMng;
class IThreadMng;
class IDirectory;
class ITimeHistoryMng;
class ICaseMng;
class IInterfaceMng;
class ITimerMng;
class ITimeStats;
class IRessourceMng;
class CommonVariables;
class IMainFactory;
class ICaseDocument;
class XmlNode;
class IMemoryInfo;
class IObservable;
class IInitialPartitioner;
class IDirectExecution;
class IPhysicalUnitSystem;
class ILoadBalanceMng;
class IModuleMaster;
class ICheckpointMng;
class IPropertyMng;
class IConfiguration;
class MeshHandle;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the subdomain manager.
 */
class ARCANE_CORE_EXPORT ISubDomain
: public IBase
{
 protected:

  virtual ~ISubDomain() {} //!< Frees resources.

 public:

  virtual void destroy() = 0;

 public:

  //! Main factory.
  virtual IMainFactory* mainFactory() = 0;

  //! Session
  virtual ISession* session() const = 0;

  //! Application
  virtual IApplication* application() = 0;

  //! Returns the variable manager
  virtual IVariableMng* variableMng() = 0;

  //! Returns the module manager
  virtual IModuleMng* moduleMng() = 0;

  //! Returns the entry point manager
  virtual IEntryPointMng* entryPointMng() = 0;

  //! Returns the parallelism manager
  virtual IParallelMng* parallelMng() = 0;

  /*!
   * \brief Returns the parallelism manager for all replicas.
   *
   * Generally, parallelMng() must be used. This manager
   * is essentially used to perform operations on all
   * subdomains and their replicas. If there is no replication,
   * this manager is the same as parallelMng().
   */
  virtual IParallelMng* allReplicaParallelMng() const = 0;

  //! Returns the thread manager
  virtual IThreadMng* threadMng() = 0;

  //! Returns the history manager
  virtual ITimeHistoryMng* timeHistoryMng() = 0;

  //! Returns the time loop manager
  virtual ITimeLoopMng* timeLoopMng() = 0;

  //! Returns the I/O manager.
  virtual IIOMng* ioMng() = 0;

  //! Returns the dataset manager.
  virtual ICaseMng* caseMng() = 0;

  //! Returns the timer manager
  virtual ITimerMng* timerMng() const = 0;

  //! Protection manager
  virtual ICheckpointMng* checkpointMng() const = 0;

  //! Property manager
  virtual IPropertyMng* propertyMng() const = 0;

  //! Execution time statistics
  virtual ITimeStats* timeStats() const = 0;

  //! Memory information manager
  virtual IMemoryInfo* memoryInfo() const = 0;

  //! Subdomain unit system.
  virtual IPhysicalUnitSystem* physicalUnitSystem() = 0;

  //! Returns the load balancing manager.
  virtual ILoadBalanceMng* loadBalanceMng() = 0;

  //! Returns the mesh manager.
  virtual IMeshMng* meshMng() const = 0;

  //! Module master interface.
  virtual IModuleMaster* moduleMaster() const = 0;

  //! Associated configuration.
  virtual const IConfiguration* configuration() const = 0;

  //! Associated configuration.
  virtual IConfiguration* configuration() = 0;

  //! Associated accelerator manager
  virtual IAcceleratorMng* acceleratorMng() = 0;

 public:

  //! Subdomain ID associated with this manager.
  virtual Int32 subDomainId() const = 0;

  //! Total number of subdomains
  virtual Int32 nbSubDomain() const = 0;

  //! Reads the mesh information from the dataset
  virtual void readCaseMeshes() = 0;

  /*!
   * \internal
   * \brief Sets a flag indicating that a
   * restart is being performed.
   *
   * This method must be called before allocating the mesh (allocateMeshes()).
   */
  virtual void setIsContinue() = 0;

  //! True if a restart is being performed, false otherwise.
  virtual bool isContinue() const = 0;

  /*!
   * \internal
   * \brief Allocates the instances.
   *
   * Mesh instances are simply allocated but do not contain entities.
   * This method must be called before any other operation involving the mesh,
   * especially before reading dataset options or reading protections.
   */
  virtual void allocateMeshes() = 0;

  /*!
   * \internal
   * \brief Reads or re-reads the meshes.
   *
   * At startup, the meshes are re-read from the dataset information.
   * During restart, the meshes are loaded from a protection.
   * This method must be called after calling allocateMeshes().
   */
  virtual void readOrReloadMeshes() = 0;

  /*!
   * \internal
   * \brief Initializes variables whose values are specified in
   * the dataset.
   */
  virtual void initializeMeshVariablesFromCaseFile() = 0;

  /*!
   * \internal
   * \brief Applies the initialization mesh partitioning.
   */
  virtual void doInitMeshPartition() = 0;

  //! Adds a mesh to the subdomain
  ARCCORE_DEPRECATED_2020("Use meshMng()->meshFactoryMng() to create and add mesh")
  virtual void addMesh(IMesh* mesh) = 0;

  //! List of meshes in the subdomain
  virtual ConstArrayView<IMesh*> meshes() const = 0;

  /*!
   * \internal
   * \brief Executes initialization modules
   * \deprecated This method does nothing anymore.
   */
  virtual ARCANE_DEPRECATED_2018 void doInitModules() = 0;

  //! Executes exit modules
  virtual void doExitModules() = 0;

  //! Displays information about the instance
  virtual void dumpInfo(std::ostream&) = 0;

  /*!
   * \brief Default mesh.
   *
   * The default mesh does not exist until the dataset
   * has been read. It is generally preferable
   * to use defautMeshHandle() instead.
   */
  virtual IMesh* defaultMesh() = 0;

  /*!
   * \brief Handle for the default mesh.
   *
   * This handle always exists even if the associated mesh has not
   * yet been created.
   */
  virtual const MeshHandle& defaultMeshHandle() = 0;

  virtual ARCANE_DEPRECATED IMesh* mesh() = 0;

  /*! \brief Searches for the mesh named \a name.
   *
   If the mesh is not found, the method throws an exception
   if \a throw_exception is \a true or returns 0 if \a throw_exception
   is \a false.
   */
  ARCCORE_DEPRECATED_2019("Use meshMng()->findMeshHandle() instead")
  virtual IMesh* findMesh(const String& name, bool throw_exception = true) = 0;

  //! Indicates if the session has been initialized.
  virtual bool isInitialized() const = 0;

  /*!
   * \internal
   * \brief Indicates that the subdomain is initialized.
   */
  virtual void setIsInitialized() = 0;

  //! Executable information
  virtual const ApplicationInfo& applicationInfo() const = 0;

  //! Case XML document.
  virtual ICaseDocument* caseDocument() = 0;

  /*!
   * \brief Checks if an identifier is valid
   *
   \exception ExceptionBadName if \a id is not a valid identifier.
   */
  virtual void checkId(const String& where, const String& id) = 0;

  //! Full file path of the dataset
  virtual const String& caseFullFileName() const = 0;

  //! Case name
  virtual const String& caseName() const = 0;

  //! Fills \a bytes with the dataset content.
  virtual void fillCaseBytes(ByteArray& bytes) const = 0;

  /*! \brief Sets the case name.
   *
   This method must be called before initialization.
  */
  virtual void setCaseName(const String& name) = 0;

  /*!
   * \brief Sets the initial partitioner.
   *
   If this method is not called, the default partitioner
   is used.
   *
   This method must be called before module initialization,
   for example in construction entry points.
   *
   The instance takes ownership of \a partitioner and will destroy it by delete
   at the end of the calculation.
   */
  virtual void setInitialPartitioner(IInitialPartitioner* partitioner) = 0;

  //! General dataset options.
  virtual const CaseOptionsMain* caseOptionsMain() const = 0;

  //! Base directory for exports.
  virtual const IDirectory& exportDirectory() const = 0;

  /*! \brief Sets the output path for exports (protections and restarts)

   The directory corresponding to \a dir must exist.

   This method must be called before initialization.
  */
  virtual void setExportDirectory(const IDirectory& dir) = 0;

  //! Base directory for exports requiring archiving.
  virtual const IDirectory& storageDirectory() const = 0;

  /*! \brief Sets the output path for exports requiring archiving.

    This directory allows specifying a directory that can be automatically archived.
    If it is null, exportDirectory() is used.

    This method must be called before initialization.
  */
  virtual void setStorageDirectory(const IDirectory& dir) = 0;

  //! Base directory for listings (logs, execution info).
  virtual const IDirectory& listingDirectory() const = 0;

  /*! \brief Sets the output path for listing info
   *
   The directory corresponding to \a dirname must exist.

   This method must be called before initialization.
  */
  virtual void setListingDirectory(const IDirectory& dir) = 0;

  //! Information on standard variables
  virtual const CommonVariables& commonVariables() const = 0;

  /*!
   * \brief Dumps internal architecture information.
   * The information is stored in an XML tree with \a root as the root element.
   * This information is for internal use by Arcane.
   */
  virtual void dumpInternalInfos(XmlNode& elem) = 0;

  /*! \brief Mesh dimension (1D, 2D, or 3D).
   *
   * \deprecated Use mesh()->dimension() instead.
   */
  virtual Integer ARCANE_DEPRECATED meshDimension() const = 0;

  /*!
   * \brief Notification before subdomain destruction
   */
  virtual IObservable* onDestroyObservable() = 0;

  //! Direct execution service (or null)
  virtual IDirectExecution* directExecution() const = 0;

  /*!
   * \brief Sets the direct execution service.
   *
   This service must be set during service creation when
   reading the dataset.
   */
  virtual void setDirectExecution(IDirectExecution* v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
