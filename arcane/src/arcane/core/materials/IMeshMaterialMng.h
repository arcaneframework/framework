// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialMng.h                                          (C) 2000-2024 */
/*                                                                           */
/* Interface for the material manager of a mesh.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALMNG_H
#define ARCANE_MATERIALS_IMESHMATERIALMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctorWithArgument.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Interface for the material and environment manager of a mesh.
 *
 * This interface manages the different components (IMeshComponent)
 * multi-materials of a mesh, as well as their associated variables.
 * These components can be either materials (IMeshMaterial),
 * or environments (IMeshEnvironment). It is possible to retrieve the list
 * of materials via materials() and the list of environments via environments().
 * It is also possible to retrieve one of these two lists in the form
 * 
 * The current implementation only manages materials and environments on the cells.
 *
 * Once this instance is created, via getReference(), the first thing to do is
 * register
 * the list of materials via registerMaterialInfo(). It is then
 * possible to create each environment by indicating the list of materials
 * that compose it via createEnvironment(). Once this is finished, you
 * must call endCreate() to finish initialization. The list of materials and environments
 * cannot be modified except during initialization. It must not change afterward.
 *
 * Any modification of the cell list of an environment or a material
 * must be done via an instance of MeshMaterialModifier.
 */
class ARCANE_CORE_EXPORT IMeshMaterialMng
{
  friend class MeshMaterialMngFactory;

 public:

  virtual ~IMeshMaterialMng() = default;

 public:

  /*!
   * \brief Retrieves or creates the reference associated with \a mesh.
   *
   * If no material manager is associated with \a mesh, it
   * will be created when this method is called if \a create is \a true.
   * If \a create is \a false, no manager is associated
   * with the mesh, a null pointer is returned.
   * The returned instance remains valid as long as the mesh \a mesh exists.
   */
  static IMeshMaterialMng* getReference(const MeshHandleOrMesh& mesh_handle, bool create = true);

  /*!
   * \brief Retrieves or creates the reference associated with \a mesh.
   *
   * If no material manager is associated with \a mesh, it
   * will be created when this method is called if \a create is \a true.
   * If \a create is \a false, no manager is associated
   * with the mesh, a null pointer is returned.
   * The returned instance remains valid as long as the mesh \a mesh exists.
   */
  static Ref<IMeshMaterialMng> getTrueReference(const MeshHandle& mesh_handle, bool create = true);

 public:

  //! Associated mesh.
  virtual IMesh* mesh() = 0;

  //! Trace manager
  virtual ITraceMng* traceMng() = 0;

 public:

  /*!
   * \brief Registers the material info with name \a name.
   *
   * This operation only registers the information of a material.
   * This information is then used when creating the environment
   * via createEnvironment().
   */
  virtual MeshMaterialInfo* registerMaterialInfo(const String& name) = 0;

  /*!
   * \brief Creates an environment with the info \a infos
   *
   * The creation of an environment can only take place during initialization.
   * The materials constituting the environment must have previously been registered via
   * \a registerMaterialInfo(). A material can belong to several environments.
   */
  virtual IMeshEnvironment* createEnvironment(const MeshEnvironmentBuildInfo& infos) = 0;

  /*!
   * \brief Creates a block.
   *
   * Creates a block with the info \a infos.
   *
   * The creation of a block can only take place during initialization,
   * (i.e., before calling endCreate()), but after the creation of environments.
   */
  virtual IMeshBlock* createBlock(const MeshBlockBuildInfo& infos) = 0;

  /*!
   * \brief Adds an environment to an existing block.
   *
   * Adds the environment \a env to the block \a block.
   *
   * The modification of a block can only take place during initialization,
   * (i.e., before calling endCreate()).
   *
   * \warning This method does not modify the block->cells() group, and it is
   * up to the caller to add the environment's cells to the group if necessary.
   */
  virtual void addEnvironmentToBlock(IMeshBlock* block, IMeshEnvironment* env) = 0;

  /*!
   * \brief Removes an environment from an existing block.
   *
   * Removes the environment \a env from the block \a block.
   *
   * The modification of a block can only take place during initialization,
   * (i.e., before calling endCreate()).
   *
   * \warning This method does not modify the block->cells() group, and it is
   * up to the caller to add the environment's cells to the group if necessary.
   */
  virtual void removeEnvironmentToBlock(IMeshBlock* block, IMeshEnvironment* env) = 0;

  /*!
   * \brief Indicates that environment creation is finished.
   *
   * The instance is not usable until this method has been called.
   *
   * If \a is_continue is true, it rebuilds for each material and environment
   * the list of their cells from the recovery information.
   */
  virtual void endCreate(bool is_continue = false) = 0;

  /*!
   * \brief Recreates the material and environment info from the dump info.
   *
   * This method replaces endCreate() and can only be used during recovery
   * and during initialization.
   */
  virtual void recreateFromDump() = 0;

  /*!
   * \brief Sets the saving of values between two modifications of the
   * materials.
   *
   * If active, the values of partial variables are preserved between
   * two modifications of the material list.
   */
  virtual void setKeepValuesAfterChange(bool v) = 0;

  //! Indicates if variable values are preserved between modifications
  virtual bool isKeepValuesAfterChange() const = 0;

  /*!
   * \brief Indicates how to initialize new values in
   * material and environment cells.
   *
   * If true, the new values are initialized to zero or the null vector
   * following the data type. If false, initialization is done with
   * the global value.
   */
  virtual void setDataInitialisationWithZero(bool v) = 0;

  //! Indicates how to initialize new values in material and environment cells.
  virtual bool isDataInitialisationWithZero() const = 0;

  /*!
   * \brief Indicates if environments and materials follow changes
   * in the mesh topology.
   *
   * This method must be called before any material creation.
   *
   * If \a v is \a false, environments and materials are not notified
   * of changes in the mesh topology. In this case, all
   * associated data is invalidated.
   */
  virtual void setMeshModificationNotified(bool v) = 0;

  //! Indicates if environments and materials follow changes in the mesh topology.
  virtual bool isMeshModificationNotified() const = 0;

  /*!
   * \brief Sets the flags to parameterize material/environment modifications.
   *
   * The possible flags are a combination of eModificationFlags.
   *
   * For example:
   \code
   IMeshMaterialMng* mm = ...;
   int flags = (int)eModificationFlags::GenericOptimize | (int)eModificationFlags::OptimizeMultiAddRemove;
   mm->setModificationFlags(flags);
   \endcode
   *
   * This method must be enabled before calling endCreate() to be taken into account.
   */
  virtual void setModificationFlags(int v) = 0;

  //! Flags to parameterize modifications
  virtual int modificationFlags() const = 0;

  /*!
   * \brief Sets the option indicating whether scalar variables
   * of environments are allocated on materials.
   *
   * If active, then environment scalar variables are still allocated
   * on materials. This allows declaring the same variable both
   * as a material variable and an environment variable (e.g., MaterialVariableCellReal and
   * EnvironmentVariableCellReal).
   *
   * By default, this option is not active.
   *
   * This method must be enabled before calling endCreate() to be taken into account.
   */
  virtual void setAllocateScalarEnvironmentVariableAsMaterial(bool v) = 0;

  //! Indicates if environment scalar variables are allocated on materials.
  virtual bool isAllocateScalarEnvironmentVariableAsMaterial() const = 0;

  //! Manager name
  virtual String name() const = 0;

  /*!
   * \brief Name of the service used to compress data during forceRecompute().
   *
   * If null (the default), no compression is performed.
   */
  virtual void setDataCompressorServiceName(const String& name) = 0;

  //! Virtual name of the service used to compress data
  virtual String dataCompressorServiceName() const = 0;

  //! List of materials
  virtual ConstArrayView<IMeshMaterial*> materials() const = 0;

  //! List of materials viewed as components
  virtual MeshComponentList materialsAsComponents() const = 0;

  //! List of environments
  virtual ConstArrayView<IMeshEnvironment*> environments() const = 0;

  //! List of environments viewed as components
  virtual MeshComponentList environmentsAsComponents() const = 0;

  /*!
   * \brief List of all components.
   *
   * This list is the concatenation of environmentsAsComponents() and
   * materialsAsComponents(). It is only valid once endCreate() has been called.
   */
  virtual MeshComponentList components() const = 0;

  //! List of blocks
  virtual ConstArrayView<IMeshBlock*> blocks() const = 0;

  /*!
   * \brief Returns the environment with name \a name.
   *
   * If no environment with this name exists, returns null if \a throw_exception is \a false
   * and throws an exception if \a throw_exception is \a true.
   */
  virtual IMeshEnvironment* findEnvironment(const String& name, bool throw_exception = true) = 0;

  /*!
   * \brief Returns the block with name \a name.
   *
   * If no block with this name exists, returns null if \a throw_exception is \a false
   * and throws an exception if \a throw_exception is \a true.
   */
  virtual IMeshBlock* findBlock(const String& name, bool throw_exception = true) = 0;

  /*!
   * \brief Fills the array \a variables with the list of used material variables.
   *
   * The array \a variables is cleared before the call.
   */
  virtual void fillWithUsedVariables(Array<IMeshMaterialVariable*>& variables) = 0;

  //! Variable with name \a name or \a nullptr if none of this name exists.
  virtual IMeshMaterialVariable* findVariable(const String& name) = 0;

  //! Material variable associated with the global variable \a global_var (\a nullptr if none)
  virtual IMeshMaterialVariable* checkVariable(IVariable* global_var) = 0;

  //! Writes the material and environment info to the stream \a o
  virtual void dumpInfos(std::ostream& o) = 0;

  //! Writes the cell info \a cell to the stream \a o
  virtual void dumpCellInfos(Cell cell, std::ostream& o) = 0;

  //! Checks the validity of internal structures
  virtual void checkValid() = 0;

  //! View of environment cells corresponding to the group \a cells
  virtual AllEnvCellVectorView view(const CellGroup& cells) = 0;

  //! View of environment cells corresponding to the group \a cells
  virtual AllEnvCellVectorView view(CellVectorView cells) = 0;

  //! View of environment cells corresponding to local cell IDs cells_local_id
  virtual AllEnvCellVectorView view(SmallSpan<const Int32> cell_local_id) = 0;

  //! Creates an instance to convert from 'Cell' to 'AllEnvCell'
  virtual CellToAllEnvCellConverter cellToAllEnvCellConverter() = 0;

  /*!
   * \brief Forces the recalculation of material information.
   *
   * This method allows forcing the recalculation of information on cells
   * mixed, for example, following a mesh change.
   * This is a temporary method that will eventually be removed.
   * Mixed values are invalidated after calling this method.
   */
  virtual void forceRecompute() = 0;

  //! Lock used for multi-threading
  virtual Mutex* variableLock() = 0;

  /*!
   * \brief Synchronizes material cells.
   *
   * This method allows synchronizing between subdomains the
   * cells of each material. It is collective
   *
   * During this call, the subdomain owning N cells
   * sends to the subdomains that possess these \a N cells as ghost cells the
   * list of materials it possesses. These latter subdomains
   * update this list by adding or removing the necessary
   * materials.
   *
   * After this call, it is guaranteed that
   * the ghost cells of a subdomain have the same list of
   * materials and environments as the cells of the subdomain that owns
   * these cells. It is possible, in particular, to synchronize variables
   * via MeshMaterialVariableRef::synchronize().
   *
   * Returns \a true if the materials of this subdomain have been modified following
   * the synchronization, \a false otherwise.
   */
  virtual bool synchronizeMaterialsInCells() = 0;

  /*!
   * \brief Checks that material cells are consistent between
   * subdomains.
   *
   * This method allows checking that all ghost cells
   * of our subdomain have the same list of materials as
   * the associated owned cells.
   *
   * In case of an error, the list of inconsistent cells is displayed
   * and a FatalErrorException is raised.
   *
   * \a max_print indicates the maximum number of errors to display in case of an error.
   * If it is negative, all cells are displayed.
   */
  virtual void checkMaterialsInCells(Integer max_print = 10) = 0;

  //! Applies the functor \a functor to all material variables
  virtual void visitVariables(IFunctorWithArgumentT<IMeshMaterialVariable*>* functor) = 0;

  /*!
   * \brief Counter for the number of modifications of the material list
   * and environments.
   *
   * This counter increases every time materials are added
   * or removed. The increment is not necessarily constant.
   *
   * \note Currently, this counter is not saved during a
   * protection and will therefore be 0 during recovery.
   */
  virtual Int64 timestamp() const = 0;

  /*!
   * \brief Sets the version of the implementation for synchronizing
   * material variables.
   */
  virtual void setSynchronizeVariableVersion(Integer version) = 0;

  /*!
   * \brief Version of the implementation for synchronizing
   * material variables.
   */
  virtual Integer synchronizeVariableVersion() const = 0;

  //! True if a mesh exchange with material management is underway.
  virtual bool isInMeshMaterialExchange() const = 0;

  //! Interface of the variable factory
  virtual IMeshMaterialVariableFactoryMng* variableFactoryMng() const = 0;

  /*!
   * \brief Activates or deactivates the construction and update of the table of
   * "connectivity" CellLocalId -> AllEnvCell for RUNCOMMAND
   *
   * It can also be activated by the environment variable ARCANE_ALLENVCELL_FOR_RUNCOMMAND.
   * Optionally, the table can be forced to be created, which can be useful during a late call
   * of this method compared to ForceRecompute()
   */
  virtual void enableCellToAllEnvCellForRunCommand(bool is_enable, bool force_create = false) = 0;
  virtual bool isCellToAllEnvCellForRunCommand() const = 0;

  /*!
   * \brief Indicates whether the material or environment value is used when transforming a cell
   * from a partial cell to a pure cell.
   *
   * When transitioning from a partial cell to a pure cell, the partial value must be copied
   * into the global value. By default, the behavior is not the same
   * depending on whether optimizations are active or not (\sa modificationFlags()).
   * Without optimization, the material value is used. If the optimization
   * eModificationFlags::GenericOptimize is active, the environment value is used.
   *
   * If this property is true, it allows using the material value
   * in all cases.
   */
  virtual void setUseMaterialValueWhenRemovingPartialValue(bool v) = 0;
  virtual bool isUseMaterialValueWhenRemovingPartialValue() const = 0;

 public:

  //!\internal
  class IFactory
  {
   public:

    virtual ~IFactory() = default;
    virtual Ref<IMeshMaterialMng> getTrueReference(const MeshHandle& mesh_handle, bool is_create) = 0;
  };

 private:

  //!\internal
  static void _internalSetFactory(IFactory* f);

 public:

  //! Internal API for %Arcane
  virtual IMeshMaterialMngInternal* _internalApi() const = 0;

  /*!
   * \internal
   * \brief Synchronizer for material and environment variables on all cells.
   */
  virtual IMeshMaterialVariableSynchronizer* _allCellsMatEnvSynchronizer() = 0;

  /*!
   * \internal
   * \brief Synchronizer for environment-only variables on all cells.
   */
  virtual IMeshMaterialVariableSynchronizer* _allCellsEnvOnlySynchronizer() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
