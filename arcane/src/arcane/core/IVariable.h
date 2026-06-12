// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariable.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Interface of the Variable class.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLE_H
#define ARCANE_CORE_IVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/SerializeGlobal.h"

#include "arcane/utils/Ref.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a variable.
 *
 * The implementation of this interface is the Variable class.
 *
 * Generally, this interface is not used directly. Variables are managed by
 * the VariableRef class and its derived classes.
 */
class ARCANE_CORE_EXPORT IVariable
{
 public:

  //! Dependency Type
  enum eDependType
  {
    DPT_PreviousTime,
    DPT_CurrentTime
  };

 public:

  /*!
   * \brief Properties of a variable.
   */
  enum
  {
    /*!
     * \brief Indicates that the variable should not be saved.
     *
     * This property is collective: it must be set on all subdomains (or on none).
     */
    PNoDump = (1 << 0),

    /*!
     * \brief Indicates that the variable is not necessarily synchronized.
     *
     * This means it is normal for the variable values to be different from
     * one processor to another on ghost cells
     */
    PNoNeedSync = (1 << 1),

    //! Indicates that the variable is traced (only in trace mode)
    PHasTrace = (1 << 2),

    /*! \brief Indicates that the variable value is dependent on the subdomain.
     *
     * This means, among other things, that the variable value is different
     * as soon as the number of subdomains changes. This is, for example, the case
     * of the variable containing the number of the subdomain owning an entity.
     */
    PSubDomainDepend = (1 << 3),

    /*! \brief Indicates that the variable is private to the subdomain.
     *
     * This means that the variable is dependent on the subdomain and specifically
     * that it does not necessarily exist on all subdomains. This
     * property cannot be set for mesh variables.
     */
    PSubDomainPrivate = (1 << 4),

    /*! \brief Indicates that the variable value is dependent on the execution
     *
     * The values of these variables change between two executions. This is, for
     * example, the case of a variable containing the CPU time used.
     */
    PExecutionDepend = (1 << 5),

    /*! \brief Indicates that the variable is private
     *
     * A private variable cannot possess more than one reference.
     * This property can only be set when the variable is created
     */
    PPrivate = (1 << 6),

    /*! \brief Indicates that the variable is temporary
     *
     * A temporary variable is temporary, as its name suggests. It
     * cannot be saved, is not transferred during mesh balancing (but can be
     * synchronized), and is not saved during rollback.
     *
     * A temporary variable that is no longer used (no references to it)
     * can be deallocated.
     */
    PTemporary = (1 << 7),

    /*! \brief Indicates that the variable should not be restored.
     *
     * A variable of this type is neither saved nor restored during rollback.
     */
    PNoRestore = (1 << 8),

    /*! \brief Indicates that the variable should not be exchanged.
     *
     * A variable of this type is not exchanged during mesh repartitioning, for
     * example. This prevents the sending of unnecessary data
     * if this variable is only used temporarily or
     * if it is recalculated in one of the entry points called
     * following a repartitioning.
     */
    PNoExchange = (1 << 9),

    /*!
     * \brief Indicates that the variable is persistent.
     *
     * A persistent variable is not destroyed even if there are no references
     * to it anymore.
     */
    PPersistant = (1 << 10),

    /*!
     * \brief Indicates that the variable does not necessarily have the same value
     * across replicas.
     *
     * This means it is normal for the variable values to be different on the same
     * subdomains of other replicas.
     */
    PNoReplicaSync = (1 << 11),

    /*!
     * \brief Indicates that the variable must be allocated in shared memory.
     *
     * The MachineShMemWinMemoryAllocator will be used.
     * The MachineShMemWinVariable class can be used with
     * this variable.
     */
    PInShMem = (1 << 12),

    /*!
     * \brief Indicates that the save will be null for this variable and for
     * this subdomain.
     *
     * A save will be performed, but with a default value
     * (value = 0 for a scalar, empty array for an array).
     *
     * Only works for unsupported variables.
     */
    PDumpNull = (1 << 13)
  };

 public:

  //! Tag used to indicate if a variable will be post-processed
  static const char* TAG_POST_PROCESSING;

  //! Tag used to indicate if a variable will be post-processed at this iteration
  static const char* TAG_POST_PROCESSING_AT_THIS_ITERATION;

 public:

  friend class VariableMng;

 public:

  virtual ~IVariable() = default; //!< Frees resources

 public:

  //! Subdomain associated with the variable (TODO deprecate end of 2023)
  virtual ISubDomain* subDomain() = 0;

 public:

  //! Variable manager associated with the variable
  virtual IVariableMng* variableMng() const = 0;

  //! Memory size (in Bytes) used by the variable
  virtual Real allocatedMemory() const = 0;

  //! Variable name
  virtual String name() const = 0;

  //! Full variable name (with family prefix)
  virtual String fullName() const = 0;

  //! Data type managed by the variable (Real, Integer, ...)
  virtual eDataType dataType() const = 0;

  /*!
   * \brief Kind of mesh entities on which the variable is based.
   *
   * For scalar or array variables, there is no kind, and the
   * method returns #IK_Unknown.
   * For other variables, it returns the kind of the mesh element (Node, Cell, ...), namely:
   * - #IK_Node for nodes
   * - #IK_Edge for edges
   * - #IK_Face for faces
   * - #IK_Cell for cells
   * - #IK_Particle for particles
   * - #IK_DoF for degrees of freedom
   */
  virtual eItemKind itemKind() const = 0;

  /*!
   * \brief Dimension of the variable.
   *
   * The possible values are as follows:
   * - 0 for a scalar variable.
   * - 1 for a mono-dimensional array variable or a mesh scalar variable.
   * - 2 for a bi-dimensional array variable or a mesh array variable.
   */
  virtual Integer dimension() const = 0;

  /*!
   * \brief Indicates if the variable is a multi-sized array.
   *
   * This value is only useful for 2D or higher arrays.
   * - 0 for a scalar variable or standard 2D array.
   * - 1 for a multi-sized 2D array variable.
   * - 2 for an ancient format 2D array variable (obsolete).
   */
  virtual Integer multiTag() const = 0;

  /*!
   * \brief Number of elements of the variable.
   *
   * The returned values depend on the dimension of the variable:
   * - for dimension 0, returns 1,
   * - for dimension 1, returns the number of elements in the array
   * - for dimension 2, returns the total number of elements by summing
   * the number of elements per dimension.
   */
  virtual Integer nbElement() const = 0;

  //! Returns the properties of the variable
  virtual int property() const = 0;

  //! Indicates that the properties of one of the references to this
  //! variable have changed (internal)
  virtual void notifyReferencePropertyChanged() = 0;

  /*!
   * \brief Adds a reference to this variable
   *
   * \pre \a var_ref must not already reference a variable.
   */
  virtual void addVariableRef(VariableRef* var_ref) = 0;

  /*!
   * \brief Removes a reference to this variable
   *
   * \pre \a var_ref must reference this variable (an call to addVariableRef()
   * must have been made on this variable).
   */
  virtual void removeVariableRef(VariableRef* var_ref) = 0;

  //! First reference (or null) on this variable
  virtual VariableRef* firstReference() const = 0;

  //! Number of references on this variable
  virtual Integer nbReference() const = 0;

 public:

  ARCANE_DEPRECATED_REASON("Y2021: This method is a noop")
  virtual void setTraceInfo(Integer id, eTraceType tt) = 0;

 public:

  /*!
   * \brief Sets the number of elements for an array variable.
   *
   * When the variable is a 1D or 2D array type, it sets the number
   * of array elements to \a new_size. For a 2D array, the number
   * of elements in the first dimension is modified.
   *
   * This operation must not be called for mesh variables
   * because the number of elements is determined automatically based on the number
   * of entities in the group it relies on. For this type of variable,
   * resizeFromGroup() must be called.
   *
   * This operation synchronizes the references (syncReferences()).
   */
  virtual void resize(Integer new_size) = 0;

  /*!
   * \brief Sets the number of elements for a mesh variable.
   *
   * Reallocates the size of the mesh variable based on the group
   * it relies on.
   *
   * This operation only has an effect on mesh variables.
   * For others, no action is performed.
   *
   * This operation synchronizes the references (syncReferences()).
   */
  virtual void resizeFromGroup() = 0;

  /*!
   * \brief Frees any additional memory allocated for the data.
   *
   * This method is only useful for non-scalar variables
   */
  virtual void shrinkMemory() = 0;

  //! Sets allocation information
  virtual void setAllocationInfo(const DataAllocationInfo& v) = 0;

  //! Allocation information
  virtual DataAllocationInfo allocationInfo() const = 0;

 public:

  /*!
   * \brief Initializes the variable on a group.
   *
   * Initializes the variable with the value \a value for all elements of the
   * group \a group.
	 *
   * This operation is only usable with mesh variables.
	 *
   * \param group_name group. It must correspond to an existing group
   * of the variable's type (e.g., CellGroup for a cell variable).
   * \param value initialization value. The string must be convertible
   * to the variable's type.
   *
   * \retval true in case of error or if the variable is not a mesh variable.
   * \retval false if the initialization is successful.
  */
  virtual bool initialize(const ItemGroup& group, const String& value) = 0;

  //! @name Verification Operations
  //@{
  /*! \brief Checks if the variable is properly synchronized.
   *
   * This operation only works for mesh variables.
   *
   * A variable is synchronized when its values are the same
   * across all subdomains, both on owned elements and ghost elements.
   *
   * For each unsynchronized element, a message is displayed.
   *
   * \param max_print maximum number of messages to display.
   * If 0, no element is displayed. If positive, display at most
   * \a max_print elements. If negative, all elements are displayed.
   *
   * \return the number of different reference values.
   */
  virtual Int32 checkIfSync(Integer max_print = 0) = 0;

  /*! \brief Checks that the variable is identical to a reference value
   *
   * This operation checks that the variable values are identical
   * to a reference value read from the reader \a reader.
   *
   * For each value different from the reference, a message is displayed.
   *
   * \param max_print maximum number of messages to display.
   * If 0, no element is displayed. If positive, display at most
   * \a max_print elements. If negative, all elements are displayed.
   * \param compare_ghost if true, compares values both on owned entities
   * and ghost entities. Otherwise, it only compares on owned entities.
   *
   * \return the number of different reference values.
   */
  virtual Int32 checkIfSame(IDataReader* reader, Integer max_print, bool compare_ghost) = 0;

  /*!
   * \brief Checks if the variable has the same values on all replicas.

   *
   * Compare the variable's values with those of the same subdomain
   * of other replicas. For each different element,
   * a message is displayed.
   *
   * This method is collective across the same subdomain as other replicas.
   * Therefore, it should only be called if the variable exists on all subdomains
   * otherwise it causes a blocking.
   *
   * This method only works for variables of numeric types.
   * In this case, it throws a NotSupportedException.
   *
   * \param max_print maximum number of messages to display.
   * If 0, no elements are displayed. If positive, displays at most
   * \a max_print elements. If negative, all elements are displayed.
   * For each different element, the minimum and
   * maximum value is displayed.
   *
   * \return the number of different values of the reference.
   */
  virtual Int32 checkIfSameOnAllReplica(Integer max_print = 0) = 0;
  //@}

  /*!
   * \brief Synchronizes the variable.
   *
   La synchronisation ne peut se faire que sur les variables du maillage.
   */
  virtual void synchronize() = 0;

  // TODO: à rendre virtuelle pure (décembre 2024)
  /*!
   * \brief Synchronizes the variable on a list of entities.
   *
   * Synchronization can only be performed on mesh variables.
   * Only the entities listed in \a local_ids will be synchronized. Note:
   * an entity present in this list on one subdomain must be present
   * in this list for any other subdomain that possesses this entity.
   */
  virtual void synchronize(Int32ConstArrayView local_ids);

  /*!
   * \brief Mesh associated with the variable.
   *
   * This operation is only meaningful for variables on
   * mesh entities.
   */
  ARCCORE_DEPRECATED_2020("Use meshHandle() instead")
  virtual IMesh* mesh() const = 0;

  /*!
   * \brief Mesh associated with the variable.
   *
   * This operation is only meaningful for variables on
   * mesh entities.
   */
  virtual MeshHandle meshHandle() const = 0;

  /*!
   * \brief Associated mesh group.
   *
   * \return the associated mesh group if for a mesh variable
   * or the null group if the variable is not a mesh variable.
   *
   * If a variable is not used or not yet allocated,
   * the returned value is the null group.
   * However, the variable can still be associated with a group.
   * In this case, you must use the itemGroupName() function to
   * retrieve the name of this group.
   */
  virtual ItemGroup itemGroup() const = 0;

  //! Name of the associated entity group.
  virtual String itemGroupName() const = 0;

  /*!
   * \brief Associated entity family.
   *
   * \return the family associated with the variable or 0
   * if the variable has no family.
   *
   * If a variable is not used or not yet allocated,
   * the returned value is null.
   * However, the variable can still be associated with a family.
   * In this case, you must use the itemFamilyName() function to
   * retrieve the name of this family.
   */
  virtual IItemFamily* itemFamily() const = 0;

  //! Name of the associated family (null if none).
  virtual String itemFamilyName() const = 0;

  //! Name of the associated mesh (null if none).
  virtual String meshName() const = 0;

  /*!
   * \brief Creates an instance containing the variable's metadata.
   *
   * The returned instance must be destroyed by calling the delete operator.
   */
  ARCANE_DEPRECATED_REASON("Y2024: Use createMetaDataRef() instead")
  virtual VariableMetaData* createMetaData() const = 0;

  //! Creates an instance containing the variable's metadata.
  virtual Ref<VariableMetaData> createMetaDataRef() const = 0;

  /*!
   * \brief Synchronizes references.
   *
   * Synchronizes the values of references (VariableRef) to this variable
   * with the variable's current value. This method is called
   * automatically when a scalar variable is modified or
   * the number of elements of an array variable changes.
   */
  virtual void syncReferences() = 0;

 public:

  /*!
   * \brief Sets the usage state of the variable
   *
   * If \v is false, the variable becomes unusable
   * and all associated resources are released.
   *
   * If \v is true, the variable is considered used and if it is
   * a mesh variable and setItemGroup() has not been called, the
   * variable is allocated to the group of all entities.
   */
  virtual void setUsed(bool v) = 0;

  //! Usage state of the variable
  virtual bool isUsed() const = 0;

  /*!
   * \brief Indicates if the variable is partial.
   *
   * A variable is partial when it is not defined on all
   * entities of a family. In this case, group()!=itemFamily()->allItems().
   */
  virtual bool isPartial() const = 0;

 public:

  /*!
   * \brief Copies the values of entities numbered @a source into entities
   * numbered @a destination
   *
   * @note This operation is internal to Arcane and must be done in
   * conjunction with the entity family corresponding to this
   * variable.
   *
   * @param source list of @b source localIds
   * @param destination list of @b destination localIds
   */
  virtual void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) = 0;

  /*!
   * \brief Copies the mean values of entities numbered
   * @a first_source and @a second_source into entities numbered
   * @a destination
   *
   * @param first_source list of @b localIds of the 1st source
   * @param second_source list of @b localIds of the 2nd source
   * @param destination list of @b destination localIds
   */
  virtual void copyItemsMeanValues(Int32ConstArrayView first_source,
                                   Int32ConstArrayView second_source,
                                   Int32ConstArrayView destination) = 0;

  /*!
   * \brief Compresses the variable's values.
   *
   * This operation is internal to Arcane and must be done in
   * conjunction with the entity family corresponding to this
   * variable.
   */
  virtual void compact(Int32ConstArrayView new_to_old_ids) = 0;

  //! pH: EXPERIMENTAL
  virtual void changeGroupIds(Int32ConstArrayView old_to_new_ids) = 0;

 public:

  //! Data associated with the variable
  virtual IData* data() = 0;

  //! Data associated with the variable
  virtual const IData* data() const = 0;

  //! Data factory associated with the variable
  virtual IDataFactoryMng* dataFactoryMng() const = 0;

  //! @name Serialization operations
  //@{
  /*! Serializes the variable.
   *
   * The \a operation is only meaningful in read mode (ISerializer::ModeGet)
   */
  virtual void serialize(ISerializer* sbuffer, IDataOperation* operation = 0) = 0;

  /*!
   * \brief Serializes the variable for identifiers \a ids.
   *
   * Serialization depends on the variable's dimension.
   * For scalar variables (dimension=0), nothing is done.
   * For array or mesh variables, \a ids corresponds to an array
   * of first dimension indirection.
   *
   * The \a operation is only meaningful in read mode (ISerializer::ModeGet)
   */
  virtual void serialize(ISerializer* sbuffer, Int32ConstArrayView ids, IDataOperation* operation = 0) = 0;

  /*!
   * \brief Saves the variable
   *
   * \deprecated Should be replaced by the following code:
   * \code
   * IVariable* var;
   * var->notifyBeginWrite();
   * writer->write(var,var->data());
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void write(IDataWriter* writer) = 0;

  /*!
   * Reads the variable.
   *
   * \deprecated Should be replaced by the following code:
   * \code
   * IVariable* var;
   * reader->read(var,var->data());
   * var->notifyEndRead();
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void read(IDataReader* reader) = 0;

  /*!
   * \brief Notifies of external modification of data().
   *
   * Signals to the instance the end of a read operation that modified
   * data(). This method must therefore be called as soon as a modification of
   * data() has been performed. This method triggers the observables registered
   * in readObservable().
   */
  virtual void notifyEndRead() = 0;

  /*!
   * \brief Notifies of the start of writing data().
   *
   * This method triggers the observables registered
   * in writeObservable().
   */
  virtual void notifyBeginWrite() = 0;

  /*!
   * \brief Write observable.
   *
   * The observers registered in this observable are called
   * before writing the variable (write operation).
   */
  virtual IObservable* writeObservable() = 0;

  /*! \brief Read observable.
   *
   * The observers registered in this observable are called
   * after reading the variable (read operation).
   */
  virtual IObservable* readObservable() = 0;

  /*! \brief Size change observable.
   *
   * The observers registered in this observable are called
   * when the number of elements of the variable changes.
   * This is the case, for example, after a remeshing for a cell variable
   */
  virtual IObservable* onSizeChangedObservable() = 0;
  //@}

  //@{ @name Tag Management
  //! Adds the tag \a tagname with the value \a tagvalue
  virtual void addTag(const String& tagname, const String& tagvalue) = 0;
  /*! \brief Removes the tag \a tagname
   *
   * If the tag \a tagname is not in the list, nothing happens.
   */
  virtual void removeTag(const String& tagname) = 0;
  //! \a true if the variable has the tag \a tagname
  virtual bool hasTag(const String& tagname) = 0;
  //! Value of the tag \a tagname. The string is null if the tag does not exist.
  virtual String tagValue(const String& tagname) = 0;
  //@}

 public:

  //! Prints the variable's values to the stream \a o
  virtual void print(std::ostream& o) const = 0;

 public:

  //! @name Dependency Management
  //@{
  /*!
   * \brief Recalculates the variable if necessary
   *
   * Through the dependency mechanism, this operation is called recursively
   * on all variables that the instance depends on. The recalculation function
   * computeFunction() is then called if it turns out that one of the variables
   * it depends on has been modified more recently.
   *
   * \pre computeFunction() != 0
   */
  virtual void update() = 0;

  virtual void update(Real wanted_time) = 0;

  /*! \brief Indicates that the variable has just been updated.
   *
   * For correct dependency management, this property
   * must be called every time a variable has been updated.
   */
  virtual void setUpToDate() = 0;

  //! Time when the variable was updated
  virtual Int64 modifiedTime() = 0;

  //! Adds \a var to the list of dependencies
  virtual void addDepend(IVariable* var, eDependType dt) = 0;

  //! Adds \a var to the list of dependencies with trace info \a tinfo
  virtual void addDepend(IVariable* var, eDependType dt, const TraceInfo& tinfo) = 0;

  /*! \brief Removes \a var from the list of dependencies
   */
  virtual void removeDepend(IVariable* var) = 0;

  /*!
   * \brief Sets the variable's recalculation function.
   *
   * The specified function \a v must be allocated via the new operator.
   * If a recalculation function already existed, it is destroyed
   * (via the delete operator) and replaced by this one.
   */
  virtual void setComputeFunction(IVariableComputeFunction* v) = 0;

  //! Function used to update the variable
  virtual IVariableComputeFunction* computeFunction() = 0;

  /*!
   * \brief Dependency information.
   *
   * Fills the array \a infos with dependency information.
   */
  virtual void dependInfos(Array<VariableDependInfo>& infos) = 0;
  //@}

 public:

  ARCANE_DEPRECATED_REASON("Y2021: This method is a noop")
  virtual IMemoryAccessTrace* memoryAccessTrace() const = 0;

  /*!
   * \brief Indicates that the variable is synchronized.
   *
   * This operation is collective.
   */
  virtual void setIsSynchronized() = 0;

  /*!
   * \brief Indicates that the variable is synchronized on the group \a item_group
   *
   * This operation is collective.
   */
  virtual void setIsSynchronized(const ItemGroup& item_group) = 0;

 public:

  //! Increments the modification counter and returns its value before modification
  static Int64 incrementModifiedTime();

 public:

  //! Internal Arcane API
  virtual IVariableInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
