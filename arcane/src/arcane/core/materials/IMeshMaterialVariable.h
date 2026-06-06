// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariable.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface of a variable on a mesh material.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLE_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class VariableDependInfo;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialVariableComputeFunction;
class MeshMaterialVariableSynchronizerList;
class MeshMaterialVariableDependInfo;
class MeshMaterialVariableRef;
class ComponentItemListBuilder;
class IMeshMaterialVariableInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a material variable on a mesh.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariable
{
 public:

  virtual ~IMeshMaterialVariable() = default;

 public:

  //! Name of the variable.
  virtual String name() const = 0;

  //! Associated global variable on the mesh.
  virtual IVariable* globalVariable() const = 0;

  /*!
   * \internal
   * \brief Builds the variable information.
   * For internal use in Arcane.
   */
  virtual void buildFromManager(bool is_continue) = 0;

  /*!
   * \brief Synchronizes references.
   *
   * Synchronizes the values of references (VariableRef) to this variable
   * with the current value of the variable. This method is called
   * automatically when the number of elements of an array variable changes.
   */
  virtual void syncReferences() = 0;

  /*!
   * \brief Adds a reference to this variable
   *
   * Precondition: var_ref must not already reference a variable.
   */
  virtual void addVariableRef(MeshMaterialVariableRef* var_ref) = 0;

  /*!
   * \brief Removes a reference to this variable
   *
   * Precondition: var_ref must reference this variable (a call to addVariableRef()
   * must have been made on this variable).
   */
  virtual void removeVariableRef(MeshMaterialVariableRef* var_ref) = 0;

  //! \internal
  virtual MeshMaterialVariableRef* firstReference() const = 0;

  /*!
   * \internal
   * \brief Variable containing the specific values of the material mat.
   */
  virtual IVariable* materialVariable(IMeshMaterial* mat) = 0;

  /*!
   * \brief Indicates whether the variable value should be kept after a change in the list of materials.
   */
  virtual void setKeepOnChange(bool v) = 0;

  /*!
   * \brief Indicates whether the variable value should be kept after a change in the list of materials.
   */
  virtual bool keepOnChange() const = 0;

  /*!
   * \brief Synchronizes the variable.
   *
   * Synchronization is performed across all materials of the mesh.
   * It is essential that all ghost cells already have the correct
   * number of materials.
   */
  virtual void synchronize() = 0;

  virtual void synchronize(MeshMaterialVariableSynchronizerList& sync_list) = 0;

  /*!
   * \brief Dumps the variable values to the stream ostr.
   */
  virtual void dumpValues(std::ostream& ostr) = 0;

  /*!
   * \brief Dumps the variable values for the view view to the stream ostr.
   */
  virtual void dumpValues(std::ostream& ostr, AllEnvCellVectorView view) = 0;

  /*!
   * \brief Fills partial values with the value of the associated global mesh.
   */
  virtual void fillPartialValuesWithGlobalValues() = 0;

  /*!
   * \brief Fills partial values with the value of the super mesh.
   * If level equals LEVEL_MATERIAL, it copies material values with the middle one.
   * If level equals LEVEL_ENVIRONMENT, it copies environment values with
   * the global mesh's.
   * If level equals LEVEL_ALLENVIRONMENT, it fills all partial values
   * with the global mesh's value (this makes this method equivalent to
   * fillGlobalValuesWithGlobalValues()).
   */
  virtual void fillPartialValuesWithSuperValues(Int32 level) = 0;

  //! Serializes the variable for local ID entities ids.
  virtual void serialize(ISerializer* sbuffer, Int32ConstArrayView ids) = 0;

  //! Variable definition space (material+environment or environment only)
  virtual MatVarSpace space() const = 0;

 public:

  //! @name Dependency Management
  //@{
  /*! \brief Recalculates the variable for material mat if necessary
   *
   * Through the dependency mechanism, this operation is called recursively
   * on all variables that the instance depends on. The recalculation function
   * computeFunction() is then called if it turns out that one of the variables
   * it depends on has been modified more recently.
   *
   * Precondition: computeFunction() != 0
   */
  virtual void update(IMeshMaterial* mat) = 0;

  /*! \brief Indicates that the variable has just been updated.
   *
   * For correct dependency management, this property must be called every
   * time a variable has been updated.
   */
  virtual void setUpToDate(IMeshMaterial* mat) = 0;

  //! Time when the variable was updated
  virtual Int64 modifiedTime(IMeshMaterial* mat) = 0;

  //! Adds var to the dependency list
  virtual void addDepend(IMeshMaterialVariable* var) = 0;

  //! Adds var to the dependency list with trace info tinfo
  virtual void addDepend(IMeshMaterialVariable* var, const TraceInfo& tinfo) = 0;

  //! Adds var to the dependency list
  virtual void addDepend(IVariable* var) = 0;

  //! Adds var to the dependency list with trace info tinfo
  virtual void addDepend(IVariable* var, const TraceInfo& tinfo) = 0;

  /*! \brief Removes var from the dependency list
   */
  virtual void removeDepend(IMeshMaterialVariable* var) = 0;

  /*! \brief Removes var from the dependency list
   */
  virtual void removeDepend(IVariable* var) = 0;

  /*! \brief Sets the variable's recalculation function.
   *
   * If a recalculation function already existed, it is destroyed
   * and replaced by this one.
   */
  virtual void setComputeFunction(IMeshMaterialVariableComputeFunction* v) = 0;

  //! Function used to update the variable
  virtual IMeshMaterialVariableComputeFunction* computeFunction() = 0;

  /*!
   * \brief Dependency information.
   *
   * Fills the array infos with dependency information on global variables
   * and the array mat_infos with those on material variables.
   */
  virtual void dependInfos(Array<VariableDependInfo>& infos,
                           Array<MeshMaterialVariableDependInfo>& mat_infos) = 0;
  //@}

 public:

  virtual IMeshMaterialVariableInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Class for managing the creation of the concrete type of the material variable.
 */
template <typename TrueType>
class MeshMaterialVariableBuildTraits
{
 public:

  static ARCANE_CORE_EXPORT MaterialVariableTypeInfo _buildVarTypeInfo(MatVarSpace space);
  static ARCANE_CORE_EXPORT TrueType* getVariableReference(const MaterialVariableBuildInfo& v, MatVarSpace mvs);
  static ARCANE_CORE_EXPORT TrueType* getVariableReference(IMeshMaterialVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
