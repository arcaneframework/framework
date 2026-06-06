// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableRef.h                                   (C) 2000-2025 */
/*                                                                           */
/* Reference to a variable on a mesh material.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MESHMATERIALVARIABLEREF_H
#define ARCANE_CORE_MATERIALS_MESHMATERIALVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file MeshMaterialVariableRef.h
 *
 * This file contains the different types managing references
 * to material variables.
 */

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array2View.h"

#include "arcane/core/Item.h"
#include "arcane/core/VariableRef.h"

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/MeshMaterialVariableComputeFunction.h"
#include "arcane/core/materials/IScalarMeshMaterialVariable.h"
#include "arcane/core/materials/IArrayMeshMaterialVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Base class for material variable references.
 */
class ARCANE_CORE_EXPORT MeshMaterialVariableRef
{
 public:
  class Enumerator
  {
   public:

    explicit Enumerator(const IMeshMaterialVariable* vp)
    : m_vref(vp->firstReference())
    {
    }
    void operator++()
    {
      m_vref = m_vref->nextReference();
    }
    MeshMaterialVariableRef* operator*() const
    {
      return m_vref;
    }
    bool hasNext() const
    {
      return m_vref;
    }

   private:

    MeshMaterialVariableRef* m_vref = nullptr;
  };

 public:

  MeshMaterialVariableRef();
  virtual ~MeshMaterialVariableRef();

 public:

  //! Previous reference (or null) on variable()
  MeshMaterialVariableRef* previousReference();

  //! Next reference (or null) on variable()
  MeshMaterialVariableRef* nextReference();

  /*!
   * \internal
   * \brief Positions the previous reference.
   *
   * For internal use only.
   */
  void setPreviousReference(MeshMaterialVariableRef* v);

  /*!
   * \internal
   * \brief Positions the next reference.
   *
   * For internal use only.
   */
  void setNextReference(MeshMaterialVariableRef* v);

  //! Registers the variable (internal)
  void registerVariable();

  //! Unregisters the variable (internal)
  void unregisterVariable();

  virtual void updateFromInternal() =0;
  
  //! Associated material variable.
  IMeshMaterialVariable* materialVariable() const { return m_material_variable; }

  //! Synchronizes values between sub-domains
  void synchronize();

  //! Adds this variable to the synchronization list \a sync_list.
  void synchronize(MeshMaterialVariableSynchronizerList& sync_list);

  //! Definition space of the variable (material+environment or environment only)
  MatVarSpace space() const { return m_material_variable->space(); }

  /*!
   * \brief Fills partial values with the super mesh value.
   * If \a level is LEVEL_MATERIAL, copies material values with those of the environment.
   * If \a level is LEVEL_ENVIRONMENT, copies environment values with
   * those of the global mesh.
   * If \a level is LEVEL_ALLENVIRONMENT, fills all partial values
   * with those of the global mesh (this makes this method equivalent to
   * fillGlobalValuesWithGlobalValues().
   */
  void fillPartialValuesWithSuperValues(Int32 level)
  {
    m_material_variable->fillPartialValuesWithSuperValues(level);
  }

 public:

  // Functions inherited from VariableRef. Eventually, the material variable
  // will derive from the classic variable.
  //@{ Functions inherited from VariablesRef. These functions apply to the associated global variable.
  String name() const;
	void setUpToDate();
	bool isUsed() const;
	void update();
	void addDependCurrentTime(const VariableRef& var);
	void addDependCurrentTime(const VariableRef& var,const TraceInfo& tinfo);
  void addDependCurrentTime(const MeshMaterialVariableRef& var);
  void addDependPreviousTime(const MeshMaterialVariableRef& var);
  void removeDepend(const MeshMaterialVariableRef& var);
	template<typename ClassType> void
	setComputeFunction(ClassType* instance,void (ClassType::*func)())
	{ m_global_variable->setComputeFunction(new VariableComputeFunction(instance,func)); }
  //@}

  //! Functions to manage dependencies on the material part of the variable.
  //@{
	void setUpToDate(IMeshMaterial*);
	void update(IMeshMaterial*);
	void addMaterialDepend(const VariableRef& var);
	void addMaterialDepend(const VariableRef& var,const TraceInfo& tinfo);
	void addMaterialDepend(const MeshMaterialVariableRef& var);
	void addMaterialDepend(const MeshMaterialVariableRef& var,const TraceInfo& tinfo);
	template<typename ClassType> void
	setMaterialComputeFunction(ClassType* instance,void (ClassType::*func)(IMeshMaterial*))
	{ m_material_variable->setComputeFunction(new MeshMaterialVariableComputeFunction(instance,func)); }
  //@}

 protected:
  
  void _internalInit(IMeshMaterialVariable* mat_variable);
  bool _isRegistered() const { return m_is_registered; }

 private:

  //! Associated variable
  IMeshMaterialVariable* m_material_variable = nullptr;

  //! Previous reference on \a m_variable
  MeshMaterialVariableRef* m_previous_reference = nullptr;

  //! Next reference on \a m_variable
  MeshMaterialVariableRef* m_next_reference = nullptr;

  //! Associated global variable
  IVariable* m_global_variable = nullptr;

  bool m_is_registered = false;

 private:
  void _checkValid() const
  {
#ifdef ARCANE_CHECK
    if (!m_material_variable)
      _throwInvalid();
#endif
  }
  void _throwInvalid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Scalar variable on the meshes of a mesh material.
 *
 * For now, this class is only instantiated for meshes
 */
template<typename DataType_>
class CellMaterialVariableScalarRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IScalarMeshMaterialVariable<Cell, DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableScalarRefT<ItemType, DataType>;
  using ThatClass = CellMaterialVariableScalarRef<DataType>;

 public:

  //! Constructs a reference to the variable specified in \a vb
  explicit ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(const VariableBuildInfo& vb);
  //! Constructs a reference to the variable specified in \a vb
  explicit ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(const MaterialVariableBuildInfo& vb);
  /*!
   * \brief Constructs a reference to the variable \a var.
   *
   * \a var must have the data type \a DataType and must be a scalar variable, otherwise
   * an exception is raised.
   */
  explicit ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(IMeshMaterialVariable* var);
  ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(const ThatClass& rhs);

 public:

  //! Copy assignment operator (deleted)
  ARCANE_CORE_EXPORT ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Default constructor (deleted)
  CellMaterialVariableScalarRef() = delete;

 public:

  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT virtual void refersTo(const ThatClass& rhs);

  /*!
   * \internal
   */
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 protected:

  DataType operator[](MatVarIndex mvi) const
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }
  DataType& operator[](MatVarIndex mvi)
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

 public:

  //! Partial value of the variable for the material mesh \a mc
  DataType operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Partial value of the variable for the material mesh \a mc
  DataType& operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Partial value of the variable for the iterator \a mc
  DataType operator[](CellComponentCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

  //! Partial value of the variable for the iterator `mc`
  DataType& operator[](CellComponentCellEnumerator mc)
  {
    return this->operator[](mc._varIndex());
  }

  //! Global value of the variable for the cell `c`
  DataType operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Global value of the variable for the cell `c`
  DataType& operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

  //! Value of the variable for the material cell `mvi`
  DataType operator[](PureMatVarIndex mvi) const
  {
    return m_value[0][mvi.valueIndex()];
  }

  //! Value of the variable for the material cell `mvi`
  DataType& operator[](PureMatVarIndex mvi)
  {
    return m_value[0][mvi.valueIndex()];
  }

  /*!
   * \brief Value of the variable for the material with index `mat_id` of
   * the cell, or 0 if absent from the cell.
   */
  ARCANE_CORE_EXPORT DataType matValue(AllEnvCell c,Int32 mat_id) const;

  /*!
   * \brief Value of the variable for the environment with index `env_id` of
   * the cell, or 0 if absent from the cell.
   */
  ARCANE_CORE_EXPORT DataType envValue(AllEnvCell c,Int32 env_id) const;

 public:
  
  ARCANE_CORE_EXPORT void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values);
  ARCANE_CORE_EXPORT void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,Int32ConstArrayView indexes);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,Array<DataType>& values);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,Array<DataType>& values,Int32ConstArrayView indexes);
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  ARCANE_CORE_EXPORT void fillPartialValues(const DataType& value);

 public:

  //! Global variable associated with this material variable
  ARCANE_CORE_EXPORT GlobalVariableRefType& globalVariable();
  //! Global variable associated with this material variable
  ARCANE_CORE_EXPORT const GlobalVariableRefType& globalVariable() const;

 private:

  PrivatePartType* m_private_part = nullptr;
  ArrayView<DataType>* m_value = nullptr;
  ArrayView<ArrayView<DataType>> m_container_value;

 private:

  void _init();
  void _setContainerView();

 public:

  // TODO: Temporary. To be removed.
  ArrayView<DataType>* _internalValue() const { return m_value; }

 public:

#ifdef ARCANE_DOTNET
  // Only for the C# wrapper
  // TODO: Eventually use 'm_container_view' instead
  void* _internalValueAsPointerOfPointer() { return reinterpret_cast<void*>(&m_value); }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Array variable over the cells of a material in the mesh.
 * For now, this class is only instantiated for cells.
 */
template<typename DataType_>
class CellMaterialVariableArrayRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IArrayMeshMaterialVariable<Cell, DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableArrayRefT<ItemType, DataType>;
  using ThatClass = CellMaterialVariableArrayRef<DataType>;

 public:

  //! Constructs a reference to the variable specified in `vb`
  explicit ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(const VariableBuildInfo& vb);
  //! Constructs a reference to the variable specified in `vb`
  explicit ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(const MaterialVariableBuildInfo& vb);
  /*!
   * \brief Constructs a reference to the variable `var`.
   *
   * \a var must have the data type `DataType` and must be an array variable, otherwise an exception is raised.
   */
  explicit ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(IMeshMaterialVariable* var);
  ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(const ThatClass& rhs);

 public:

  //! Copy operator (deleted)
  ARCANE_CORE_EXPORT ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Default constructor (deleted)
  CellMaterialVariableArrayRef() = delete;

 public:

  //! Positions the instance reference to the variable `rhs`.
  ARCANE_CORE_EXPORT virtual void refersTo(const ThatClass& rhs);

  /*!
   * \internal
   */
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 public:

  //! Global variable associated with this material variable
  ARCANE_CORE_EXPORT GlobalVariableRefType& globalVariable();
  //! Global variable associated with this material variable
  ARCANE_CORE_EXPORT const GlobalVariableRefType& globalVariable() const;

 public:

  /*!
   * \brief Resizes the number of elements in the array.
   *
   * The first dimension always remains equal to the number of elements in the mesh.
   * Only the second component is resized.
   */
  ARCANE_CORE_EXPORT void resize(Integer dim2_size);

 protected:

  ConstArrayView<DataType> operator[](MatVarIndex mvi) const
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }
  ArrayView<DataType> operator[](MatVarIndex mvi)
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

 public:

  //! Partial value of the variable for the material cell `mc`
  ConstArrayView<DataType> operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Partial value of the variable for the material cell `mc`
  ArrayView<DataType> operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Global value of the variable for the cell `c`
  ConstArrayView<DataType> operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Global value of the variable for the cell `c`
  ArrayView<DataType> operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

  //! Value of the variable for the material cell `mvi`
  ConstArrayView<DataType> operator[](PureMatVarIndex mvi) const
  {
    return m_value[0][mvi.valueIndex()];
  }

  //! Value of the variable for the material cell `mvi`
  ArrayView<DataType> operator[](PureMatVarIndex mvi)
  {
    return m_value[0][mvi.valueIndex()];
  }

 private:

  PrivatePartType* m_private_part = nullptr;
  Array2View<DataType>* m_value = nullptr;
  ArrayView<Array2View<DataType>> m_container_value;

 private:

  void _init();
  void _setContainerView();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! %Material variable of type `Byte`
typedef CellMaterialVariableScalarRef<Byte> MaterialVariableCellByte;
//! %Material variable of type `Real`
typedef CellMaterialVariableScalarRef<Real> MaterialVariableCellReal;
//! %Material variable of type `Int16`
typedef CellMaterialVariableScalarRef<Int16> MaterialVariableCellInt16;
//! %Material variable of type `Int32`
typedef CellMaterialVariableScalarRef<Int32> MaterialVariableCellInt32;
//! %Material variable of type `Int64`
typedef CellMaterialVariableScalarRef<Int64> MaterialVariableCellInt64;
//! %Material variable of type `Real2`
typedef CellMaterialVariableScalarRef<Real2> MaterialVariableCellReal2;
//! %Material variable of type `Real3`
typedef CellMaterialVariableScalarRef<Real3> MaterialVariableCellReal3;
//! %Material variable of type `Real2x2`
typedef CellMaterialVariableScalarRef<Real2x2> MaterialVariableCellReal2x2;
//! %Material variable of type `Real3x3`
typedef CellMaterialVariableScalarRef<Real3x3> MaterialVariableCellReal3x3;

#ifdef ARCANE_64BIT
//! %Material variable of type `Integer`
typedef MaterialVariableCellInt64 MaterialVariableCellInteger;
#else
//! %Material variable of type `Integer`
typedef MaterialVariableCellInt32 MaterialVariableCellInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! %Material variable of type array of `Byte`
typedef CellMaterialVariableArrayRef<Byte> MaterialVariableCellArrayByte;
//! %Material variable of type array of `Real`
typedef CellMaterialVariableArrayRef<Real> MaterialVariableCellArrayReal;
//! %Material variable of type array of `Int16`
typedef CellMaterialVariableArrayRef<Int16> MaterialVariableCellArrayInt16;
//! %Material variable of type array of `Int32`
typedef CellMaterialVariableArrayRef<Int32> MaterialVariableCellArrayInt32;
//! %Material variable of type array of `Int64`
typedef CellMaterialVariableArrayRef<Int64> MaterialVariableCellArrayInt64;
//! %Material variable of type array of `Real2`
typedef CellMaterialVariableArrayRef<Real2> MaterialVariableCellArrayReal2;
//! %Material variable of type array of `Real3`
typedef CellMaterialVariableArrayRef<Real3> MaterialVariableCellArrayReal3;
//! %Material variable of type array of `Real2x2`
typedef CellMaterialVariableArrayRef<Real2x2> MaterialVariableCellArrayReal2x2;
//! %Material variable of type array of `Real3x3`
typedef CellMaterialVariableArrayRef<Real3x3> MaterialVariableCellArrayReal3x3;

#ifdef ARCANE_64BIT
//! %Material variable of type array of `Integer`
typedef MaterialVariableCellArrayInt64 MaterialVariableCellArrayInteger;
#else
//! %Material variable of type array of `Integer`
typedef MaterialVariableCellArrayInt32 MaterialVariableCellArrayInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
