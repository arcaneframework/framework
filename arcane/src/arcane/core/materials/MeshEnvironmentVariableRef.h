// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironmentVariableRef.h                                (C) 2000-2024 */
/*                                                                           */
/* Reference to a variable on a mesh environment.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHENVIRONMENTVARIABLEREF_H
#define ARCANE_MATERIALS_MESHENVIRONMENTVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file MeshEnvironmentVariableRef.h
 *
 * This file contains the different types managing references
 * on environment variables.
 */

#include "arcane/core/materials/MeshMaterialVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Scalar variable on the cells of a mesh environment.
 *
 * This type of variable is identical except that it only has values
 * on environments and global cells but not on materials.
 */
template <typename DataType_>
class CellEnvironmentVariableScalarRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IScalarMeshMaterialVariable<Cell, DataType>;
  using ThatClass = CellEnvironmentVariableScalarRef<DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableScalarRefT<ItemType, DataType>;

 public:

  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableScalarRef(const VariableBuildInfo& vb);
  //! Constructs a reference to the variable specified in \a vb
  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableScalarRef(const MaterialVariableBuildInfo& vb);
  ARCANE_CORE_EXPORT CellEnvironmentVariableScalarRef(const ThatClass& rhs);

 public:

  //! Copy assignment operator (deleted)
  ARCANE_CORE_EXPORT ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Default constructor (deleted)
  CellEnvironmentVariableScalarRef() = delete;

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

  //! Partial value of the variable for material cell \a mc
  DataType operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Partial value of the variable for material cell \a mc
  DataType& operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Global value of the variable for cell \a c
  DataType operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Global value of the variable for cell \a c
  DataType& operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

  /*!
   * \brief Value of the variable for the environment index \a env_id of
   * cell \a or 0 if absent from the cell.
   */
  ARCANE_CORE_EXPORT DataType envValue(AllEnvCell c, Int32 env_id) const;

 public:

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

 public:

  // TODO: Temporary. To be deleted.
  ArrayView<DataType>* _internalValue() const { return m_value; }

 private:

  void _init();
  void _setContainerView();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Array variable on the cells of a mesh material.
 * For now, this class is only instantiated for cells
 */
template <typename DataType_>
class CellEnvironmentVariableArrayRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IArrayMeshMaterialVariable<Cell, DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableArrayRefT<ItemType, DataType>;
  using ThatClass = CellEnvironmentVariableArrayRef<DataType>;

 public:

  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableArrayRef(const VariableBuildInfo& vb);
  //! Constructs a reference to the variable specified in \a vb
  explicit ARCANE_CORE_EXPORT CellEnvironmentVariableArrayRef(const MaterialVariableBuildInfo& vb);
  ARCANE_CORE_EXPORT CellEnvironmentVariableArrayRef(const ThatClass& rhs);

 public:

  //! Copy assignment operator (deleted)
  ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Default constructor (deleted)
  CellEnvironmentVariableArrayRef() = delete;

 public:

  //! Positions the instance reference to the variable \a rhs.
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
   * The first dimension always remains equal to the number of mesh elements.
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

  //! Partial value of the variable for material cell \a mc
  ConstArrayView<DataType> operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Partial value of the variable for material cell \a mc
  ArrayView<DataType> operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Global value of the variable for cell \a c
  ConstArrayView<DataType> operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Global value of the variable for cell \a c
  ArrayView<DataType> operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
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

//! Environment variable of type \a #Byte
typedef CellEnvironmentVariableScalarRef<Byte> EnvironmentVariableCellByte;
//! Environment variable of type \a #Real
typedef CellEnvironmentVariableScalarRef<Real> EnvironmentVariableCellReal;
//! Environment variable of type \a #Int16
typedef CellEnvironmentVariableScalarRef<Int16> EnvironmentVariableCellInt16;
//! Environment variable of type \a #Int32
typedef CellEnvironmentVariableScalarRef<Int32> EnvironmentVariableCellInt32;
//! Environment variable of type \a #Int64
typedef CellEnvironmentVariableScalarRef<Int64> EnvironmentVariableCellInt64;
//! Environment variable of type \a Real2
typedef CellEnvironmentVariableScalarRef<Real2> EnvironmentVariableCellReal2;
//! Environment variable of type \a Real3
typedef CellEnvironmentVariableScalarRef<Real3> EnvironmentVariableCellReal3;
//! Environment variable of type \a Real2x2
typedef CellEnvironmentVariableScalarRef<Real2x2> EnvironmentVariableCellReal2x2;
//! Environment variable of type \a Real3x3
typedef CellEnvironmentVariableScalarRef<Real3x3> EnvironmentVariableCellReal3x3;

#ifdef ARCANE_64BIT
//! Environment variable of type \a #Integer
typedef EnvironmentVariableCellInt64 EnvironmentVariableCellInteger;
#else
//! Environment variable of type \a #Integer
typedef EnvironmentVariableCellInt32 EnvironmentVariableCellInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Environment variable of type array of \a #Byte
typedef CellEnvironmentVariableArrayRef<Byte> EnvironmentVariableCellArrayByte;
//! Environment variable of type array of \a #Real
typedef CellEnvironmentVariableArrayRef<Real> EnvironmentVariableCellArrayReal;
//! Environment variable of type array of \a #Int16
typedef CellEnvironmentVariableArrayRef<Int16> EnvironmentVariableCellArrayInt16;
//! Environment variable of type array of \a #Int32
typedef CellEnvironmentVariableArrayRef<Int32> EnvironmentVariableCellArrayInt32;
//! Environment variable of type array of \a #Int64
typedef CellEnvironmentVariableArrayRef<Int64> EnvironmentVariableCellArrayInt64;
//! Environment variable of type array of \a Real2
typedef CellEnvironmentVariableArrayRef<Real2> EnvironmentVariableCellArrayReal2;
//! Environment variable of type array of \a Real3
typedef CellEnvironmentVariableArrayRef<Real3> EnvironmentVariableCellArrayReal3;
//! Environment variable of type array of \a Real2x2
typedef CellEnvironmentVariableArrayRef<Real2x2> EnvironmentVariableCellArrayReal2x2;
//! Environment variable of type array of \a Real3x3
typedef CellEnvironmentVariableArrayRef<Real3x3> EnvironmentVariableCellArrayReal3x3;

#ifdef ARCANE_64BIT
//! Environment variable of type array of \a #Integer
typedef EnvironmentVariableCellInt64 EnvironmentVariableCellArrayInteger;
#else
//! Environment variable of type array of \a #Integer
typedef EnvironmentVariableCellInt32 EnvironmentVariableCellArrayInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
