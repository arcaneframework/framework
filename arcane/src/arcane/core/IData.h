// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IData.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Interface of a data item.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATA_H
#define ARCANE_CORE_IDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a data item.
 *
 * This class manages the memory associated with a variable.
 */
class ARCANE_CORE_EXPORT IData
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~IData() = default;

 public:

  //! Data type
  virtual eDataType dataType() const = 0;

  //! Dimension. 0 for a scalar, 1 for a mono-dim array, 2 for a bi-dim array.
  virtual Integer dimension() const = 0;

  //! Multi-tag. 0 if not multiple, 1 if multiple, 2 if multiple for MultiArray variables (obsolete)
  virtual Integer multiTag() const = 0;

  //! Clone the data. The created instance must be destroyed by the 'delete' operator
  ARCCORE_DEPRECATED_2020("Use cloneRef() instead")
  virtual IData* clone() = 0;

  //! Clone the data but without elements. The created instance must be destroyed by the 'delete' operator
  ARCCORE_DEPRECATED_2020("Use cloneEmptyRef() instead")
  virtual IData* cloneEmpty() = 0;

  //! Clone the data
  virtual Ref<IData> cloneRef() = 0;

  //! Clone the data but without elements.
  virtual Ref<IData> cloneEmptyRef() = 0;

  //! Information about the data container type
  virtual DataStorageTypeInfo storageTypeInfo() const = 0;

  //! Serializes the data by applying the \a operation
  virtual void serialize(ISerializer* sbuf, IDataOperation* operation) = 0;

  /*!
   * \brief Resize the data.
   *
   * This operation only makes sense for data of dimension 1 or more.
   * If the new number of elements is greater than the old one, the values added to
   * the data are not initialized.
   */
  virtual void resize(Integer new_size) = 0;

  /*!
   * \brief Serialize the data for the indices \a ids.
   *
   * This operation only makes sense for data of dimension 1 or more.
   */
  virtual void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) = 0;

  //! Fills the data with its default value.
  virtual void fillDefault() = 0;

  //! Sets the name of the data (internal)
  virtual void setName(const String& name) = 0;

  /*!
   * \brief Serialize the data.
   *
   * For performance reasons, the returned instance may directly reference
   * the memory area of this data. Consequently, it is only valid as long as this data is not
   * modified. If you wish to modify this instance, you must
   * first clone it (via IData::cloneRef()) and then serialize the cloned data.
   *
   * If \a use_basic_type is true, the data is serialized for a basic type,
   * namely #DT_Byte, #DT_Int16, #DT_Int32, #DT_Int64 or #DT_Real. Otherwise,
   * the type can be a POD, namely #DT_Byte, #DT_Int16, #DT_Int32, #DT_Int64,
   * #DT_Real, #DT_Real2, #DT_Real3, #DT_Real2x2, #DT_Real3x3.
   */
  virtual Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const = 0;

  /*!
   * \brief Assign the serialized values \a sdata to the data.
   *
   * The buffer containing the serialization values must have
   * be allocated by calling allocateBufferForSerializedData().
   */
  virtual void assignSerializedData(const ISerializedData* sdata) = 0;

  /*!
   * \brief Allocate memory to read the serialized values \a sdata.
   *
   * This method sets sdata->setBuffer(), which will contain the
   * memory needed to read the serialized data.
   */
  virtual void allocateBufferForSerializedData(ISerializedData* sdata) = 0;

  /*!
   * \brief Copy the data \a data into the current instance.
   *
   * The data \a data must be of the same type as the instance.
   */
  virtual void copy(const IData* data) = 0;

  /*!
   * \brief Swap the values of \a data with those of the instance.
   *
   * The data \a IData must be of the same type as the instance. Only
   * the values are swapped and other possible properties
   * (such as the name, for example) are not modified.
   */
  virtual void swapValues(IData* data) = 0;

  /*!
   * \brief Compute a hash key on this data.
   *
   * The key is added to \a output. The length of the key depends
   * on the algorithm used.
   */
  virtual void computeHash(IHashAlgorithm* algo, ByteArray& output) const = 0;

  /*!
   * \brief Array shape for a 1D or 2D data item.
   *
   * The shape is only considered for dimensions greater than 1.
   * For a 1D data item, the shape is therefore by default {1}. For a 2D array,
   * the shape defaults to {dim2_size}. It is possible to change the rank
   * of the shape and its values as long as shape().totalNbElement()==dim2_size.
   * For example, if the number of values dim2_size is 12, then it is
   * possible to have { 12 }, { 6, 2 } or { 3, 2, 2 } as the shape.
   *
   * The values are not preserved during a restart, so the shape must
   * be repositioned in this case. It is up to the user to ensure
   * that the shape is homogeneous across sub-domains.
   */
  virtual ArrayShape shape() const = 0;

  //! Sets the array shape.
  virtual void setShape(const ArrayShape& new_shape) = 0;

 public:

  //! Sets the allocation information
  virtual void setAllocationInfo(const DataAllocationInfo& v) = 0;

  //! Allocation information
  virtual DataAllocationInfo allocationInfo() const = 0;

 public:

  //! Applies the visitor to the data
  virtual void visit(IDataVisitor* visitor) = 0;

  /*!
   * \brief Apply the visitor to the data.
   *
   * If the data is not scalar, a
   * NotSupportedException is thrown.
   */
  virtual void visitScalar(IScalarDataVisitor* visitor) = 0;

  /*!
   * \brief Apply the visitor to the data.
   *
   * If the data is not a 1D array, an exception 
   * NotSupportedException is thrown.
   */
  virtual void visitArray(IArrayDataVisitor* visitor) = 0;

  /*!
   * \brief Apply the visitor to the data.
   *
   * If the data is not a 2D array, an exception 
   * NotSupportedException is thrown.
   */
  virtual void visitArray2(IArray2DataVisitor* visitor) = 0;

  /*!
   * \brief Apply the visitor to the data.
   *
   * If the data is not a 2D array, an exception 
   * NotSupportedException is thrown.
   *
   * \deprecated This visitor is obsolete because there are no more
   * IMultiArray2 implementations.
   */
  virtual void visitMultiArray2(IMultiArray2DataVisitor* visitor);

  //! \internal
  virtual IDataInternal* _commonInternal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a scalar data item.
 */
class IScalarData
: public IData
{
 public:

  virtual void visit(IDataVisitor* visitor) = 0;
  //! Applies the visitor to the data.
  virtual void visit(IScalarDataVisitor* visitor) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a scalar data item of type \a T
 */
template <class DataType>
class IScalarDataT
: public IScalarData
{
 public:

  typedef IScalarDataT<DataType> ThatClass;

 public:

  //! Data value
  virtual DataType& value() = 0;

  //! Data value
  virtual const DataType& value() const = 0;

  //! Clone the data
  ARCCORE_DEPRECATED_2020("Use cloneTrueRef() instead")
  virtual ThatClass* cloneTrue() = 0;

  //! Clone the data but without elements.
  ARCCORE_DEPRECATED_2020("Use cloneTrueEmpty() instead")
  virtual ThatClass* cloneTrueEmpty() = 0;

  //! Clone the data
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone the data but without elements.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a 1D array data item.
 */
class IArrayData
: public IData
{
 public:

  virtual void visit(IDataVisitor* visitor) = 0;
  //! Applies the visitor to the data.
  virtual void visit(IArrayDataVisitor* visitor) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a 1D array data item of type \a T
 */
template <class DataType>
class IArrayDataT
: public IArrayData
{
 public:

  typedef IArrayDataT<DataType> ThatClass;

 public:

  //! Data value
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual Array<DataType>& value() = 0;

  //! Constant data value
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual const Array<DataType>& value() const = 0;

 public:

  //! Constant view on the data
  virtual ConstArrayView<DataType> view() const = 0;

  //! View on the data
  virtual ArrayView<DataType> view() = 0;

  //! Clone the data
  ARCCORE_DEPRECATED_2020("Use cloneTrueRef() instead")
  virtual ThatClass* cloneTrue() = 0;

  //! Clone the data but without elements.
  ARCCORE_DEPRECATED_2020("Use cloneTrueEmptyRef() instead")
  virtual ThatClass* cloneTrueEmpty() = 0;

  //! Clone the data
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone the data but without elements.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;

  //! \internal
  virtual IArrayDataInternalT<DataType>* _internal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a 2D array data item.
 */
class IArray2Data
: public IData
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a multi 2D array data item.
 * \deprecated This interface is no longer used.
 */
class IMultiArray2Data
: public IData
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a bi-dimensional array data item of type \a T
 */
template <class DataType>
class IArray2DataT
: public IArray2Data
{
 public:

  typedef IArray2DataT<DataType> ThatClass;

  //! Data value
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual Array2<DataType>& value() = 0;

  //! Data value
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual const Array2<DataType>& value() const = 0;

 public:

  //! Constant view on the data
  virtual ConstArray2View<DataType> view() const = 0;

  //! View on the data
  virtual Array2View<DataType> view() = 0;

  //! Clone the data
  ARCCORE_DEPRECATED_2020("Use cloneTrueRef() instead")
  virtual ThatClass* cloneTrue() = 0;

  //! Clone the data but without elements.
  ARCCORE_DEPRECATED_2020("Use cloneTrueEmptyRef() instead")
  virtual ThatClass* cloneTrueEmpty() = 0;

  //! Clone the data
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone the data but without elements.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;

  //! \internal
  virtual IArray2DataInternalT<DataType>* _internal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a multi-sized 2D array data item of type \a T
 * \deprecated This interface is no longer used.
 */
template <class DataType>
class IMultiArray2DataT
: public IMultiArray2Data
{
 public:

  typedef IMultiArray2DataT<DataType> ThatClass;

  //! Data value
  virtual MultiArray2<DataType>& value() = 0;

  //! Data value
  virtual const MultiArray2<DataType>& value() const = 0;

  //! Clone the data
  virtual ThatClass* cloneTrue() = 0;

  //! Clone the data but without elements.
  virtual ThatClass* cloneTrueEmpty() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
