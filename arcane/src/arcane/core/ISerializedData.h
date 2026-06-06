// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializedData.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface of a serialized data.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERIALIZEDDATA_H
#define ARCANE_CORE_ISERIALIZEDDATA_H
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
 * \internal
 * \brief Interface of a serialized data.
 *
 * A data (IData) is serialized into an instance of this class.
 *
 * Regardless of the data type, the serialized type must be
 * a base type among the following: DT_Byte, DT_Int16, DT_Int32, DT_Int64, DT_Real.
 *
 * An instance of this class is only valid as long as the reference data
 * is not modified.
 * 
 * To serialize a data \a data for writing:
 * \code
 * IData* data = ...;
 * ISerializedData* sdata = data->createSerializedData();
 * // sdata->constBytes() contains the serialized data.
 * Span<const Byte> buf(sdata->constBytes());
 * std::cout.write(reinterpret_cast<const char*>(buf.data()),buf.size());
 * \endcode
 *
 * To serialize a data \a data for reading:
 * \code
 * IData* data = ...
 * // Create an instance of an ISerializedData.
 * Ref<ISerializedData> sdata = arcaneCreateSerializedDataRef(...);
 * data->allocateBufferForSerializedData(sdata);
 * // Fills sdata->writableBytes() from your source
 * Span<Byte> buf(sdata->writableBytes());
 * std::cin.read(reinterpret_cast<char*>(buf.data()),buf.size());
 * // Assigns the value to \a data
 * data->assignSerializedData(sdata);
 * \endcode
 */
class ARCANE_CORE_EXPORT ISerializedData
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  //! Frees resources
  virtual ~ISerializedData() = default;

 public:

  //! Data type
  virtual eDataType baseDataType() const = 0;

  //! Dimension. 0 for a scalar, 1 for a 1D array, ...
  virtual Integer nbDimension() const = 0;

  //! Number of elements
  virtual Int64 nbElement() const = 0;

  //! Number of base elements
  virtual Int64 nbBaseElement() const = 0;

  //! Indicates if it is a multi-size array. (only relevant if nbDimension()>1)
  virtual bool isMultiSize() const = 0;

  //! Indicates the number of bytes that must be allocated to store or read the data
  virtual Int64 memorySize() const = 0;

  //! Array containing the number of elements for each dimension
  virtual Int64ConstArrayView extents() const = 0;

  //! Shape of the array associated with the data
  virtual ArrayShape shape() const = 0;

  //! Serialized values.
  virtual Span<const Byte> constBytes() const = 0;

  /*!
   * \brief View of the serialized values
   *
   * \warning This method returns a non-empty view only if one
   * has called allocateMemory() or setWritableBytes(Span<Byte>) beforehand.
   */
  virtual Span<Byte> writableBytes() = 0;

  /*!
   * \brief Positions the serialized values.
   *
   * The view \a bytes must remain valid as long as this instance is used.
   */
  virtual void setWritableBytes(Span<Byte> bytes) = 0;

  /*!
   * \brief Positions the serialized values for reading
   *
   * The view \a bytes must remain valid as long as this instance is used.
   */
  virtual void setConstBytes(Span<const Byte> bytes) = 0;

  /*!
   * \brief Allocates an array to hold the serialized elements.
   *
   * After calling this method, it is possible to retrieve a
   * view of the serialized values via writableBytes() or constBytes().
   */
  virtual void allocateMemory(Int64 size) = 0;

 public:

  /*!
   * \brief Serialize the data for reading or writing
   */
  virtual void serialize(ISerializer* buffer) = 0;

  /*!
   * \brief Serialize the data for reading
   */
  virtual void serialize(ISerializer* buffer) const = 0;

 public:

  /*!
   * \brief Compute a hash key on this data.
   *
   * The key is added to \a output. The length of the key depends
   * on the algorithm used.
   */
  virtual void computeHash(IHashAlgorithm* algo, ByteArray& output) const = 0;

 public:

  /*!
   * \brief Serialized values.
   * \deprecated Use bytes() instead.
   */
  ARCANE_DEPRECATED_2018_R("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual ByteConstArrayView buffer() const = 0;

  /*!
   * \brief Serialized values.
   * \deprecated Use bytes() instead.
   */
  ARCANE_DEPRECATED_2018_R("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual ByteArrayView buffer() = 0;

  //! Serialized values.
  ARCCORE_DEPRECATED_2021("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual Span<const Byte> bytes() const = 0;

  /*!
   * \brief Positions the serialized values.
   *
   * The array \a buffer must not be modified
   * as long as this instance is used.
   * \deprecated Use setBytes() instead.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setWritableBytes()' instead")
  virtual void setBuffer(ByteArrayView buffer) = 0;

  /*!
   * \brief Positions the serialized values.
   *
   * The array \a buffer must not be modified
   * as long as this instance is used.
   * \deprecated Use setBytes() instead.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setConstBytes()' instead")
  virtual void setBuffer(ByteConstArrayView buffer) = 0;

  /*!
   * \brief Positions the serialized values.
   *
   * The array \a bytes must not be modified
   * as long as this instance is used.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setWritableBytes()' instead")
  virtual void setBytes(Span<Byte> bytes) = 0;

  /*!
   * \brief Positions the serialized values.
   *
   * The array \a bytes must not be modified
   * as long as this instance is used.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setConstBytes()' instead")
  virtual void setBytes(Span<const Byte> bytes) = 0;

  /*!
   * \brief Serialized values
   *
   * \warning This method returns a non-empty view only if one
   * has called setBytes(Span<Byte>) or allocateMemory().
   */
  ARCCORE_DEPRECATED_2021("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual Span<Byte> bytes() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates serialized data.
 *
 * The arrays \a dimensions and \a values are not duplicated and must not
 * be modified as long as the serialized object is used.
 *
 * The type \a data_type must be a type among \a DT_Byte, \a DT_Int16, \a DT_Int32,
 * \a DT_Int64 or DT_Real.
 */
extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type, Int64 memory_size,
                              Integer nb_dim, Int64 nb_element, Int64 nb_base_element,
                              bool is_multi_size, Int64ConstArrayView dimensions);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates serialized data.
 *
 * The arrays \a dimensions and \a values are not duplicated and must not
 * be modified as long as the serialized object is used.
 *
 * The type \a data_type must be a type among \a DT_Byte, \a DT_Int16, \a DT_Int32,
 * \a DT_Int64 or DT_Real.
 */
extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type, Int64 memory_size,
                              Integer nb_dim, Int64 nb_element, Int64 nb_base_element,
                              bool is_multi_size, Int64ConstArrayView dimensions,
                              const ArrayShape& shape);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates serialized data.
 *
 * The serialized data is empty. It can only be used after a
 * call to ISerializedData::serialize() in ISerializer::ModePut mode.
 */
extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateEmptySerializedDataRef();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
