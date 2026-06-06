// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataFactoryMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface of the data factory manager.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAFACTORYMNG_H
#define ARCANE_CORE_IDATAFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of the data factory manager.
 *
 * \note This interface replaces 'IDataFactory'
 *
 * This interface allows registering factories to create 'IData'
 * (via registerDataStorageFactory()) and to construct the correct IData instance
 * by calling createSimpleDataRef().
 */
class IDataFactoryMng
{
 public:

  virtual ~IDataFactoryMng() = default;

 public:

  //! Builds the instance
  virtual void build() = 0;

  //! Associated trace manager
  virtual ITraceMng* traceMng() const = 0;

  //! Registers the factory \a factory.
  virtual void registerDataStorageFactory(Ref<IDataStorageFactory> factory) = 0;

  /*
   * \brief Creates a data object.
   *
   * The factory used to create the data is derived from a DataStorageTypeInfo instance using
   * the DataStorageTypeInfo::fullName() method.
   */
  virtual Ref<IData>
  createSimpleDataRef(const String& storage_type, const DataStorageBuildInfo& build_info) = 0;

  /*!
   * \brief Creates an operation performing a reduction of type \a rt.
   */
  virtual IDataOperation* createDataOperation(Parallel::eReduceType rt) = 0;

  /*!
   * \brief Creates serialized data.
   *
   * The arrays \a dimensions and \a values are not duplicated and must not
   * be modified while the serialized object is in use.
   *
   * The \a data_type must be one of the types: \a DT_Byte, \a DT_Int16, \a DT_Int32,
   * \a DT_Int64, or DT_Real.
   *
   * \deprecated Use arcaneCreateSerializedDataRef() instead
   */
  ARCCORE_DEPRECATED_2021("Use global method arcaneCreateSerializedDataRef() instead")
  virtual Ref<ISerializedData>
  createSerializedDataRef(eDataType data_type, Int64 memory_size,
                          Integer nb_dim, Int64 nb_element,
                          Int64 nb_base_element, bool is_multi_size,
                          Int64ConstArrayView dimensions) = 0;

  /*!
   * \brief Creates serialized data.
   *
   * The serialized data is empty. It can only be used after calling an
   * ISerializedData::serialize() in ISerializer::ModePut mode.
   *
   * \deprecated Use arcaneCreateEmptySerializedDataRef() instead.
   */
  ARCCORE_DEPRECATED_2021("Use global method arcaneCreateEmptySerializedDataRef() instead")
  virtual Ref<ISerializedData> createEmptySerializedDataRef() = 0;

  //! Retrieves the old factory (obsolete)
  ARCCORE_DEPRECATED_2021("Do not use deprecated interface 'IDataFactory'")
  virtual IDataFactory* deprecatedOldFactory() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
