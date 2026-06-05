// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataStorageFactory.h                                        (C) 2000-2021 */
/*                                                                           */
/* Data container factory.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASTORAGEFACTORY_H
#define ARCANE_IMPL_DATASTORAGEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IDataStorageFactory.h"
#include "arcane/core/IData.h"
#include "arcane/core/IDataFactory.h"
#include "arcane/core/IDataFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Data container factory.
 */
class ARCANE_IMPL_EXPORT AbstractDataStorageFactory
: public IDataStorageFactory
{
 public:

  explicit AbstractDataStorageFactory(const DataStorageTypeInfo& dsti)
  : m_storage_type_info(dsti)
  {}

 public:

  //! Information about the created container type
  DataStorageTypeInfo storageTypeInfo() override
  {
    return m_storage_type_info;
  }

 private:

  DataStorageTypeInfo m_storage_type_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Data container factory.
 */
template <typename DataType> class DataStorageFactory
: public AbstractDataStorageFactory
{
 public:

  explicit DataStorageFactory(const DataStorageTypeInfo& dsti)
  : AbstractDataStorageFactory(dsti)
  {}

 public:

  //! Create simple data of a type.
  Ref<IData> createSimpleDataRef(const DataStorageBuildInfo& dsbi) override
  {
    IData* d = new DataType(dsbi);
    return makeRef(d);
  }

  //! Registers a factory for the data \a DataType in \a dfm
  static void registerDataFactory(IDataFactoryMng* dfm)
  {
    using DataContainerType = DataType;
    ITraceMng* trace = dfm->traceMng();
    DataStorageTypeInfo t = DataContainerType::staticStorageTypeInfo();
    const bool print_info = false;
    if (print_info && trace)
      trace->info() << "TYPE=" << t.basicDataType()
                    << " nb_basic=" << t.nbBasicElement()
                    << " dimension=" << t.dimension()
                    << " multi_tag=" << t.multiTag()
                    << " full_name=" << t.fullName()
                    << "\n";
    IDataStorageFactory* sf = new DataStorageFactory<DataContainerType>(t);
    dfm->registerDataStorageFactory(makeRef(sf));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
