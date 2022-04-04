// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataStorageFactory.h                                        (C) 2000-2021 */
/*                                                                           */
/* Fabrique de conteneur d'une donnée.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASTORAGEFACTORY_H
#define ARCANE_IMPL_DATASTORAGEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/IDataStorageFactory.h"
#include "arcane/IData.h"
#include "arcane/IDataFactory.h"
#include "arcane/IDataFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique de conteneur d'une donnée.
 */
class ARCANE_IMPL_EXPORT AbstractDataStorageFactory
: public IDataStorageFactory
{
 public:
  explicit AbstractDataStorageFactory(const DataStorageTypeInfo& dsti)
  : m_storage_type_info(dsti){}
 public:
  //! Informations sur le type de conteneur créé
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
 * \brief Fabrique de conteneur d'une donnée.
 */
template<typename DataType> class DataStorageFactory
: public AbstractDataStorageFactory
{
 public:

  explicit DataStorageFactory(const DataStorageTypeInfo& dsti)
  : AbstractDataStorageFactory(dsti){}

 public:

  //! Créé une donnée d'un type simple.
  Ref<IData> createSimpleDataRef(const DataStorageBuildInfo& dsbi) override
  {
    IData* d = new DataType(dsbi);
    return makeRef(d);
  }

  //! Enregistre dans \a dfm une fabrique pour la donnée \a DataType
  static void registerDataFactory(IDataFactoryMng* dfm)
  {
    using DataContainerType = DataType;
    ITraceMng* trace = dfm->traceMng();
    DataStorageTypeInfo t = DataContainerType::staticStorageTypeInfo();
    bool print_info = false;
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
