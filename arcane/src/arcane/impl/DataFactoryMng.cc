// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataFactoryMng.cc                                           (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire de fabriques de données.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/DataFactoryMng.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IApplication.h"
#include "arcane/IData.h"
#include "arcane/IDataStorageFactory.h"
#include "arcane/MathUtils.h"

#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

#include "arcane/impl/SerializedData.h"
#include "arcane/impl/DataFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
arcaneRegisterSimpleData(IDataFactoryMng* df);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataFactoryMng::
DataFactoryMng(IApplication* app)
: m_application(app)
, m_old_factory(new DataFactory(app))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataFactoryMng::
~DataFactoryMng()
{
  delete m_old_factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IData> DataFactoryMng::
createSimpleDataRef(const String& storage_type,const DataStorageBuildInfo& build_info)
{
  auto x = m_factories.find(storage_type);
  if (x==m_factories.end())
    ARCANE_FATAL("Can not find data factory named={0}",storage_type);
  // Positionne les valeurs de \a build_info qui ne sont pas encore
  // initialisées.
  DataStorageBuildInfo b = build_info;
  if (!b.memoryAllocator())
    b.setMemoryAllocator(platform::getDefaultDataAllocator());

  return x->second->createSimpleDataRef(b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> DataFactoryMng::
createSerializedDataRef(eDataType data_type,Int64 memory_size,
                        Integer nb_dim,Int64 nb_element,Int64 nb_base_element,
                        bool is_multi_size,Int64ConstArrayView dimensions)
{
  auto* x = new SerializedData(data_type,memory_size,nb_dim,nb_element,
                               nb_base_element,is_multi_size,dimensions);
  return makeRef(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> DataFactoryMng::
createEmptySerializedDataRef()
{
  return makeRef(new SerializedData());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataFactoryMng::
build()
{
  arcaneRegisterSimpleData(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* DataFactoryMng::
traceMng() const
{
  return m_application->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataFactoryMng::
registerDataStorageFactory(Ref<IDataStorageFactory> factory)
{
  DataStorageTypeInfo t = factory->storageTypeInfo();
  m_factories.insert(std::make_pair(t.fullName(),factory));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataFactoryMng>
arcaneCreateDataFactoryMngRef(IApplication* app)
{
  Ref<IDataFactoryMng> df(Arccore::makeRef(new DataFactoryMng(app)));
  df->build();
  return df;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

