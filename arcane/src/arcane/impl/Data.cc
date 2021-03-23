// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Data.cc                                                     (C) 2000-2020 */
/*                                                                           */
/* Classes de base d'une donnée.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/IDataFactory.h"
#include "arcane/IApplication.h"
#include "arcane/IDataFactoryMng.h"
#include "arcane/IApplication.h"

#include "arcane/datatype/DataStorageTypeInfo.h"

#include "arcane/impl/StringScalarData.h"
#include "arcane/impl/StringArrayData.h"
#include "arcane/impl/ScalarData.h"
#include "arcane/impl/ArrayData.h"
#include "arcane/impl/Array2Data.h"
#include "arcane/impl/DataStorageFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataContainerType> inline void
_registerOneData(IDataFactoryMng* dfm)
{
  IDataFactory* df = dfm->deprecatedOldFactory();
  ITraceMng* trace = df->application()->traceMng();
  df->registerData(new DataContainerType(trace));
  DataStorageTypeInfo t = DataContainerType::staticStorageTypeInfo();
  bool print_info = false;
  if (print_info)
    std::cout << "TYPE=" << t.basicDataType()
              << " nb_basic=" << t.nbBasicElement()
              << " dimension=" << t.dimension()
              << " multi_tag=" << t.multiTag()
              << " full_name=" << t.fullName()
              << "\n";
  IDataStorageFactory* sf = new DataStorageFactory<DataContainerType>(t);
  dfm->registerDataStorageFactory(makeRef(sf));
}

namespace
{
void
_registerSimpleData(IDataFactoryMng* dfm)
{
  // Enregistre les types de donnée standard.

  _registerOneData<ScalarDataT<Byte>>(dfm);
  _registerOneData<ScalarDataT<Real>>(dfm);
  _registerOneData<ScalarDataT<Int16>>(dfm);
  _registerOneData<ScalarDataT<Int32>>(dfm);
  _registerOneData<ScalarDataT<Int64>>(dfm);
  _registerOneData<ScalarDataT<Real2>>(dfm);
  _registerOneData<ScalarDataT<Real3>>(dfm);
  _registerOneData<ScalarDataT<Real2x2>>(dfm);
  _registerOneData<ScalarDataT<Real3x3>>(dfm);
  _registerOneData<StringScalarData>(dfm);

  _registerOneData<ArrayDataT<Byte>>(dfm);
  _registerOneData<ArrayDataT<Real>>(dfm);
  _registerOneData<ArrayDataT<Int16>>(dfm);
  _registerOneData<ArrayDataT<Int32>>(dfm);
  _registerOneData<ArrayDataT<Int64>>(dfm);
  _registerOneData<ArrayDataT<Real2>>(dfm);
  _registerOneData<ArrayDataT<Real3>>(dfm);
  _registerOneData<ArrayDataT<Real2x2>>(dfm);
  _registerOneData<ArrayDataT<Real3x3>>(dfm);
  _registerOneData<StringArrayData>(dfm);

  _registerOneData<Array2DataT<Byte>>(dfm);
  _registerOneData<Array2DataT<Real>>(dfm);
  _registerOneData<Array2DataT<Int16>>(dfm);
  _registerOneData<Array2DataT<Int32>>(dfm);
  _registerOneData<Array2DataT<Int64>>(dfm);
  _registerOneData<Array2DataT<Real2>>(dfm);
  _registerOneData<Array2DataT<Real3>>(dfm);
  _registerOneData<Array2DataT<Real2x2>>(dfm);
  _registerOneData<Array2DataT<Real3x3>>(dfm);
}
}

extern "C++" void
arcaneRegisterSimpleData(IDataFactoryMng* dfm)
{
  _registerSimpleData(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
