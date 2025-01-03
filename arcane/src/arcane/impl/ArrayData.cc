// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayData.cc                                                (C) 2000-2024 */
/*                                                                           */
/* Donnée du type 'Array'.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/ArrayData.h"

#include "arcane/utils/NumericTypes.h"

#include "arcane/impl/DataStorageFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerArrayDataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<ArrayDataT<Byte>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Float16>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<BFloat16>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Float32>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int8>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int16>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int32>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Int64>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real2>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real3>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real2x2>>::registerDataFactory(dfm);
  DataStorageFactory<ArrayDataT<Real3x3>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
