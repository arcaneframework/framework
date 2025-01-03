// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Data.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Donnée du type 'Array2'.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/Array2Data.h"

#include "arcane/utils/NumericTypes.h"

#include "arcane/impl/DataStorageFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
registerArray2DataFactory(IDataFactoryMng* dfm)
{
  DataStorageFactory<Array2DataT<Byte>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<BFloat16>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Float16>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Float32>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int8>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int16>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int32>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Int64>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real2>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real3>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real2x2>>::registerDataFactory(dfm);
  DataStorageFactory<Array2DataT<Real3x3>>::registerDataFactory(dfm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
