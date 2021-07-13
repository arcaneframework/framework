// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataFactory.cc                                              (C) 2000-2021 */
/*                                                                           */
/* Fabrique de données.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Deleter.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IApplication.h"
#include "arcane/IData.h"
#include "arcane/MathUtils.h"

#include "arcane/impl/DataFactory.h"
#include "arcane/impl/DataOperation.h"
#include "arcane/impl/SerializedData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SumDataOperator
{
public:
  template<typename DataType>
  DataType operator()(const DataType& input1,const DataType& input2)
  {
    return (DataType)(input1 + input2);
  }
};

class MinusDataOperator
{
public:
  template<typename DataType>
  DataType operator()(const DataType& input1,const DataType& input2)
  {
    return input1 - input2;
  }
};

class MaxDataOperator
{
public:
  template<typename DataType>
  DataType operator()(const DataType& input1,const DataType& input2)
  {
    return math::max(input1,input2);
  }
};

class MinDataOperator
{
public:
  template<typename DataType>
  DataType operator()(const DataType& input1,const DataType& input2)
  {
    return math::min(input1,input2);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IDataFactory*
arcaneCreateDataFactory(IApplication* sm);
extern "C++" void
arcaneRegisterSimpleData(IDataFactory* df);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataFactory::
DataFactory(IApplication* sm)
: m_application(sm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataFactory::
~DataFactory()
{
  m_data.each(Deleter());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IData* DataFactory::
registerData(IData* data)
{
  ITraceMng* trace = m_application->traceMng();
  // Recherche si une donnée de même type est déjà enregistrée et
  // la stocke dans \a old_data
  IData* old_data = 0;
  for( DataArray::Enumerator i(m_data); ++i; ){
    IData* d = *i;
    if (d->dataType()==data->dataType() && d->dimension()==data->dimension()
        && d->multiTag()==data->multiTag()){
      old_data = d;
      trace->debug() << "Fabrique déjà présente (" << d << ") supprimée";
      break;
    }
  }
  if (old_data)
    m_data.remove(old_data);
  m_data.add(data);
  return old_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IData> DataFactory::
createSimpleDataRef(eDataType data_type,Integer dimension,Integer multi_tag)
{
  for( DataArray::Enumerator i(m_data); ++i; ){
    IData* d = *i;
    if (d->dataType()==data_type && d->dimension()==dimension && d->multiTag()==multi_tag)
      return d->cloneRef();
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataOperation* DataFactory::
createDataOperation(Parallel::eReduceType rt)
{
  switch(rt){
  case Parallel::ReduceSum:
    return new DataOperationT< SumDataOperator >();
    break;
  case Parallel::ReduceMax:
    return new DataOperationT< MaxDataOperator >();
    break;
  case Parallel::ReduceMin:
    return new DataOperationT< MinDataOperator >();
    break;
  }
  ARCANE_FATAL("Operation not found");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataFactory::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataFactory*
arcaneCreateDataFactory(IApplication* sm)
{
  IDataFactory* df = new DataFactory(sm);
  df->build();
  return df;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

