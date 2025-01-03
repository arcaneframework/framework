// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataOperation.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Opération sur une donnée.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/DataOperation.h"

#include "arcane/utils/FatalErrorException.h"

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

extern "C++" ARCANE_IMPL_EXPORT IDataOperation*
arcaneCreateDataOperation(Parallel::eReduceType rt)
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
