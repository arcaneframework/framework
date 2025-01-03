// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdScalarMeshVariables.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Définition de variables scalaires du maillage pour des tests.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableView.h"

#include "arcane/tests/StdScalarMeshVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> StdScalarMeshVariables<ItemType>::
StdScalarMeshVariables(const MeshHandle& mesh_handle,const String& basestr)
: StdMeshVariables< StdMeshVariableTraits2<ItemType,0> >(mesh_handle,basestr,"Scalar")
, m_nb_display_error(10)
, m_nb_displayed_error(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> StdScalarMeshVariables<ItemType>::
StdScalarMeshVariables(const MeshHandle& mesh_handle,const String& basestr,const String& family_name)
: StdMeshVariables< StdMeshVariableTraits2<ItemType,0> >(mesh_handle,basestr,"Scalar",family_name)
, m_nb_display_error(10)
, m_nb_displayed_error(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void _setReferenceValue(Int64 n,MultiScalarValue& sv)
{
  Real r = Convert::toReal(n);
  float fr = static_cast<float>(r);
  sv.m_byte = (Byte)(n % 255);
  sv.m_real = r;
  sv.m_int64 = n;
  sv.m_int32 = static_cast<Int32>(n);
  sv.m_int16 = static_cast<Int16>(n);
  sv.m_real2 = Real2 (r, r+1);
  sv.m_real2x2 = Real2x2::fromLines (r, r+1., r+2., r+3.);
  sv.m_real3 = Real3 (r, r+1., r+2.0);
  sv.m_real3x3 = Real3x3::fromLines (r, r+1., r+2., r+3., r+4., r+5., r+6., r+7., r+8.);
  sv.m_int8 = static_cast<Int8>(n);
  sv.m_bfloat16 = fr;
  sv.m_float16 = fr;
  sv.m_float32 = fr;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdScalarMeshVariables<ItemType>::
setItemValues(Int64 n, ItemType item)
{
  MultiScalarValue sv;
  
  _setReferenceValue(n,sv);

  this->m_byte[item] = sv.m_byte;
  this->m_real[item] = sv.m_real;
  this->m_int64[item] = sv.m_int64;
  this->m_int32[item] = sv.m_int32;
  this->m_int16[item] = sv.m_int16;
  this->m_real2[item] = sv.m_real2;
  this->m_real2x2[item] = sv.m_real2x2;
  this->m_real3[item] = sv.m_real3;
  this->m_real3x3[item] = sv.m_real3x3;
  this->m_int8[item] = sv.m_int8;
  this->m_bfloat16[item] = sv.m_bfloat16;
  this->m_float16[item] = sv.m_float16;
  this->m_float32[item] = sv.m_float32;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdScalarMeshVariables<ItemType>::
setEvenValues(Integer iteration, const GroupType& group)
{
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    setItemValues(n,item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdScalarMeshVariables<ItemType>::
setOddValues(Integer iteration, const GroupType& group)
{
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    setItemValues(-n,item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdScalarMeshVariables<ItemType>::
setValues(Integer iteration, const GroupType& group)
{
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    setItemValues(n,item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdScalarMeshVariables<ItemType>::
setValuesWithViews(Integer seed, const GroupType& group)
{
  auto out_byte = viewOut(this->m_byte);
  auto out_real = viewOut(this->m_real);
  auto out_int16 = viewOut(this->m_int16);
  auto out_int32 = viewOut(this->m_int32);
  auto out_int64 = viewOut(this->m_int64);
  auto out_real2 = viewOut(this->m_real2);
  auto out_real2x2 = viewOut(this->m_real2x2);
  auto out_real3 = viewOut(this->m_real3);
  auto out_real3x3 = viewOut(this->m_real3x3);
  auto out_int8 = viewOut(this->m_int8);
  auto out_bfloat16 = viewOut(this->m_bfloat16);
  auto out_float16 = viewOut(this->m_float16);
  auto out_float32 = viewOut(this->m_float32);

  MultiScalarValue sv;

  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + seed*7);
    _setReferenceValue(n,sv);

    out_byte[item] = sv.m_byte;
    out_real[item] = sv.m_real;
    out_int64[item] = sv.m_int64;
    out_int32[item] = sv.m_int32;
    out_int16[item] = sv.m_int16;
    out_real2[item] = sv.m_real2;
    out_real2x2[item] = sv.m_real2x2;
    out_real3[item] = sv.m_real3;
    out_real3x3[item] = sv.m_real3x3;
    out_int8[item] = sv.m_int8;
    out_bfloat16[item] = sv.m_bfloat16;
    out_float16[item] = sv.m_float16;
    out_float32[item] = sv.m_float32;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
_writeError(ITraceMng* tm,const char* type_name,const DataType& value,
            const DataType& expected_value,const Item& item)
{
  tm->info() << "Bad scalar value type=" << type_name
             << " value=" << value
             << " expected=" << expected_value
             << " item=" << ItemPrinter(item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define CHECK_VALUE(type_name)\
    if (ref_sv.m_##type_name!=current_sv.m_##type_name){\
      ++nb_error;\
      ++m_nb_displayed_error;\
      if (m_nb_displayed_error<m_nb_display_error){\
        _writeError(tm,#type_name,current_sv.m_##type_name,ref_sv.m_##type_name,item);\
      }\
    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer
StdScalarMeshVariables<ItemType>::
_checkValue(ItemType item,const MultiScalarValue& ref_sv,
            const MultiScalarValue& current_sv)
{
  ITraceMng* tm = this->m_mesh_handle.mesh()->traceMng();
  Integer nb_error = 0;

  CHECK_VALUE(byte);
  CHECK_VALUE(real);
  CHECK_VALUE(int64);
  CHECK_VALUE(int32);
  CHECK_VALUE(int16);
  CHECK_VALUE(real2);
  CHECK_VALUE(real2x2);
  CHECK_VALUE(real3);
  CHECK_VALUE(real3x3);
  CHECK_VALUE(int8);
  CHECK_VALUE(bfloat16);
  CHECK_VALUE(float16);
  CHECK_VALUE(float32);
  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdScalarMeshVariables<ItemType>::
_checkItemValues(Integer seed, ItemType item, const MultiScalarValue& current_sv)
{
  MultiScalarValue ref_sv;

  Int64 n = 1 + (item.uniqueId().asInt64() + seed*7);
  _setReferenceValue(n,ref_sv);

  return _checkValue(item,ref_sv,current_sv);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdScalarMeshVariables<ItemType>::
checkGhostValuesOddOrEven(Integer iteration, const GroupType& group)
{
  Integer nb_error = 0;
  MultiScalarValue current_sv;
  m_nb_displayed_error = 0;
  
  auto check_value = [&](ItemType item)
  {
    MultiScalarValue ref_sv;
    
    Int64 uid = item.uniqueId();
    Int64 n = 1 + (uid + iteration*7);
    
    if (uid % 2 != 0) {
      n = -n;
    }
    
    _setReferenceValue(n, ref_sv);
    
    return _checkValue(item,ref_sv,current_sv);
  };
  
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();

    if (!item.isOwn()){
      current_sv.m_byte = this->m_byte[item];
      current_sv.m_real = this->m_real[item];
      current_sv.m_int64 = this->m_int64[item];
      current_sv.m_int32 = this->m_int32[item];
      current_sv.m_int16 = this->m_int16[item];
      current_sv.m_real2 = this->m_real2[item];
      current_sv.m_real2x2 = this->m_real2x2[item];
      current_sv.m_real3 = this->m_real3[item];
      current_sv.m_real3x3 = this->m_real3x3[item];
      current_sv.m_int8 = this->m_int8[item];
      current_sv.m_bfloat16 = this->m_bfloat16[item];
      current_sv.m_float16 = this->m_float16[item];
      current_sv.m_float32 = this->m_float32[item];
      nb_error += check_value(item);
    }
  }

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdScalarMeshVariables<ItemType>::
checkValues (Integer iteration, const GroupType& group)
{
  Integer nb_error = 0;
  MultiScalarValue current_sv;
  m_nb_displayed_error = 0;
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();

    current_sv.m_byte = this->m_byte[item];
    current_sv.m_real = this->m_real[item];
    current_sv.m_int64 = this->m_int64[item];
    current_sv.m_int32 = this->m_int32[item];
    current_sv.m_int16 = this->m_int16[item];
    current_sv.m_real2 = this->m_real2[item];
    current_sv.m_real2x2 = this->m_real2x2[item];
    current_sv.m_real3 = this->m_real3[item];
    current_sv.m_real3x3 = this->m_real3x3[item];
    current_sv.m_int8 = this->m_int8[item];
    current_sv.m_bfloat16 = this->m_bfloat16[item];
    current_sv.m_float16 = this->m_float16[item];
    current_sv.m_float32 = this->m_float32[item];

    nb_error += _checkItemValues(iteration,item,current_sv);
  }

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdScalarMeshVariables<ItemType>::
checkValuesWithViews(Integer seed, const GroupType& group)
{
  auto in_byte = viewIn(this->m_byte);
  auto in_real = viewIn(this->m_real);
  auto in_int16 = viewIn(this->m_int16);
  auto in_int32 = viewIn(this->m_int32);
  auto in_int64 = viewIn(this->m_int64);
  auto in_real2 = viewIn(this->m_real2);
  auto in_real2x2 = viewIn(this->m_real2x2);
  auto in_real3 = viewIn(this->m_real3);
  auto in_real3x3 = viewIn(this->m_real3x3);
  auto in_int8 = viewIn(this->m_int8);
  auto in_bfloat16 = viewIn(this->m_bfloat16);
  auto in_float16 = viewIn(this->m_float16);
  auto in_float32 = viewIn(this->m_float32);

  MultiScalarValue current_sv;

  m_nb_displayed_error = 0;

  Integer nb_error = 0;
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();

    current_sv.m_byte = in_byte[item];
    current_sv.m_real = in_real[item];
    current_sv.m_int64 = in_int64[item];
    current_sv.m_int32 = in_int32[item];
    current_sv.m_int16 = in_int16[item];
    current_sv.m_real2 = in_real2[item];
    current_sv.m_real2x2 = in_real2x2[item];
    current_sv.m_real3 = in_real3[item];
    current_sv.m_real3x3 = in_real3x3[item];
    current_sv.m_int8 = in_int8[item];
    current_sv.m_bfloat16 = in_bfloat16[item];
    current_sv.m_float16 = in_float16[item];
    current_sv.m_float32 = in_float32[item];

    nb_error += _checkItemValues(seed,item,current_sv);
  }

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdScalarMeshVariables<ItemType>::
checkReplica()
{
  Integer nb_error = 0;
  Integer max_print = 10;

  nb_error += this->m_byte.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_int64.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_int32.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_int16.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real2.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real3.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real2x2.checkIfSameOnAllReplica(max_print);
  nb_error += this->m_real3x3.checkIfSameOnAllReplica(max_print);
  //nb_error += this->m_int8.checkIfSameOnAllReplica(max_print);
  //nb_error += this->m_bfloat16.checkIfSameOnAllReplica(max_print);
  //nb_error += this->m_float16.checkIfSameOnAllReplica(max_print);
  //nb_error += this->m_float32.checkIfSameOnAllReplica(max_print);

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class StdScalarMeshVariables<Node>;
template class StdScalarMeshVariables<Edge>;
template class StdScalarMeshVariables<Face>;
template class StdScalarMeshVariables<Cell>;
template class StdScalarMeshVariables<Particle>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
