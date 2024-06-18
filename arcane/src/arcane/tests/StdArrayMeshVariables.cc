// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdArrayMeshVariables.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Définition de variables tableaux du maillage pour des tests.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/IMesh.h"
#include "arcane/ItemGroup.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ItemPrinter.h"

#include "arcane/tests/StdArrayMeshVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> StdArrayMeshVariables<ItemType>::
StdArrayMeshVariables(const MeshHandle& mesh_handle,const String& basestr)
: StdMeshVariables< StdMeshVariableTraits2<ItemType,1> >(mesh_handle,basestr,"Array")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> StdArrayMeshVariables<ItemType>::
StdArrayMeshVariables(const MeshHandle& mesh_handle,const String& basestr,
                      const String& family_name)
: StdMeshVariables< StdMeshVariableTraits2<ItemType,1> >(mesh_handle,basestr,"Array",family_name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdArrayMeshVariables<ItemType>::
initialize()
{
  // Attention: si on change une taille il faut changer la valeur
  // de nbValuePerItem().
  this->m_byte.resize(8);
  this->m_real.resize(3);
  this->m_int64.resize(2);
  this->m_int32.resize(3);
  this->m_int16.resize(5);
  this->m_real2.resize(4);
  this->m_real2x2.resize(6);
  this->m_real3.resize(5);
  this->m_real3x3.resize(7);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdArrayMeshVariables<ItemType>::
nbValuePerItem()
{
  return (8+3+2+3+5+4+6+5+7);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
_writeError(ITraceMng* tm,const char* type_name,const DataType& value,
            const DataType& expected_value,const Item& item)
{
  tm->info() << "Bad array value type=" << type_name
             << " value=" << value
             << " expected=" << expected_value
             << " item=" << ItemPrinter(item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define CHECK_VALUE(type_name)                 \
  if (!math::isNearlyEqual(array_##type_name[index],expected)){	\
    ++nb_error;\
    if(nb_error<nb_display_error)\
      _writeError(tm,#type_name,array_##type_name[index],expected,item); \
  }

template<class ItemType> Integer StdArrayMeshVariables<ItemType>::
checkValues (Integer iteration, const GroupType& group)
{
  Integer nb_error = 0;
  ITraceMng* tm = this->m_mesh_handle.mesh()->traceMng();
  Integer nb_display_error = 10;
  ENUMERATE_ITEM(iter,group){
    const ItemType& item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    Real r = Convert::toReal(n);

    ByteConstArrayView array_byte = this->m_byte[item];
    for (Integer index=0, size=array_byte.size(); index<size; ++index){
      Byte expected = (Byte)((n+index) % 255);
      CHECK_VALUE(byte);
    }

    RealConstArrayView array_real = this->m_real[item];
    for (Integer index=0, size=array_real.size(); index<size; ++index){
      Real expected = r + Convert::toReal(index);
      CHECK_VALUE(real);
    }

    Int64ConstArrayView array_int64 = this->m_int64[item];
    for(Integer index=0, size=array_int64.size(); index<size; ++index){
      Int64 expected = (Int64)(n + index);
      CHECK_VALUE(int64);
    }

    Int32ConstArrayView array_int32 = this->m_int32[item];
    for(Integer index=0, size=array_int32.size(); index<size; ++index){
      Int32 expected = (Int32)(n + 1 + index);
      CHECK_VALUE(int32);
    }

    Int16ConstArrayView array_int16 = this->m_int16[item];
    for(Integer index=0, size=array_int16.size(); index<size; ++index){
      Int16 expected = (Int16)(n + 2 + index);
      CHECK_VALUE(int16);
    }

    Real2ConstArrayView array_real2 = this->m_real2[item];
    for (Integer index=0, size=array_real2.size(); index<size; ++index){
      Real2 expected;
      expected = (r + (Real)(index)) / 2.0;
      CHECK_VALUE(real2);
    }

    Real2x2ConstArrayView array_real2x2 = this->m_real2x2[item];
    for( Integer index=0, size=array_real2x2.size(); index<size; ++index){
      Real2x2 expected;
      expected = (r + (Real)(index)) / 4.0;
      CHECK_VALUE(real2x2);
    }

    Real3ConstArrayView array_real3 = this->m_real3[item];
    for (Integer index=0, size=array_real3.size(); index<size; ++index){
      Real3 expected;
      expected = (r + static_cast<Real>(index)) / 3.0;
      CHECK_VALUE(real3);
    }

    Real3x3ConstArrayView array_real3x3 = this->m_real3x3[item];
    for (Integer index=0, size=array_real3x3.size(); index<size; ++index){
      Real3x3 expected;
      expected = (r + static_cast<Real>(index)) / 9.0;
      CHECK_VALUE(real3x3);
    }
  }

  return nb_error;
}

template<class ItemType> Integer StdArrayMeshVariables<ItemType>::
checkGhostValuesOddOrEven (Integer iteration, const GroupType& group)
{
  Integer nb_error = 0;
  ITraceMng* tm = this->m_mesh_handle.mesh()->traceMng();
  Integer nb_display_error = 10;
  ENUMERATE_ITEM(iter,group){
    const ItemType& item = (*iter).itemBase();
    
    if (item.isOwn()){
      continue;
    }
    
    Int64 uid = item.uniqueId();
    Int64 n = 1 + (uid + iteration*7);
    
    if (uid % 2 != 0) { // impair
      n = -n;
    }
    
    Real r = Convert::toReal(n);

    ByteConstArrayView array_byte = this->m_byte[item];
    for (Integer index=0, size=array_byte.size(); index<size; ++index){
      Byte expected = (Byte)((n+index) % 255);
      CHECK_VALUE(byte);
    }

    RealConstArrayView array_real = this->m_real[item];
    for (Integer index=0, size=array_real.size(); index<size; ++index){
      Real expected = r + Convert::toReal(index);
      CHECK_VALUE(real);
    }

    Int64ConstArrayView array_int64 = this->m_int64[item];
    for(Integer index=0, size=array_int64.size(); index<size; ++index){
      Int64 expected = (Int64)(n + index);
      CHECK_VALUE(int64);
    }

    Int32ConstArrayView array_int32 = this->m_int32[item];
    for(Integer index=0, size=array_int32.size(); index<size; ++index){
      Int32 expected = (Int32)(n + 1 + index);
      CHECK_VALUE(int32);
    }

    Int16ConstArrayView array_int16 = this->m_int16[item];
    for(Integer index=0, size=array_int16.size(); index<size; ++index){
      Int16 expected = (Int16)(n + 2 + index);
      CHECK_VALUE(int16);
    }

    Real2ConstArrayView array_real2 = this->m_real2[item];
    for (Integer index=0, size=array_real2.size(); index<size; ++index){
      Real2 expected;
      expected = (r + (Real)(index)) / 2.0;
      CHECK_VALUE(real2);
    }

    Real2x2ConstArrayView array_real2x2 = this->m_real2x2[item];
    for( Integer index=0, size=array_real2x2.size(); index<size; ++index){
      Real2x2 expected;
      expected = (r + (Real)(index)) / 4.0;
      CHECK_VALUE(real2x2);
    }

    Real3ConstArrayView array_real3 = this->m_real3[item];
    for (Integer index=0, size=array_real3.size(); index<size; ++index){
      Real3 expected;
      expected = (r + static_cast<Real>(index)) / 3.0;
      CHECK_VALUE(real3);
    }

    Real3x3ConstArrayView array_real3x3 = this->m_real3x3[item];
    for (Integer index=0, size=array_real3x3.size(); index<size; ++index){
      Real3x3 expected;
      expected = (r + static_cast<Real>(index)) / 9.0;
      CHECK_VALUE(real3x3);
    }
  }

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdArrayMeshVariables<ItemType>::
setValues(Integer iteration, const GroupType& group)
{
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    setValue(n, item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdArrayMeshVariables<ItemType>::
setEvenValues(Integer iteration, const GroupType& group)
{
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    setValue(n, item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdArrayMeshVariables<ItemType>::
setOddValues(Integer iteration, const GroupType& group)
{
  ENUMERATE_ITEM(iter,group){
    ItemType item = (*iter).itemBase();
    Int64 n = 1 + (item.uniqueId().asInt64() + iteration*7);
    n = -n;
    setValue(n, item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> void StdArrayMeshVariables<ItemType>::
setValue(Int64 n, ItemType item)
{
  Real r = Convert::toReal(n);
  Integer i = (Integer)(n);
  
  ByteArrayView array_byte = this->m_byte[item];
  for (Integer index=0, size=array_byte.size(); index<size; ++index)
    array_byte[index] = (Byte)((n+index) % 255);
  
  RealArrayView array_real = this->m_real[item];
  for (Integer index=0, size=array_real.size(); index<size; ++index)
    array_real[index] = r + Convert::toReal(index);
  
  Int64ArrayView array_int64 = this->m_int64[item];
  for (Integer index=0, size=array_int64.size(); index<size; ++index)
    array_int64[index] = n + index;
  
  Int32ArrayView array_int32 = this->m_int32[item];
  for (Integer index=0, size=array_int32.size(); index<size; ++index)
    array_int32[index] = static_cast<Int32>(i + 1 + index);
  
  Int16ArrayView array_int16 = this->m_int16[item];
  for (Integer index=0, size=array_int16.size(); index<size; ++index)
    array_int16[index] = static_cast<Int16>(i + 2 + index);
  
  Real2ArrayView array_real2 = this->m_real2[item];
  for (Integer index=0, size=array_real2.size(); index<size; ++index){
    Real2 r2;
    r2 = (static_cast<Real>(index) + r) / 2.0;
    array_real2[index] = r2;
  }
  Real2x2ArrayView array_real2x2 = this->m_real2x2[item];
  for (Integer index=0, size=array_real2x2.size(); index<size; ++index){
    Real2x2 r2x2;
    r2x2 = (static_cast<Real>(index) + r) / 4.0;
    array_real2x2[index] = r2x2;
  }
  
  Real3ArrayView array_real3 = this->m_real3[item];
  for (Integer index=0, size=array_real3.size(); index<size; ++index)
  {
    Real3 r3;
    r3 = (static_cast<Real>(index) + r) / 3.0;
    array_real3[index] = r3;
  }
  
  
  Real3x3ArrayView array_real3x3 = this->m_real3x3[item];
  for (Integer index=0, size=array_real3x3.size(); index<size; ++index){
    Real3x3 r3x3;
    r3x3 = (static_cast<Real>(index) + r) / 9.0;
    array_real3x3[index] = r3x3;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType> Integer StdArrayMeshVariables<ItemType>::
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

  return nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class StdArrayMeshVariables<Node>;
template class StdArrayMeshVariables<Edge>;
template class StdArrayMeshVariables<Face>;
template class StdArrayMeshVariables<Cell>;
template class StdArrayMeshVariables<Particle>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
