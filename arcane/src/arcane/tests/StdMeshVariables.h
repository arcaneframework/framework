// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdMeshVariables.h                                          (C) 2000-2016 */
/*                                                                           */
/* Définition de variables du maillage pour des tests.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_STDMESHVARIABLES_H
#define ARCANE_TEST_STDMESHVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/MeshVariable.h"
#include "arcane/VariableBuildInfo.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType,int dim>
class StdMeshVariablesTraits;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
class StdMeshVariablesTraits<ItemType,DataType,0>
{
 public:
  typedef MeshVariableScalarRefT<ItemType,DataType> VariableRefType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
class StdMeshVariablesTraits<ItemType,DataType,1>
{
 public:
  typedef MeshVariableArrayRefT<ItemType,DataType> VariableRefType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,int dim>
class StdMeshVariableTraits2
{
 public:
  typedef typename StdMeshVariablesTraits<ItemType,Byte,dim>::VariableRefType VariableByteType;
  typedef typename StdMeshVariablesTraits<ItemType,Real,dim>::VariableRefType VariableRealType;
  typedef typename StdMeshVariablesTraits<ItemType,Int64,dim>::VariableRefType VariableInt64Type;
  typedef typename StdMeshVariablesTraits<ItemType,Int32,dim>::VariableRefType VariableInt32Type;
  typedef typename StdMeshVariablesTraits<ItemType,Int16,dim>::VariableRefType VariableInt16Type;
  typedef typename StdMeshVariablesTraits<ItemType,Real3,dim>::VariableRefType VariableReal3Type;
  typedef typename StdMeshVariablesTraits<ItemType,Real3x3,dim>::VariableRefType VariableReal3x3Type;
  typedef typename StdMeshVariablesTraits<ItemType,Real2,dim>::VariableRefType VariableReal2Type;
  typedef typename StdMeshVariablesTraits<ItemType,Real2x2,dim>::VariableRefType VariableReal2x2Type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct MultiScalarValue
{
  Byte m_byte;
  Real m_real;
  Int64 m_int64;
  Int32 m_int32;
  Int16 m_int16;
  Real2 m_real2;
  Real2x2 m_real2x2;
  Real3 m_real3;
  Real3x3 m_real3x3;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Traits>
class StdMeshVariables
{
 public:

  typedef typename Traits::VariableByteType VarByte;
  typedef typename Traits::VariableRealType VarReal;
  typedef typename Traits::VariableInt64Type VarInt64;
  typedef typename Traits::VariableInt32Type VarInt32;
  typedef typename Traits::VariableInt16Type VarInt16;
  typedef typename Traits::VariableReal3Type VarReal3;
  typedef typename Traits::VariableReal3x3Type VarReal3x3;
  typedef typename Traits::VariableReal2Type VarReal2;
  typedef typename Traits::VariableReal2x2Type VarReal2x2;

 public:

  StdMeshVariables(const MeshHandle& mesh_handle,const String& basestr,const String& base2str);
  StdMeshVariables(const MeshHandle& mesh_handle,const String& basestr,const String& base2str,const String& family_name);

 public:

  void synchronize();
  void addToCollection(VariableCollection vars);

 public:

  /*!
   * \brief Applique le fonctor \a lambda sur toutes les variables.
   *
   * Le fonctor doit définir l'opérateur operator()(T&) avec
   * \a T le type de la variable.
   */
  template<typename Functor> void
  applyFunctor(Functor& lambda)
  {
    lambda(m_byte);
    lambda(m_real);
    lambda(m_int64);
    lambda(m_int32);
    lambda(m_int16);
    lambda(m_real2);
    lambda(m_real2x2);
    lambda(m_real3);
    lambda(m_real3x3);
  }

 public:

  VarByte m_byte;
  VarReal m_real;
  VarInt64 m_int64;
  VarInt32 m_int32;
  VarInt16 m_int16;
  VarReal2 m_real2;
  VarReal2x2 m_real2x2;
  VarReal3 m_real3;
  VarReal3x3 m_real3x3;

 protected:

  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

