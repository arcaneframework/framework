// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IDISCRETEVARTYPES_H_
#define IDISCRETEVARTYPES_H_

#include "Utils/Utils.h"
#include "Utils/ItemGroupMap.h"

using namespace Arcane;

struct IDiscreteVarTypes
{
  typedef enum {
    Scalar,
    Vectorial,
    Tensorial
  } eVarDim ;
  
  template<eVarDim dim>
  struct ValueArrayType ;
  
  template<eVarDim dim>
  struct MeshVarType ;
  
  template< eVarDim dim>
  struct AssignOp ;
};

template<>
struct IDiscreteVarTypes::ValueArrayType<IDiscreteVarTypes::Scalar>
{
  typedef Array<Real> type ;
} ;

template<>
struct IDiscreteVarTypes::ValueArrayType<IDiscreteVarTypes::Vectorial>
{
  typedef RealArray2 type ;
} ;


template<>
struct IDiscreteVarTypes::MeshVarType<IDiscreteVarTypes::Scalar>
{
  typedef VariableCellReal type ;
  typedef ItemGroupMapT<Face,Real> boundary_type ;
} ;


template<>
struct IDiscreteVarTypes::MeshVarType<IDiscreteVarTypes::Vectorial>
{
  typedef VariableCellArrayReal type ;
  typedef ItemGroupMapArrayT<Face,Real> boundary_type ;
} ;


template<>
struct IDiscreteVarTypes::AssignOp<IDiscreteVarTypes::Scalar>
{
  template<typename VarType1,typename VarType2>
  void operator()(VarType1& dest, const VarType2& source) {
    dest = source ;
  }
} ;

template<>
struct IDiscreteVarTypes::AssignOp<IDiscreteVarTypes::Vectorial>
{
  template<typename VarArrayType1,typename VarArrayType2>
  void operator()(VarArrayType1& dest, const VarArrayType2& source) {
    dest.copy(source) ;
  }
  void operator()(ArrayView<Real> dest, Real source) {
    dest.fill(source) ;
  }
  void operator()(ArrayView<Real> dest, ConstArrayView<Real> source) {
    dest.copy(source) ;
  }
} ;

#endif /*IDISCRETEVARTYPES_H_*/
