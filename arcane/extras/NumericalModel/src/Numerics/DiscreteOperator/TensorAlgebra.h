// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// -*- C++ -*-
#ifndef TENSOR_ALGEBRA_H
#define TENSOR_ALGEBRA_H

#include <arcane/ArcaneTypes.h>
#include <arcane/MathUtils.h>
#include <arcane/VariableTypedef.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace DiscreteOperator {

  using namespace Arcane;

  template<typename T> struct ArcaneVariableTypeTraits {
  };

  template<> struct ArcaneVariableTypeTraits<Real> {
    typedef VariableCellReal type;
  };

  template<> struct ArcaneVariableTypeTraits<Real3> {
    typedef VariableCellReal3 type;
  };

  template<> struct ArcaneVariableTypeTraits<Real3x3> {
    typedef VariableCellReal3x3 type;
  };

  /*---------------------------------------------------------------------------*/

  struct i_tensor_vector_prod {
    struct Error {};    
  };

  template<typename T1>
  struct tensor_vector_prod 
    : public i_tensor_vector_prod {
    static Real3 eval (const T1& t, const Real3& v) {
      throw(i_tensor_vector_prod::Error());
    }
  };

  template<> struct tensor_vector_prod<Real> 
    : public i_tensor_vector_prod {
    static Real3 eval (const Real& t, const Real3& v) {
      return t * v;
    }
  };

  template<> struct tensor_vector_prod<Real3> 
    : public i_tensor_vector_prod {
    static Real3 eval (const Real3& t, const Real3& v) {
      return Real3(t.x * v.x,
                   t.y * v.y,
                   t.z * v.z);
    }
  };

  template<> struct tensor_vector_prod<Real3x3> 
    : public i_tensor_vector_prod {
    static Real3 eval (const Real3x3& t, const Real3& v) {
      return math::prodTensVec(t, v);
    }
  };

  /*---------------------------------------------------------------------------*/

  struct MinimumEigenvalue
  {
    static Real compute(const Real & A);
    static Real compute(const Real3 & A);
    static Real compute(const Real3x3 & A);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif
