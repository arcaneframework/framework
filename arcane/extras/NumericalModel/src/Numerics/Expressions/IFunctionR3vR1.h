// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// -*- C++ -*-
#ifndef ARCGEOSIM_EXPRESSIONS_IFUNCTIONR3VR1_H
#define ARCGEOSIM_EXPRESSIONS_IFUNCTIONR3VR1_H
/***************************************************************/
/* This is an automatically generated file                     */
/*              DO NOT MODIFY                                  */
/***************************************************************/
// Generated from IFunctionRnvRm.h.template
// by Template Tool Kit at Mon Jul 27 13:30:43 2009


namespace Arcane { }
using namespace Arcane ;

#include "Numerics/Expressions/IIFunction.h"

#include <arcane/utils/UtilsTypes.h>

class IFunctionR3vR1 : public IIFunction
{
public:
  /** Constructeur de la classe */
  IFunctionR3vR1() 
    {
      ;
    }
  
  /** Destructeur de la classe */
  virtual ~IFunctionR3vR1() { }
  
public:
  //! Getting dimension of in-space
  Integer getInDimension() const { return 3; }
  
  //! Getting dimension of out-space
  Integer getOutDimension() const { return 1; }

  //! Point-wise evaluation
  /*! An optimized syntax will be introduce for single return evaluation 
   */
  virtual void eval(const Real & var0,
                    const Real & var1,
                    const Real & var2,
                    Real & res0) = 0;

  //! Vector evaluation
  virtual void eval(const CArrayT<Real> & var0,
                    const CArrayT<Real> & var1,
                    const CArrayT<Real> & var2,
                    CArrayT<Real> & res0) = 0;

  //! Vector evaluation
  virtual void eval(const ConstArrayView<Real> var0,
                    const ConstArrayView<Real> var1,
                    const ConstArrayView<Real> var2,
                    ArrayView<Real> res0) = 0;

  
  //! Scalar return for point-wise evaluation
  virtual Real eval(const Real & var0,
                    const Real & var1,
                    const Real & var2) = 0;
  
};

#endif /* ARCGEOSIM_EXPRESSIONS_IFUNCTIONR3VR1_H */
