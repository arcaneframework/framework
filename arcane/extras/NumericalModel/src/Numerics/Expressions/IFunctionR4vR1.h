// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// -*- C++ -*-
#ifndef ARCGEOSIM_EXPRESSIONS_IFUNCTIONR4VR1_H
#define ARCGEOSIM_EXPRESSIONS_IFUNCTIONR4VR1_H
/***************************************************************/
/* This is an automatically generated file                     */
/*              DO NOT MODIFY                                  */
/***************************************************************/
// Generated from IFunctionRnvRm.h.template
// by Template Tool Kit at Tue Jul 28 13:37:53 2009


namespace Arcane { }
using namespace Arcane ;

#include "Numerics/Expressions/IIFunction.h"

#include <arcane/utils/UtilsTypes.h>

class IFunctionR4vR1 : public IIFunction
{
public:
  /** Constructeur de la classe */
  IFunctionR4vR1() 
    {
      ;
    }
  
  /** Destructeur de la classe */
  virtual ~IFunctionR4vR1() { }
  
public:
  //! Getting dimension of in-space
  Integer getInDimension() const { return 4; }
  
  //! Getting dimension of out-space
  Integer getOutDimension() const { return 1; }

  //! Point-wise evaluation
  /*! An optimized syntax will be introduce for single return evaluation 
   */
  virtual void eval(const Real & var0,
                    const Real & var1,
                    const Real & var2,
                    const Real & var3,
                    Real & res0) = 0;

  //! Vector evaluation
  virtual void eval(const CArrayT<Real> & var0,
                    const CArrayT<Real> & var1,
                    const CArrayT<Real> & var2,
                    const CArrayT<Real> & var3,
                    CArrayT<Real> & res0) = 0;

  //! Vector evaluation
  virtual void eval(const ConstArrayView<Real> var0,
                    const ConstArrayView<Real> var1,
                    const ConstArrayView<Real> var2,
                    const ConstArrayView<Real> var3,
                    ArrayView<Real> res0) = 0;

  
  //! Scalar return for point-wise evaluation
  virtual Real eval(const Real & var0,
                    const Real & var1,
                    const Real & var2,
                    const Real & var3) = 0;
  
};

#endif /* ARCGEOSIM_EXPRESSIONS_IFUNCTIONR4VR1_H */
