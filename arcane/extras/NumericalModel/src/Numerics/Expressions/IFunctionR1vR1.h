// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// -*- C++ -*-
#ifndef ARCGEOSIM_EXPRESSIONS_IFUNCTIONR1VR1_H
#define ARCGEOSIM_EXPRESSIONS_IFUNCTIONR1VR1_H
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

class IFunctionR1vR1 : public IIFunction
{
public:
  /** Constructeur de la classe */
  IFunctionR1vR1() 
    {
      ;
    }
  
  /** Destructeur de la classe */
  virtual ~IFunctionR1vR1() { }
  
public:
  //! Getting dimension of in-space
  Integer getInDimension() const { return 1; }
  
  //! Getting dimension of out-space
  Integer getOutDimension() const { return 1; }

  //! Point-wise evaluation
  /*! An optimized syntax will be introduce for single return evaluation 
   */
  virtual void eval(const Real & var0,
                    Real & res0) = 0;

  //! Vector evaluation
  virtual void eval(const CArrayT<Real> & var0,
                    CArrayT<Real> & res0) = 0;

  //! Vector evaluation
  virtual void eval(const ConstArrayView<Real> var0,
                    ArrayView<Real> res0) = 0;

  
  //! Scalar return for point-wise evaluation
  virtual Real eval(const Real & var0) = 0;
  
};

#endif /* ARCGEOSIM_EXPRESSIONS_IFUNCTIONR1VR1_H */
