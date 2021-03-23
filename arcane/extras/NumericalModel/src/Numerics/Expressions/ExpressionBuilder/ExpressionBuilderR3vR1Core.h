// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// -*- C++ -*-
#ifndef ARCGEOSIM_EXPRESSIONS_EXPRESSIONBUILDER_EXPRESSIONBUILDER3VR1CORE_H
#define ARCGEOSIM_EXPRESSIONS_EXPRESSIONBUILDER_EXPRESSIONBUILDER3VR1CORE_H
/***************************************************************/
/* This is an automatically generated file                     */
/*              DO NOT MODIFY                                  */
/***************************************************************/
// Generated from ExpressionBuilderRnvRmCore.h.template
// by Template Tool Kit at Mon Jul 27 13:28:48 2009


namespace Arcane { }
using namespace Arcane;

#include <arcane/utils/CArray.h>

#include "Numerics/Expressions/IFunctionR3vR1.h"

class FunctionParser;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ExpressionBuilderR3vR1Core
  : public IFunctionR3vR1
{
public:
  ExpressionBuilderR3vR1Core(FunctionParser * function_parser,
                                         bool delegate_destroy = false);

  virtual ~ExpressionBuilderR3vR1Core();

public:
  
  //@{ @name Methods from IIFunction

  //! Initialisation
  void init();
  
  //! Setting parameter
  void setParameter(const String & name, const Real & parameter);
  void setParameter(      Integer index, const Real & parameter);
  
  //! Getting number of parameter
  Integer nbParameter() const;
  
  //! Getting name of parameter
  String parameterName(Integer index) const;

  //! Setting vectorized variable
  void setVariable(const String & name, const CArrayT<Real> & variable);
  void setVariable(      Integer index, const CArrayT<Real> & variable);
  
  //! Getting number of variable
  Integer nbVariable() const;

  //! Getting name of variable
  String variableName(Integer index) const;

  //! Setting evaluation vectorized result
  void setEvaluationResult(CArrayT<Real> & result);

  //! Setting derivation vectorized result
  /*! Derivation following @name di variable */
  void setDerivationResult(Integer di, CArrayT<Real> & result);
  
  //! Setting derivation vectorized result
  /*! Derivation following @name di variable */
  void setDerivationResult(const String & di, CArrayT<Real> & result);
  
  //! Eval vectorized data service function
  void eval();

  //! Cleanup
  void cleanup();

  //@}
  
  //@{ @name Local methods

  //! Point-wise evaluation
  void eval(const Real & var0,
            const Real & var1,
            const Real & var2,
            Real & res0);    

  //! Vector evaluation
  void eval(const CArrayT<Real> & var0,
            const CArrayT<Real> & var1,
            const CArrayT<Real> & var2,
            CArrayT<Real> & res0);
 
  //! Vector evaluation
  void eval(const ConstArrayView<Real> var0,
            const ConstArrayView<Real> var1,
            const ConstArrayView<Real> var2,
            ArrayView<Real> res0);

  
  //! Scalar return for point-wise evaluation
  Real eval(const Real & var0,
            const Real & var1,
            const Real & var2);
  
  
  //@}

protected:
  //! Function parser
  FunctionParser * m_function_parser;

  //! Delegate destruction of FunctionParser
  bool m_delegate_destroy;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /*  ARCGEOSIM_EXPRESSIONS_EXPRESSIONBUILDER_EXPRESSIONBUILDER3VR1CORE_H */
