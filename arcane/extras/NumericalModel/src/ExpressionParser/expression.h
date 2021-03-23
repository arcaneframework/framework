// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <cmath>
#include <iostream>

#include <arcane/VariableTypes.h>
#include <arcane/VariableTypedef.h>

#include "ExpressionParser/IXYZTFunction.h"

#include "boost/shared_ptr.hpp"

using namespace Arcane;

namespace expression_parser {

  ////////////////////////////////////////////////////////////
  // Interfaces

  class IExpression : public IXYZTFunction {
  public:
    virtual ~IExpression() {}
    virtual Real eval(const Real3& P, Real t) = 0;
  };

  class IUnaryOperator {
  public:
    virtual ~IUnaryOperator() {}
    virtual Real apply(Real a) = 0;
  };

  class IBinaryOperator {
  public:
    virtual ~IBinaryOperator() { }
    virtual Real apply(Real a, Real b) = 0;
  };

  typedef boost::shared_ptr<IExpression> IExpressionSharedPtr;

  ////////////////////////////////////////////////////////////
  // Implementations

  class PlaceHolderX : public IExpression {
  public:
    inline Real eval(const Real3& P, Real t) {
      return P.x;
    }
  };

  class PlaceHolderY : public IExpression {
  public:
    virtual ~PlaceHolderY() {}
    inline Real eval(const Real3& P, Real t) {
      return P.y;
    }
  };

  class PlaceHolderZ : public IExpression {
  public:
    virtual ~PlaceHolderZ() {}
    inline Real eval(const Real3& P, Real t) {
      return P.z;
    }
  };

  class PlaceHolderT : public IExpression { 
  public:
    virtual ~PlaceHolderT() {}
    inline Real eval(const Real3& P, Real t) {
      return t;
    }
  };

  class Scalar : public IExpression {
  public:
    virtual ~Scalar() {}
    Scalar(Real value = 1) : m_value(value) {}
    inline Real eval(const Real3& P, Real t) {
      return m_value;
    }
  private:
    Real m_value;
  };

  ////////////////////////////////////////////////////////////
  // Expression and constants

  class Expression : public IExpression {
  public:
    virtual ~Expression() {}
    Expression(Real s) : m_e( new Scalar(s) ) {}
    Expression(IExpression* e = new Scalar(1)) : m_e(e) {}

    inline Real eval(const Real3& P, Real t) {
      return m_e->eval(P, t);
    }
  private:
    IExpressionSharedPtr m_e;
  };

  typedef Expression* ExpressionPtr;

  ////////////////////////////////////////////////////////////
  // Unary modification and binary composition

  class UnaryModification : public IExpression {
  public:
    UnaryModification(Expression e, IUnaryOperator* op) :
      m_e(e), m_op(op) {}
    virtual ~UnaryModification() {}

    inline Real eval(const Real3& P, Real t) {
      Real fv = m_op->apply( m_e.eval(P, t) );
      return fv;
    }
  private:
    Expression m_e;
    IUnaryOperator* m_op;
  };

  class BinaryComposition : public IExpression {
  public:
    BinaryComposition(Expression e1, Expression e2, IBinaryOperator* op) :
      m_e1(e1), m_e2(e2), m_op(op) {}
    virtual ~BinaryComposition() {}

    inline Real eval(const Real3& P, Real t) {
      return m_op->apply( m_e1.eval(P, t), m_e2.eval(P, t) );
    }

  private:
    Expression m_e1;
    Expression m_e2;
    IBinaryOperator* m_op;
  };

  ////////////////////////////////////////////////////////////
  // Elementary operations

  class Addition : public IBinaryOperator {
  public:
    inline Real apply(Real a, Real b) { return a + b; }
  };

  class Subtraction : public IBinaryOperator {
  public:
    inline Real apply(Real a, Real b) { return a - b; }
  };

  class Multiplication : public IBinaryOperator {
  public:
    inline Real apply(Real a, Real b) { return a * b; }
  };

  class Division : public IBinaryOperator {
  public:
    inline Real apply(Real a, Real b) { return a / b; }
  };

  class Power : public IBinaryOperator {
  public:
    inline Real apply(Real a, Real b) { return pow(a, b); }
  };

  ////////////////////////////////////////////////////////////
  // Functions

  class AbsoluteValue : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return fabs(a); }
  };

  class SquareRoot : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return sqrt(a); }
  };

  class Sign : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return ( (a==0)? 0 : ( (a>0) ? 1 : -1) ) ;  }
  };

  class Heavyside : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return ( (a>=0) ? 1 : 0)  ; }
  };

  class ErrorFunction : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return erf(a); }
  };

  class ComplementaryErrorFunction : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return erfc(a); }
  };

  class Sinus : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return sin(a); }
  };

  class Cosinus : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return cos(a); }
  };

  class Logarithm : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return log(a); }
  };

  class Exponential : public IUnaryOperator {
  public:
    inline Real apply(Real a) { return exp(a); }
  };

  typedef Expression& (*FunctionPtr)(Expression&);

  ////////////////////////////////////////////////////////////
  // Operator and function headers

#define FUNCTION_NAME(X) _ ## X

  Expression& FUNCTION_NAME(fabs)(Expression& e);
  Expression& FUNCTION_NAME(sqrt)(Expression& e);
  Expression& FUNCTION_NAME(sgn)(Expression& e);
  Expression& FUNCTION_NAME(h)(Expression& e);
  Expression& FUNCTION_NAME(erf)(Expression& e);
  Expression& FUNCTION_NAME(erfc)(Expression& e);
  Expression& FUNCTION_NAME(sin)(Expression& e);
  Expression& FUNCTION_NAME(cos)(Expression& e);
  Expression& FUNCTION_NAME(log)(Expression& e);
  Expression& FUNCTION_NAME(exp)(Expression& e);
  

  Expression& operator+(Expression& e1, Expression& e2);
  Expression& operator-(Expression& e1, Expression& e2);
  Expression& operator*(Expression& e1, Expression& e2);
  Expression& operator/(Expression& e1, Expression& e2);
  Expression& _power(Expression& e1, Expression& e2);

  ////////////////////////////////////////////////////////////
  // Macro to construct binary operators and underscored
  // functions

#define REGISTER_BINARY_OPERATOR(NAME, BINARY_OPERATOR) \
BINARY_OPERATOR __ ## BINARY_OPERATOR; \
Expression& NAME(Expression& e1, Expression& e2) { \
	IExpression* ieptr = new BinaryComposition(e1, e2, &__ ## BINARY_OPERATOR); \
	Expression* eptr = new Expression( ieptr ); \
	return *eptr; \
}

#define REGISTER_FUNCTION(NAME, UNARY_OPERATOR) \
UNARY_OPERATOR __ ## UNARY_OPERATOR; \
Expression& FUNCTION_NAME(NAME)(Expression& e) { \
  IExpression* ieptr = new UnaryModification( e, &__ ## UNARY_OPERATOR); \
  Expression*   eptr = new Expression ( ieptr ); \
  return *eptr; \
}

#define REGISTER_PLACEHOLDER(NAME, PLACEHOLDER) \
Expression _ ## NAME( new PLACEHOLDER );
}

#endif
