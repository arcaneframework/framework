// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "ExpressionParser/expression.h"

namespace expression_parser {

  ////////////////////////////////////////////////////////////
  // Operators

  REGISTER_BINARY_OPERATOR(operator+, Addition);
  REGISTER_BINARY_OPERATOR(operator-, Subtraction);
  REGISTER_BINARY_OPERATOR(operator*, Multiplication);
  REGISTER_BINARY_OPERATOR(operator/, Division);
  REGISTER_BINARY_OPERATOR(_power, Power);

  ////////////////////////////////////////////////////////////
  // Functions

  REGISTER_FUNCTION(fabs, AbsoluteValue);
  REGISTER_FUNCTION(sqrt, SquareRoot);
  REGISTER_FUNCTION(sgn, Sign);
  REGISTER_FUNCTION(h, Heavyside);
  REGISTER_FUNCTION(erf, ErrorFunction);
  REGISTER_FUNCTION(erfc, ComplementaryErrorFunction);
  REGISTER_FUNCTION(sin, Sinus);
  REGISTER_FUNCTION(cos, Cosinus);
  REGISTER_FUNCTION(log, Logarithm);
  REGISTER_FUNCTION(exp, Exponential);
  
}
