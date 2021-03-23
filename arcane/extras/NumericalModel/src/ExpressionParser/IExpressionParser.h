#ifndef IEXPRESSIONPARSER_H
#define IEXPRESSIONPARSER_H

#include <arcane/VariableTypedef.h>

#include "ExpressionParser/expression.h"

using namespace Arcane;

class IExpressionParser {
 public:
  virtual ~IExpressionParser() {}

  virtual void init() = 0;

  // Parsing
  virtual void parse(const std::string& s) = 0;
  virtual void parse(const String& s) = 0;

  // Getters
  virtual expression_parser::Expression getResult() = 0;
};

#endif
