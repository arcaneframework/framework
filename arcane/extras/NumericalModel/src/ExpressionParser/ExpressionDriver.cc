// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <sstream>

#include "ExpressionParser/ExpressionDriver.h"

#include "ExpressionParser/expression.h"

using namespace Arcane;

ExpressionDriver::ExpressionDriver() :
  m_trace_scanning(false), 
  m_trace_parsing(false) 
{
  // Variables
  m_variables['x'] = new expression_parser::PlaceHolderX;
  m_variables['y'] = new expression_parser::PlaceHolderY;
  m_variables['z'] = new expression_parser::PlaceHolderZ;
  m_variables['t'] = new expression_parser::PlaceHolderT;

  // Functions
  m_functions["fabs"] = &expression_parser::_fabs;
  m_functions["sqrt"] = &expression_parser::_sqrt;
  m_functions["sgn"] = &expression_parser::_sgn;
  m_functions["h"] = &expression_parser::_h;
  m_functions["erf"] = &expression_parser::_erf;
  m_functions["erfc"] = &expression_parser::_erfc;
  m_functions["sin"] = &expression_parser::_sin;
  m_functions["cos"] = &expression_parser::_cos;
  m_functions["log"] = &expression_parser::_log;
  m_functions["exp"] = &expression_parser::_exp;

  // Constants
  m_constants["PI"]  = 3.14159265358979e+00;
}

void ExpressionDriver::parse(const std::string& expression) {
  m_expression = expression;
  _scan_begin();
  ep::ExpressionParserImpl parser(*this);
  parser.set_debug_level(m_trace_parsing);
  parser.parse();
  _scan_end();
}

void ExpressionDriver::error(const ep::location& l, const std::string& m) {
  std::ostringstream err_msg;
  err_msg << l << ": " << m << std::endl;
  throw(Error(err_msg.str()));
}

void ExpressionDriver::error(const std::string& m) {
  throw(m);
}
