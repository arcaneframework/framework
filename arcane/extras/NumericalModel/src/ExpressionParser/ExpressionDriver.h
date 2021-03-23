// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_EXPRESSIONDRIVER_H
#define ARCGEOSIM_EXPRESSIONDRIVER_H

#include <string>

#include <map>
#include <stack>

#include <arcane/utils/ITraceMng.h>

#include "expression_parser.hh"
#include "ExpressionParser/expression.h"

/*!
  \class ExpressionDriver
  \author Daniele A. Di Pietro
  \date 2007-08-10
  \brief Expression driver core
*/

class ExpressionDriver {
 public:
  struct Error {
    std::string msg;
    Error(const std::string& _msg) : msg(_msg) {}
  };

 public:
  ExpressionDriver();
  virtual ~ExpressionDriver() {}

 public:
  //! Parse the expression
  void parse(const std::string& expression);

 public:
  //! Error handling (print location and error message)
  void error(const ep::location& l, const std::string& m);
  //! Error handling (print error message only)
  void error(const std::string& m);

 public:
  //! Return parsed expression
  std::string& expression() { return m_expression; }

  //! Return variable
  expression_parser::Expression variable(const char& v) { return m_variables[v]; }
  //! Return function
  expression_parser::FunctionPtr function(const std::string& f) { return m_functions[f]; }
  //! Return constant
  Real constant(std::string c) { return m_constants[c]; }

  //! Return result
  const expression_parser::Expression& getResult() { return m_result; }
  //! Return result (internal use only)
  expression_parser::Expression& result() { return m_result; }

  //! Return stack
  std::stack<expression_parser::ExpressionPtr>& nodes() { return m_nodes; }
 public:
  //! Set trace scanning
  void setTraceScanning(bool value) { m_trace_scanning = value; }
  //! Set trace parsing
  void setTraceParsing(bool value) { m_trace_parsing = value; }

  //! Clear stack
  void clear_stack() {
    while (!m_nodes.empty ()) {
      delete m_nodes.top ();
      m_nodes.pop ();
    }
  }

 private:
  //! File to process
  std::string m_expression;

  //! Trace scanning flag
  bool m_trace_scanning;
  //! Trace parsing flag
  bool m_trace_parsing;

  //! Variables
  std::map<char, expression_parser::Expression> m_variables;
  //! Functions
  std::map<std::string, expression_parser::FunctionPtr> m_functions;
  //! Constants
  std::map<std::string, Real> m_constants;

  //! Result
  expression_parser::Expression m_result;

  //! Nodes
  std::stack<expression_parser::ExpressionPtr> m_nodes;

 private:
  //! Begin scanning
  void _scan_begin();
  //! End scanning
  void _scan_end();
};

// Announce to Flex the prototype we want for lexing function, ...
# define YY_DECL					\
  ep::ExpressionParserImpl::token_type                         \
  eplex (ep::ExpressionParserImpl::semantic_type* yylval,      \
         ep::ExpressionParserImpl::location_type* yylloc,      \
         ExpressionDriver& driver)
// ... and declare it for the parser's sake.
YY_DECL;

#endif /* ARCGEOSIM_EXPRESSIONDRIVER_H */
