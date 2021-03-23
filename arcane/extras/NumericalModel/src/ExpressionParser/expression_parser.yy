/* -*- C++ -*- */
%skeleton "lalr1.cc"
%require "2.1a"
%name-prefix="ep"
%defines
%define "parser_class_name" "ExpressionParserImpl"

%{
  #include<string>
  #include<stack>

  #include "expression.h"

  extern expression_parser::Expression _x;
  extern expression_parser::Expression _y;
  extern expression_parser::Expression _z;
  extern expression_parser::Expression _t;

  class ExpressionDriver;
%}

// The parsing context.
%parse-param { ExpressionDriver& driver }
%lex-param   { ExpressionDriver& driver }

%locations
%initial-action
{
};

%debug
%error-verbose

// Symbols.
%union
{
  double                           dval;
  char                             cval;
  std::string*                     sval;
  expression_parser::ExpressionPtr eval;
  expression_parser::FunctionPtr   fval;
};

%{
# include "ExpressionDriver.h"
%}

%token        END      0   "end of file"
%token <dval> NUMBER       "number"
%token <cval> IDENTIFIER   "identifier"
%token <sval> CONSTANT     "constant"
%token <fval> VAR FUNCTION "function"
%type  <eval> exp          "expression"

%%
input: /* empty */
  | input line
;

line: exp { 
    driver.result() = *$1; 
    driver.clear_stack(); 
  }
;

%left '+' '-';
%left '*' '/';
%right '^';
exp: exp '+' exp 
  {
    $$ = new expression_parser::Expression( *$1 + *$3 );
    driver.nodes().pop(); driver.nodes().pop(); // Remove operands
    driver.nodes().push($$);
  }
  | exp '-' exp 
  {
    $$ = new expression_parser::Expression( *$1 - *$3 );
    driver.nodes().pop(); driver.nodes().pop(); // Remove operands
    driver.nodes().push($$);
  }
  | exp '*' exp 
  {
    $$ = new expression_parser::Expression( *$1 * *$3 );
    driver.nodes().pop(); driver.nodes().pop(); // Remove operands
    driver.nodes().push($$);
  }
  | exp '/' exp 
  {
    $$ = new expression_parser::Expression( *$1 / *$3 );
    driver.nodes().pop(); driver.nodes().pop(); // Remove operands
    driver.nodes().push($$);
  }
  | exp '^' exp
  {
    $$ = new expression_parser::Expression( _power(*$1, *$3) );
    driver.nodes().pop(); driver.nodes().pop(); // Remove operands
    driver.nodes().push($$);
  }
  | '(' exp ')' 
  {
    $$ = $2;  
  }
  | FUNCTION '(' exp ')' {
    $$ = new expression_parser::Expression( (*$1)( *$3 ) );
  }
  | NUMBER 
  { 
    $$ = new expression_parser::Expression($1); 
    driver.nodes().push($$); 
  }
  | IDENTIFIER 
  {
    $$ = new expression_parser::Expression( driver.variable($1) );
    driver.nodes().push($$);
  }
  | CONSTANT
  {
    $$ = new expression_parser::Expression( driver.constant(*$1) );
    driver.nodes().push($$);
  }
%%

void ep::ExpressionParserImpl::error(const ep::ExpressionParserImpl::location_type& l,
                                     const std::string& m) {
  driver.error(l, m);
}

