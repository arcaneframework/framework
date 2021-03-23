/* -*- C++ -*- */
%{
#include <cstdlib>
#include <errno.h>
#include <limits.h>
#include <string>
#include <sstream>
#include "ExpressionDriver.h"
#include "expression_parser.hh"

// Work around an incompatibility in flex
#undef yywrap
#define yywrap() 1

// By default yylex returns int, we use token_type.
#define yyterminate() return token::END  

// Alias for token type
typedef ep::ExpressionParserImpl::token token;
%}

%option noyywrap nounput batch debug
%option prefix="ep"

ID    [xyzt]
DIGIT [0-9]
BLANK [ \t]

%{
# define YY_USER_ACTION  yylloc->columns(yyleng);
%}
%%

%{
  yylloc->step();
%}
{BLANK}+ {                              // Blanks and tabs
  yylloc->step();
}
[\n]+ {                                 // New line
  yylloc->lines(yyleng); yylloc->step();
}
[-+*/^()] {                             // Admissible symbols
  return ep::ExpressionParserImpl::token_type(yytext[0]);
}
{DIGIT}+ {                              // Integers
  errno = 0;
  yylval->dval = atof( yytext );
  return token::NUMBER;
}
{DIGIT}+"."{DIGIT}* {                   // Floats
  errno = 0;
  yylval->dval = atof( yytext );
  return token::NUMBER;
}
{ID} {                                  // Variables
  // Variables
  yylval->cval = yytext[0]; 
  return token::IDENTIFIER;
}
fabs |
sqrt |
sgn |
h |
erf |
erfc |
sin |
cos |
log |
exp {                                   // Functions
  errno = 0;
  yylval->fval = driver.function(yytext);
  return token::FUNCTION;
}
PI {
  errno = 0;
  yylval->sval = new std::string (yytext);
  return token::CONSTANT;
}
.
{                                       // Non-admissible characters
  driver.error (*yylloc, "invalid character");
}
%%

void ExpressionDriver::_scan_begin () {
  yy_flex_debug = m_trace_scanning;
  yy_scan_string( this->expression().c_str() );
}

void ExpressionDriver::_scan_end () {}
