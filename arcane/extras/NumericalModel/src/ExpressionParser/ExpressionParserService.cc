// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "ExpressionParser/ExpressionParserService.h"

void ExpressionParserService::parse(const std::string& s) {
  try {
    m_driver.parse(s); 
  }
  catch(ExpressionDriver::Error e) {
    fatal() << e.msg;
  }
}

void ExpressionParserService::parse(const String& s) { 
  if(s.null())
    fatal() << "Null string cannot be parsed";
  try{    
    m_driver.parse(s.localstr()); 
  }
  catch(ExpressionDriver::Error e) {
    fatal() << e.msg;
  }
}

ARCANE_REGISTER_SERVICE_EXPRESSIONPARSER(ExpressionParser, ExpressionParserService);
