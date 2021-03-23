// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_EXPRESSIONPARSERSERVICE_H
#define ARCGEOSIM_EXPRESSIONPARSERSERVICE_H

#include "ExpressionParser/IExpressionParser.h"

#include "ExpressionParser_axl.h"

#include "ExpressionParser/ExpressionDriver.h"

using namespace Arcane;

/*!
  \class ExpressionParserService
  \author Daniele A. Di Pietro
  \date 2007-08-10
  \brief Expression driver service implementation
*/

class ExpressionParserService : public ArcaneExpressionParserObject {
 public:
  ExpressionParserService(const ServiceBuildInfo& sbi) :  
    ArcaneExpressionParserObject(sbi)
    {
      // Do nothing
    }
  virtual ~ExpressionParserService()
    {
      // Do nothing
    }
  
  //! Return version number
  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }

 public:
  // Initialize
  void init() { }
  //! Parse the string
  void parse(const std::string& s);
  //! Parse the string
  void parse(const String& s);

 public:
  //! Return result
  inline expression_parser::Expression getResult() { return m_driver.getResult(); }

 public:
  //! Set trace scanning
  void setTraceScanning(bool value) { m_driver.setTraceScanning(value); }
  //! Set trace parsing
  void setTraceParsing(bool value) { m_driver.setTraceScanning(value); }

 private:
  //! Expression driver core
  ExpressionDriver m_driver;
};

#endif /* ARCGEOSIM_EXPRESSIONPARSERSERVICE_H */
