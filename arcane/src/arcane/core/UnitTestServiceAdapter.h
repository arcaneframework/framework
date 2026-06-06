// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnitTestServiceAdapter.h                                    (C) 2000-2025 */
/*                                                                           */
/* Adapts a service that declares tests to the IUnitTest interface.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_UNITTESTADAPTER_H
#define ARCANE_CORE_UNITTESTADAPTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IUnitTest.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/Assertion.h"
#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Adapts a service that declares tests to the IUnitTest interface.
 */
template <typename T>
class UnitTestServiceAdapter
: public IXmlUnitTest
, public Assertion
{
 public:

  typedef void (T::*FuncPtr)(); //!< Type of the pointer to the test methods

 public:

  explicit UnitTestServiceAdapter(T* service)
  : m_service(service)
  {}

 public:

  void setClassSetUpFunction(FuncPtr f) { m_class_set_up_function = f; }
  void setTestSetUpFunction(FuncPtr f) { m_set_up_function = f; }
  void setClassTearDownFunction(FuncPtr f) { m_class_tear_down_function = f; }
  void setTestTearDownFunction(FuncPtr f) { m_tear_down_function = f; }
  void addTestFunction(FuncPtr f, String name, String method_name)
  {
    TestFuncInfo info(f, name, method_name);
    m_test_functions.add(info);
  }

 public:

  //! Implementation of the IUnitTest interface
  void initializeTest() override
  {
    if (m_class_set_up_function)
      (m_service->*m_class_set_up_function)();
  }

  //! Implementation of the IUnitTest interface
  bool executeTest(XmlNode& report) override
  {
    bool success = true;
    report.setAttrValue("name", m_service->serviceInfo()->localName());
    for (TestFuncInfo func_info : m_test_functions) {
      XmlNode xunittest = report.createAndAppendElement("unit-test");
      try {
        xunittest.setAttrValue("name", func_info.m_name);
        xunittest.setAttrValue("method-name", func_info.m_method_name);
        if (m_set_up_function)
          (m_service->*m_set_up_function)();
        (m_service->*func_info.m_test_func)();
        if (m_tear_down_function)
          (m_service->*m_tear_down_function)();
        xunittest.setAttrValue("result", "success");
        m_service->info() << "[OK   ] " << func_info.m_name;
      }
      catch (const AssertionException& e) {
        xunittest.setAttrValue("result", "failure");
        XmlNode xexception = xunittest.createAndAppendElement("exception");
        xexception.setAttrValue("where", e.where());
        xexception.setAttrValue("file", e.file());
        xexception.setAttrValue("line", Arcane::String::fromNumber(e.line()));
        xexception.setAttrValue("message", e.message());
        m_service->info() << "[FAILURE] " << func_info.m_name << " (line " << e.line() << " in " << e.where() << ")";
        m_service->info() << "        " << e.message();
        success = false;
      }
    }
    return success;
  }

  //! Implementation of the IUnitTest interface
  void finalizeTest() override
  {
    if (m_class_tear_down_function)
      (m_service->*m_class_tear_down_function)();
  }

 private:

  struct TestFuncInfo
  {
    TestFuncInfo(FuncPtr test_func, String name, String method_name)
    : m_test_func(test_func)
    , m_name(name)
    , m_method_name(method_name)
    {}

    FuncPtr m_test_func;
    String m_name;
    String m_method_name;
  };

 private:

  //! Pointer to the class initialization method.
  FuncPtr m_class_set_up_function = nullptr;
  //! Pointer to the initialization method of each test.
  FuncPtr m_set_up_function = nullptr;
  //! Pointer to the class teardown method.
  FuncPtr m_class_tear_down_function = nullptr;
  //! Pointer to the teardown method of each test.
  FuncPtr m_tear_down_function = nullptr;
  //!< Pointers to the test methods.
  UniqueArray<TestFuncInfo> m_test_functions;
  //!< Associated service.
  T* m_service;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
