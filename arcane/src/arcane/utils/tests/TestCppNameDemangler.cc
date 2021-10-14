// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/CppNameDemangler.h"
#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void _printType(CppNameDemangler& demangler,const std::type_info& ti)
{
  const char* buf = demangler.demangle(ti.name());
  std::cerr << "BUF=" << buf << "\n";
}
}

TEST(CppNameDemangler,Misc)
{
  CppNameDemangler demangler(20);
  using Type1 = std::tuple<std::vector<Int32>,std::vector<std::vector<Int64>>>;
  using Type2 = std::tuple<std::vector<Type1>,std::vector<std::vector<Type1>>>;
  using Type3 = std::tuple<std::vector<Type2>,std::vector<std::vector<Type2>>>;
  _printType(demangler,typeid(Type1));
  _printType(demangler,typeid(Type2));
  const char* buf2 = demangler.demangle("NotATrueType!");
  std::cout << "BUF2=" << buf2 << "\n";
  _printType(demangler,typeid(Type1));
  _printType(demangler,typeid(Type3));
  std::cout.flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
