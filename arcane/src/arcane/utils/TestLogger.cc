// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestLogger.cc                                               (C) 2000-2022 */
/*                                                                           */
/* Classe utilitaire pour enregistrer les informations de tests.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TestLogger.h"

#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include <sstream>
#include <iostream>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class TestLoggerImpl
{
 public:

  std::ostringstream m_stream;
};

namespace
{
  TestLoggerImpl& _impl()
  {
    static thread_local TestLoggerImpl logger;
    return logger;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
TestLogger::stream()
{
  return _impl().m_stream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TestLogger::
dump(std::ostream& o)
{
  std::string str = _impl().m_stream.str();
  o << str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int TestLogger::
compare()
{
  String result_file = platform::getEnvironmentVariable("ARCANE_TEST_RESULT_FILE");
  if (result_file.null())
    return 0;

  UniqueArray<std::byte> bytes;
  if (platform::readAllFile(result_file, false, bytes))
    ARCANE_FATAL("Can not read test result file '{0}'", result_file);

  std::string expected_str(reinterpret_cast<const char*>(bytes.data()), bytes.length());
  std::string str = _impl().m_stream.str();

  if (expected_str != str) {
    std::cout << "TestLogger: ERROR during test comparison:\n";
    std::cout << "Current:\n";
    std::cout << "\n'''" << str << "'''\n";
    std::cout << "\nExpected:\n";
    std::cout << "\n'''" << expected_str << "'''\n";
    return 1;
  }

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
