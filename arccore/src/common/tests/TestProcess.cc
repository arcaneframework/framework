// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/common/internal/Process.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_OS_LINUX) || defined(ARCCORE_OS_MACOS)

TEST(TestProcess, ProcessEcho)
{
  // Teste 'echo -n TotoTiti'
  // Doit retourner 'TotoTiti' dans outputBytes()
  ProcessExecArgs process_args;
  process_args.setCommand("/bin/echo");
  UniqueArray<String> command_args;
  command_args.add("-n");
  String input_str("TotoTiti");
  command_args.add(input_str);
  process_args.setArguments(command_args);

  ProcessExecArgs::ExecStatus r = Process::execute(process_args);
  ASSERT_EQ(r, ProcessExecArgs::ExecStatus::OK);

  ConstArrayView<Byte> output = process_args.outputBytes();
  String output_str(output);
  std::cout << "output_str = '" << output_str << "'\n";
  ASSERT_EQ(input_str, output_str);
}

TEST(TestProcess, ProcessCat)
{
  // Teste 'cat -' avec TotoTitiTata en entrée.
  // Doit retourner 'TotoTitiTata' dans outputBytes()
  ProcessExecArgs process_args;
  process_args.setCommand("/bin/cat");
  UniqueArray<String> command_args;
  command_args.add("-");
  String input_str("TotoTitiTata");
  process_args.setArguments(command_args);
  ConstArrayView<Byte> input_bytes = input_str.bytes().constSmallView();
  process_args.setInputBytes(input_bytes);

  ProcessExecArgs::ExecStatus r = Process::execute(process_args);
  ASSERT_EQ(r, ProcessExecArgs::ExecStatus::OK);

  ConstArrayView<Byte> output = process_args.outputBytes();
  String output_str(output);
  ASSERT_EQ(input_str, output_str);
  std::cout << "output_str = '" << output_str << "'\n";
}

#endif
