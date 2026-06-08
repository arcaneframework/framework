// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Process.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Process management.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_PROCESS_H
#define ARCCORE_COMMON_INTERNAL_PROCESS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"

#include "arccore/common/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT ProcessExecArgs
{
  friend class Process;

 public:

  enum class ExecStatus
  {
    // The child process terminated correctly (but may have errors)
    OK,
    //! fork() failed
    CanNotFork,
    //! The call to pipe2() failed
    CanNotCreatePipe,
    // The child process did not terminate correctly (void waitpid())
    AbnormalExit
  };

 public:

  //! Command to execute. Must correspond to an executable.
  String command() const { return m_command; }
  void setCommand(const String& v) { m_command = v; }

  //! List of arguments
  ConstArrayView<String> arguments() const { return m_arguments; }
  void addArguments(const String& v) { m_arguments.add(v); }
  void setArguments(const Array<String>& v) { m_arguments = v; }

  //! String to send to the process's standard input (STDIN).
  ConstArrayView<Byte> inputBytes() const { return m_input_bytes; }
  void setInputBytes(ConstArrayView<Byte> s) { m_input_bytes = s; }

  //! Contains the result of the process's standard output (STDOUT).
  ConstArrayView<Byte> outputBytes() const { return m_output_bytes; }
  //! Return code of the executed process.
  int exitCode() const { return m_exit_code; }

 private:

  String m_command;
  UniqueArray<String> m_arguments;
  UniqueArray<Byte> m_input_bytes;
  UniqueArray<Byte> m_output_bytes;
  int m_exit_code = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the execution of an external process.
 */
class ARCCORE_COMMON_EXPORT Process
{
 public:

  //! Executes a process whose information is contained in args.
  static ProcessExecArgs::ExecStatus execute(ProcessExecArgs& args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
