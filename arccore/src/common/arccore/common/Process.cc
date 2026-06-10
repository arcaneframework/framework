// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Process.cc                                                  (C) 2000-2025 */
/*                                                                           */
/* Process management.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/Process.h"

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/FixedArray.h"

/*
 * NOTE: for now this class is only implemented for Linux
 * (it should however work with other Unix systems).
 */

#ifdef ARCCORE_OS_LINUX
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProcessExecArgs::ExecStatus Process::
execute(ProcessExecArgs& args)
{
#ifdef ARCCORE_OS_LINUX
  args.m_output_bytes.clear();
  ByteConstArrayView input_bytes = args.inputBytes();

  // Create two pipes to redirect the inputs and outputs of the
  // process we are going to launch.
  int pipefd_out[2];
  int pipefd_in[2];
  int r0 = pipe2(pipefd_out, O_CLOEXEC);
  if (r0 != 0)
    return ProcessExecArgs::ExecStatus::CanNotCreatePipe;
  r0 = pipe2(pipefd_in, O_CLOEXEC);
  if (r0 != 0)
    return ProcessExecArgs::ExecStatus::CanNotCreatePipe;
  pid_t cpid = ::fork();
  if (cpid < 0)
    return ProcessExecArgs::ExecStatus::CanNotFork;

  ProcessExecArgs::ExecStatus exec_status = ProcessExecArgs::ExecStatus::OK;
  if (cpid == 0) {
    // I am the child process.

    // TODO: check errors for close() and dup2().

    // Indicates that pipefd_out[1] corresponds to my STDOUT
    ::close(STDOUT_FILENO);
    ::close(pipefd_out[0]);
    ::dup2(pipefd_out[1], STDOUT_FILENO);

    // Indicates that pipefd_in[0] corresponds to my STDIN
    ::close(STDIN_FILENO);
    ::close(pipefd_in[1]);
    ::dup2(pipefd_in[0], STDIN_FILENO);

    const char* cmd_name = args.command().localstr();

    ConstArrayView<String> arguments = args.arguments();
    Integer nb_arg = arguments.size();
    // The array passed to execve() for arguments must end with NULL
    // and start with the name of the executable
    UniqueArray<const char*> command_args(nb_arg + 2);
    for (Integer i = 0; i < nb_arg; ++i)
      command_args[i + 1] = arguments[i].localstr();
    command_args[0] = cmd_name;
    command_args[nb_arg + 1] = nullptr;

    const char* const newenviron[] = { NULL };
    ::execve(cmd_name, (char* const*)command_args.data(), (char* const*)newenviron);
    // The execve() call does not return.
  }
  else {
    ::close(pipefd_out[1]);
    ::close(pipefd_in[0]);
    // Write the bytes of \a input_bytes to the input pipe
    Int64 nb_wanted_write = input_bytes.size();
    Int64 nb_written = ::write(pipefd_in[1], input_bytes.data(), nb_wanted_write);
    if (nb_written != nb_wanted_write)
      std::cerr << "Error writing to pipe\n";
    ::close(pipefd_in[1]);
    const int BUF_SIZE = 4096;
    FixedArray<Byte, BUF_SIZE + 1> buf;
    buf[BUF_SIZE] = '\0';
    Int32 max_iteration = 1000000;
    Int32 current_iteration = 0;
    // Uses a finite loop to avoid coverity/codacy warnings
    for (Int32 i = 0; i < max_iteration; ++i) {
      ssize_t nb_read = ::read(pipefd_out[0], buf.data(), BUF_SIZE);
      if (nb_read == EINTR)
        continue;
      if (nb_read <= 0)
        break;
      Int32 i_nb_read = static_cast<Int32>(nb_read);
      buf[i_nb_read] = '\0';
      args.m_output_bytes.addRange(buf.view().subView(0, i_nb_read));
      //::write(STDOUT_FILENO, buf, r);
    }

    // Wait for the child process to finish.
    int status = 0;
    pid_t child_pid = 0;
    do {
      child_pid = ::waitpid(cpid, &status, 0); /* Wait for child */
    } while (child_pid == -1 && errno == EINTR);

    if (WIFEXITED(status)) {
      args.m_exit_code = WEXITSTATUS(status);
      //printf("exited, status=%d\n", WEXITSTATUS(status));
    }
    else
      exec_status = ProcessExecArgs::ExecStatus::AbnormalExit;

    close(pipefd_out[0]);

    // Adds a terminal '\0' to the output stream.
    args.m_output_bytes.add('\0');
  }
  return exec_status;
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
