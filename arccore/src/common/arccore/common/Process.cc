// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Process.cc                                                  (C) 2000-2025 */
/*                                                                           */
/* Gestion des processus.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/Process.h"

#include "arccore/base/NotImplementedException.h"

/*
 * NOTE: pour l'instant cette classe n'est implémentée que pour Linux
 * (elle devrait cependant fonctionner avec les autres Unix).
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

  // Créé deux pipes pour rediriger les entrées et les sorties du
  // processus qu'on va lancer.
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
    // Je suis le processus fils.

    // TODO: vérifier les erreurs des close() et dup2().

    // Indique que pipefd_out[1] correspond à mon STDOUT
    ::close(STDOUT_FILENO);
    ::close(pipefd_out[0]);
    ::dup2(pipefd_out[1], STDOUT_FILENO);

    // Indique que pipefd_in[0] correspond à mon STDIN
    ::close(STDIN_FILENO);
    ::close(pipefd_in[1]);
    ::dup2(pipefd_in[0], STDIN_FILENO);

    const char* cmd_name = args.command().localstr();

    ConstArrayView<String> arguments = args.arguments();
    Integer nb_arg = arguments.size();
    // Le tableau passé à execve() pour les arguments doit se terminer par NULL
    // et commencer par le nom de l'exécutable
    UniqueArray<const char*> command_args(nb_arg + 2);
    for (Integer i = 0; i < nb_arg; ++i)
      command_args[i + 1] = arguments[i].localstr();
    command_args[0] = cmd_name;
    command_args[nb_arg + 1] = nullptr;

    const char* const newenviron[] = { NULL };
    ::execve(cmd_name, (char* const*)command_args.data(), (char* const*)newenviron);
    // L'appel à execve() ne retourne pas.
  }
  else {
    ::close(pipefd_out[1]);
    ::close(pipefd_in[0]);
    // Ecrit sur le pipe d'entrée les octets de \a input_bytes
    Int64 nb_wanted_write = input_bytes.size();
    Int64 nb_written = ::write(pipefd_in[1], input_bytes.data(), nb_wanted_write);
    if (nb_written != nb_wanted_write)
      std::cerr << "Error writing to pipe\n";
    ::close(pipefd_in[1]);
    const int BUF_SIZE = 4096;
    Byte buf[BUF_SIZE];
    ssize_t r = 0;
    // TODO: gérer les interruptions et recommencer si nécessaire
    while ((r = ::read(pipefd_out[0], buf, BUF_SIZE)) > 0) {
      args.m_output_bytes.addRange(ByteConstArrayView((Integer)r, buf));
      //::write(STDOUT_FILENO, buf, r);
    }

    // Attend que le processus fils soit fini.
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

    // Ajoute un '\0' terminal au flux de sortie.
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
