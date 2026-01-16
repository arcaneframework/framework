// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Process.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Gestion des processus.                                                    */
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
    // Le processus fils a terminé correctement (mais peut être en erreur)
    OK,
    //! Le fork() a échoué
    CanNotFork,
    //! L'appel à pipe2() a échoué
    CanNotCreatePipe,
    // Le processus fils n'a pas terminé correctement (void waitpid())
    AbnormalExit
  };

 public:

  //! Commande à exécuter. Doit correspondre à un exécutable.
  String command() const { return m_command; }
  void setCommand(const String& v) { m_command = v; }

  //! Liste des arguments
  ConstArrayView<String> arguments() const { return m_arguments; }
  void addArguments(const String& v) { m_arguments.add(v); }
  void setArguments(const Array<String>& v) { m_arguments = v; }

  //! Chaîne de caractères à envoyer sur l'entrée standard (STDIN) du processsus.
  ConstArrayView<Byte> inputBytes() const { return m_input_bytes; }
  void setInputBytes(ConstArrayView<Byte> s) { m_input_bytes = s; }

  //! Contient le résultat de la sortie standard (STDOUT) du processus
  ConstArrayView<Byte> outputBytes() const { return m_output_bytes; }
  //! Code de retour du processus exécuté.
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
 * \brief Classe permettant d'exécuter un processus externe.
 */
class ARCCORE_COMMON_EXPORT Process
{
 public:

  //! Exécute un processus dont les infos sont contenues dans \a args.
  static ProcessExecArgs::ExecStatus execute(ProcessExecArgs& args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
