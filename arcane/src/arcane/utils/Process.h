// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Process.h                                                   (C) 2000-2016 */
/*                                                                           */
/* Gestion des processus.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PROCESS_H
#define ARCANE_UTILS_PROCESS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT ProcessExecArgs
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
  StringConstArrayView arguments() const { return m_arguments; }
  void addArguments(const String& v) { m_arguments.add(v); }
  void setArguments(const StringArray& v) { m_arguments = v; }

  //! Chaîne de caractère à envoyer sur l'entrée standard (STDIN) du processsus.
  ByteConstArrayView inputBytes() const { return m_input_bytes; }
  void setInputBytes(ByteConstArrayView s) { m_input_bytes = s; }

  //! Contient le résultat de la sortie standard (STDOUT) du processus
  ByteConstArrayView outputBytes() const { return m_output_bytes; }
  //! Code de retour du processus exécuté.
  int exitCode() const { return m_exit_code; }

 private:

  String m_command;
  StringUniqueArray m_arguments;
  UniqueArray<Byte> m_input_bytes;
  UniqueArray<Byte> m_output_bytes;
  int m_exit_code;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant d'exécuter un processus externe.
 */
class ARCANE_UTILS_EXPORT Process
{
 public:
  //! Exécute un processus dont les infos sont contenues dans \a args.
  static ProcessExecArgs::ExecStatus execute(ProcessExecArgs& args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

