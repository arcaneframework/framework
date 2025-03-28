// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'une exception.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_EXCEPTION_H
#define ARCCORE_BASE_EXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/StackTrace.h"
// On n'a pas explicitement besoin de ce .h mais il est plus simple
// de l'avoir pour pouvoir facilement lancer des exceptions avec les traces
#include "arccore/base/TraceInfo.h"

#include <exception>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une exception.
 *
 * \ingroup Core
 Les exceptions sont gérées par le mécanisme <tt>try</tt> et <tt>catch</tt>
 du C++. Toutes les exceptions lancées dans le code <strong>doivent</strong>
 dériver de cette classe.

 Une exception peut-être collective. Cela signifie qu'elle sera lancée
 par tous les processeurs. Il est possible dans ce cas de n'afficher qu'une
 seule fois le message et éventuellement d'arrêter proprement l'exécution.
 */
class ARCCORE_BASE_EXPORT Exception
: public std::exception
{
 private:

  /*!
   * \internal
   *
   * Cette méthode est privée pour interdire d'affecter une exception.
   */
  const Exception& operator=(const Exception&); //PURE

 public:

  /*!
   * Construit une exception de nom \a name et
   * envoyée depuis la fonction \a where.
   */
  Exception(const String& name,const String& where);
  /*!
   * Construit une exception de nom \a name et
   * envoyée depuis la fonction \a where.
   */
  Exception(const String& name,const TraceInfo& where);
  /*!
   * Construit une exception de nom \a name,
   * envoyée depuis la fonction \a where et avec le message \a message.
   */
  Exception(const String& name,const String& where,const String& message);
  /*!
   * Construit une exception de nom \a name,
   * envoyée depuis la fonction \a where et avec le message \a message.
   */
  Exception(const String& name,const TraceInfo& trace,const String& message);
  /*!
   * Construit une exception de nom \a name et
   * envoyée depuis la fonction \a where.
   */
  Exception(const String& name,const String& where,const StackTrace& stack_trace);
  /*!
   * Construit une exception de nom \a name et
   * envoyée depuis la fonction \a where.
   */
  Exception(const String& name,const TraceInfo& where,const StackTrace& stack_trace);
  /*!
   * Construit une exception de nom \a name,
   * envoyée depuis la fonction \a where et avec le message \a message.
   */
  Exception(const String& name,const String& where,
            const String& message,const StackTrace& stack_trace);
  /*!
   * Construit une exception de nom \a name,
   * envoyée depuis la fonction \a where et avec le message \a message.
   */
  Exception(const String& name,const TraceInfo& trace,
            const String& message,const StackTrace& stack_trace);
  //! Constructeur par copie.
  Exception(const Exception&);
  //! Libère les ressources
  ~Exception() ARCCORE_NOEXCEPT override;

 public:
 
  virtual void write(std::ostream& o) const;

  //! Vrai s'il s'agit d'une erreur collective (concerne tous les processeurs)
  bool isCollective() const { return m_is_collective; }

  //! Positionne l'état collective de l'expression
  void setCollective(bool v) { m_is_collective = v; }

  //! Positionne les infos supplémentaires
  void setAdditionalInfo(const String& v) { m_additional_info = v; }

  //! Retourne les infos supplémentaires
  const String& additionalInfo() const { return m_additional_info; }

  //! Pile d'appel au moment de l'exception (nécessite un service de stacktrace)
  const StackTrace& stackTrace() const { return m_stack_trace; }

  //! Pile d'appel au moment de l'exception (nécessite un service de stacktrace)
  const String& stackTraceString() const { return m_stack_trace.toString(); }

  //! Indique si des exceptions sont en cours
  static bool hasPendingException();

  static void staticInit();

  //! Message de l'exception
  const String& message() const { return m_message; }

  //! Localisation de l'exception
  const String& where() const { return m_where; }

  //! Nom de l'exception
  const String& name() const { return m_name; }

 protected:
  
  /*! \brief Explique la cause de l'exception dans le flot \a o.
   *
   * Cette méthode permet d'ajouter des informations supplémentaires
   * au message d'exception.
   */
  virtual void explain(std::ostream& o) const;

  //! Positionne le message de l'exception
  void setMessage(const String& msg)
  {
    m_message = msg;
  }

 private:

  String m_name;
  String m_where;
  StackTrace m_stack_trace;
  String m_message;
  String m_additional_info;
  bool m_is_collective;

  void _setStackTrace();
  void _setWhere(const TraceInfo& where);
  void _checkExplainAndPause();

 private:

  static std::atomic<Int32> m_nb_pending_exception;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT std::ostream&
operator<<(std::ostream& o,const Exception& ex);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

