// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2018 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISTAT_H
#define ARCCORE_MESSAGEPASSING_ISTAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

#include "arccore/base/String.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace MessagePassing
{

//! Une statistique
class ARCCORE_MESSAGEPASSING_EXPORT OneStat
{
 public:
  OneStat(const String& name)
  : m_name(name), m_nb_msg(0), m_total_size(0), m_total_time(0.0)
  {
  }
  void print(std::ostream& o);
 public:
  const String& name() const { return m_name; }
  Int64 nbMessage() const { return m_nb_msg; }
  Int64 totalSize() const { return m_total_size; }
  double totalTime() const { return m_total_time; }
  void addMessage(Int64 msg_size,double elapsed_time)
  {
    m_total_size += msg_size;
    m_total_time += elapsed_time;
    ++m_nb_msg;
  }

  
 private:
  String m_name; //!< Nom de la statistique
  Int64 m_nb_msg; //!< Nombre de message envoyés.
  Int64 m_total_size; //!< Taille total des messages envoyés
  double m_total_time; //!< Temps total écoulé
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur le parallélisme.
 * \todo rendre thread-safe
 */
class ARCCORE_MESSAGEPASSING_EXPORT IStat
{
 public:

  //! Libère les ressources.
  virtual ~IStat() {}

 public:

 public:

  /*!
   * \brief Ajoute une statistique.
   * \param name nom de la statistique
   * \param elapsed_time temps utilisé pour le message
   * \param msg_size taille du message envoyé.
   */
  virtual void add(const String& name,double elapsed_time,Int64 msg_size) =0;
  
  //! Active ou désactive les statistiques
  virtual void enable(bool is_enabled) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

