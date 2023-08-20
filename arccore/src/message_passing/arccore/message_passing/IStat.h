// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2023 */
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
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistique sur un message.
 */
class ARCCORE_MESSAGEPASSING_EXPORT OneStat
{
 public:

  explicit OneStat(const String& name, Int64 msg_size = 0, double elapsed_time = 0.0);

 public:

  Int64 cumulativeNbMessage() const { return m_cumulative_nb_msg; }

  Int64 totalSize() const { return m_total_size; }
  Int64 cumulativeTotalSize() const { return m_cumulative_total_size; }

  double totalTime() const { return m_total_time; }
  double cumulativeTotalTime() const { return m_cumulative_total_time; }

  const String& name() const { return m_name; }
  Int64 nbMessage() const { return m_nb_msg; }

  void print(std::ostream& o);
  void addMessage(Int64 msg_size, double elapsed_time);
  void resetCurrentStat();

 private:

  String m_name; //!< Nom de la statistique
  Int64 m_nb_msg = 0; //!< Nombre de message envoyés.
  Int64 m_total_size = 0; //!< Taille total des messages envoyés
  double m_total_time = 0.0; //!< Temps total écoulé
  Int64 m_cumulative_nb_msg = 0; //! < Nombre de message envoyés sur toute la duree d'execution du programme
  Int64 m_cumulative_total_size = 0; //!< Taille total des messages envoyés sur toute la duree d'execution du programme
  double m_cumulative_total_time = 0.0; //!< Temps total écoulé sur toute la duree d'execution du programme};
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

  // DEPRECATED
  using OneStatMap = std::map<String, OneStat*>;

 public:

  //! Libère les ressources.
  virtual ~IStat() = default;

 public:

  /*!
   * \brief Ajoute une statistique.
   *
   * \param name nom de la statistique
   * \param elapsed_time temps utilisé pour le message
   * \param msg_size taille du message envoyé.
   */
  virtual void add(const String& name, double elapsed_time, Int64 msg_size) = 0;

  /*!
   * \brief Active ou désactive les statistiques.
   *
   * Si les statistiques sont désactivées, l'appel à add() est sans effet.
   */
  virtual void enable(bool is_enabled) = 0;

  //! Récuperation des statistiques
  virtual const OneStatMap& stats() const = 0;

  //! Remèt à zéro les statistiques courantes
  virtual void resetCurrentStat() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

