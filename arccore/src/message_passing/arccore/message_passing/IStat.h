// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2025 */
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
#include <list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistique sur un message.
 */
class ARCCORE_MESSAGEPASSING_EXPORT OneStat
{
 public:

  explicit OneStat(const String& name)
  : m_name(name)
  {}
  OneStat(const String& name, Int64 msg_size, double elapsed_time);

 public:

  //! Nom de la statistique
  const String& name() const { return m_name; }

  //! Nombre de message envoyés.
  Int64 nbMessage() const { return m_nb_msg; }
  void setNbMessage(Int64 v) { m_nb_msg = v; }

  //! Nombre de message envoyés sur toute la durée d'exécution
  Int64 cumulativeNbMessage() const { return m_cumulative_nb_msg; }
  void setCumulativeNbMessage(Int64 v) { m_cumulative_nb_msg = v; }

  //! Taille totale des messages envoyés
  Int64 totalSize() const { return m_total_size; }
  void setTotalSize(Int64 v) { m_total_size = v; }

  //! Taille totale des messages envoyés sur toute la durée d'exécution
  Int64 cumulativeTotalSize() const { return m_cumulative_total_size; }
  void setCumulativeTotalSize(Int64 v) { m_cumulative_total_size = v; }

  //! Temps total écoulé
  double totalTime() const { return m_total_time; }
  void setTotalTime(double v) { m_total_time = v; }

  //! Temps total écoulé sur toute la durée d'exécution du programme
  double cumulativeTotalTime() const { return m_cumulative_total_time; }
  void setCumulativeTotalTime(double v) { m_cumulative_total_time = v; }

 public:

  //! Affiche sur \a o les informations de l'instance
  void print(std::ostream& o);

  //! Ajoute un message
  void addMessage(Int64 msg_size, double elapsed_time);

  //! Remet à zéro les statistiques courantes (non cumulées)
  void resetCurrentStat();

 private:

  String m_name;
  Int64 m_nb_msg = 0;
  Int64 m_total_size = 0;
  double m_total_time = 0.0;
  Int64 m_cumulative_nb_msg = 0;
  Int64 m_cumulative_total_size = 0;
  double m_cumulative_total_time = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de statistiques.
 *
 * Il est uniquement possible d'itérer sur les éléments de la collection.
 *
 * L'implémentation utilisée peut évoluer et il ne faut donc pas utiliser
 * le type explicite de l'itérateur.
 */
class ARCCORE_MESSAGEPASSING_EXPORT StatCollection
{
  friend class StatData;
  using Impl = std::list<OneStat>;

 public:

  using const_iterator = Impl::const_iterator;

 public:

  const_iterator begin() const { return m_stats.begin(); }
  const_iterator end() const { return m_stats.end(); }
  const_iterator cbegin() const { return m_stats.begin(); }
  const_iterator cend() const { return m_stats.end(); }

 private:

  Impl m_stats;
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
  virtual const StatCollection& statList() const = 0;

  //! Remèt à zéro les statistiques courantes
  virtual void resetCurrentStat() = 0;

 public:

  //! Récuperation des statistiques
  ARCCORE_DEPRECATED_REASON("Y2023: Use statList() instead")
  virtual const OneStatMap& stats() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

