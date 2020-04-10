// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2019 */
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

namespace Arccore
{

namespace MessagePassing
{

//! Une statistique
class ARCCORE_MESSAGEPASSING_EXPORT OneStat
{
 public:
  OneStat(const String& name, const Int64 msg_size = 0, const double elapsed_time = 0.0)
  : m_name(name), m_nb_msg(0), m_total_size(msg_size), m_total_time(elapsed_time),
    m_cumulative_nb_msg(0), m_cumulative_total_size(0), m_cumulative_total_time(0.0) {}

  OneStat(const OneStat&) = default;
  OneStat(OneStat&&) = default;
  OneStat& operator=(const OneStat&) = default;
  void print(std::ostream& o);

 public:

  Int64 cumulativeNbMessage() const { return m_cumulative_nb_msg; }

  Int64 totalSize() const { return m_total_size; }
  Int64 cumulativeTotalSize() const { return m_cumulative_total_size; }

  double totalTime() const { return m_total_time; }
  double cumulativeTotalTime() const { return m_cumulative_total_time; }

  void addMessage(Int64 msg_size, double elapsed_time)
  {
    ++m_nb_msg;
    m_total_size += msg_size;
    m_total_time += elapsed_time;
    ++m_cumulative_nb_msg;
    m_cumulative_total_size += msg_size;
    m_cumulative_total_time += elapsed_time;
  }

  const String& name() const { return m_name; }
  Int64 nbMessage() const { return m_nb_msg; }
  void resetCurrentStat() { m_nb_msg = 0; m_total_size = 0; m_total_time = 0.0; }
  
 private:
  String m_name; //!< Nom de la statistique
  Int64 m_nb_msg; //!< Nombre de message envoyés.
  Int64 m_total_size; //!< Taille total des messages envoyés
  double m_total_time; //!< Temps total écoulé
  Int64 m_cumulative_nb_msg;  //! < Nombre de message envoyés sur toute la duree d'execution du programme
  Int64 m_cumulative_total_size; //!< Taille total des messages envoyés sur toute la duree d'execution du programme
  double m_cumulative_total_time; //!< Temps total écoulé sur toute la duree d'execution du programme};
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
  using OneStatMap = std::map<String,OneStat*>;

 public:

  //! Libère les ressources.
  virtual ~IStat() {}

 public:

  /*!
   * \brief Ajoute une statistique.
   *
   * \param name nom de la statistique
   * \param elapsed_time temps utilisé pour le message
   * \param msg_size taille du message envoyé.
   */
  virtual void add(const String& name,double elapsed_time,Int64 msg_size) =0;
  
  //! Active ou désactive les statistiques
  virtual void enable(bool is_enabled) =0;

  //! Récuperation des statistiques
  virtual const OneStatMap& stats() const =0;

  //! Remèt à zéro les statistiques courantes
  virtual void resetCurrentStat() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

