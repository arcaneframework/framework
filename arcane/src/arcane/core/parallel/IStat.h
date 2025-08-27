// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStat.h                                                     (C) 2000-2023 */
/*                                                                           */
/* Statistiques sur les messages de 'IParallelMng'.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_ISTAT_H
#define ARCANE_PARALLEL_ISTAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/utils/JSONWriter.h"
#include "arccore/message_passing/IStat.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Properties;
}

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur le parallélisme.
 * \todo rendre thread-safe
 */
class IStat
{
 public:

  //! Libère les ressources.
  virtual ~IStat() {}

 public:

  /*!
   * \brief Ajoute une statistique.
   *
   * \param name nom de la statistique.
   * \param elapsed_time temps utilisé pour le message.
   * \param msg_size taille du message envoyé.
   */
  virtual void add(const String& name,double elapsed_time,Int64 msg_size) =0;
  
  //! Imprime sur \a trace les statistiques.
  virtual void print(ITraceMng* trace) =0;

  /*!
   * \brief Affiche les statistiques de manière collective.
   *
   * Affiche les statistiques communes à tous les rangs associés à \a pm.
   *
   * Cette opération est collective.
   */
  virtual void printCollective(IParallelMng* pm) = 0;

  //! Active ou désactive les statistiques
  virtual void enable(bool is_enabled) =0;

  //! Sort les statistiques au format JSON
  virtual void dumpJSON(JSONWriter& writer) =0;

 public:

  //! Sauve les valeurs actuelles dans \a p
  virtual void saveValues(ITraceMng* tm, Properties* p) =0;

  //! Fusionne les valeurs actuelles avec celles sauvées dans \a p
  virtual void mergeValues(ITraceMng* tm, Properties* p) =0;

 public:

  virtual Arccore::MessagePassing::IStat* toArccoreStat() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Créé une instance par défaut
extern "C++" ARCANE_CORE_EXPORT IStat*
createDefaultStat();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Free function pour le dump d'une stat de message dans un JSON
extern "C++" ARCANE_CORE_EXPORT void
dumpJSON(JSONWriter& writer, const Arccore::MessagePassing::OneStat& os, bool cumulative_stat = true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Affiche les statistiques cumulées sur l'ensemble des rangs de \a pm.
 */
extern "C++" ARCANE_DEPRECATED_REASON("Y2023: Use IStat::printCollective() instead")
ARCANE_CORE_EXPORT void printStatsCollective(IStat* s, IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

