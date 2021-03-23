// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeStats.h                                                (C) 2000-2021 */
/*                                                                           */
/* Interface gérant les statistiques sur les temps d'exécution.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITIMESTATS_H
#define ARCANE_ITIMESTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Arccore::ITimeMetricCollector;
class Properties;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Timer;
class JSONWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface gérant les statistiques sur les temps d'exécution.
 */
class ITimeStats
{
 public:

  // Libère les ressources.
  virtual ~ITimeStats() = default;

 public:

  virtual void beginGatherStats() =0;
  virtual void endGatherStats() =0;

 public:

  virtual void beginAction(const String& action_name) =0;
  virtual void endAction(const String& action_name,bool print_time) =0;

  virtual void beginPhase(eTimePhase phase) =0;
  virtual void endPhase(eTimePhase phase) =0;

 public:

  /*!
    \brief Retourne le temps réel écoulé (en seconde) pour la phase \a phase
    depuis l'appel à beginGatherStats().
  */
  virtual Real elapsedTime(eTimePhase phase) =0;

  /*!
    \brief Retourne le temps réel écoulé (en seconde) pour la phase \a phase
    de l'action \a action depuis l'appel à beginGatherStats(). Le temps
    retourné est celui de l'action et de chacune de ses filles.
  */
  virtual Real elapsedTime(eTimePhase phase,const String& action) =0;

  //! Affiche les statistiques de l'action \a name pour l'itération courante
  virtual void dumpCurrentStats(const String& name) =0;

 public:

  /*!
   * \brief Affiche les statistiques sur les temps d'exécution.
   *
   * Il est possible de spécifier une valeur pour avoir un temps
   * par itération ou par entité. Si \a use_elapsed_time est vrai,
   * utilise le temps horloge, sinon utilise le temps CPU.
   */
  virtual void dumpStats(std::ostream& ostr,bool is_verbose,Real nb,
                         const String& name, bool use_elapsed_time=false) =0;

  /*!
   * \brief Affiche la date actuelle et la mémoire consommée.
   *
   * Cette opération est collective sur \a pm.
   *
   * Cette opération affiche la mémoire consommée pour le sous-domaine
   * courant ainsi que le min et le max pour tous les sous-domaines.
   */
  virtual void dumpTimeAndMemoryUsage(IParallelMng* pm) =0;

  /*!
   * \brief Indique si les statistiques sont actives.
   *
   * Les statistiques sont actives entre l'appel à beginGatherStats()
   * et endGatherStats().
   */
  virtual bool isGathering() const =0;

  //! Sérialise dans l'écrivain \a writer les statistiques temporelles.
  virtual void dumpStatsJSON(JSONWriter& writer) =0;

  virtual ITimeMetricCollector* metricCollector() =0;

  /*!
   * \brief Notifie qu'on commence une nouvelle itération de la boucle de calcul.
   *
   * Cette information est utilisée pour calculer les temps par itération.
   */
  virtual void notifyNewIterationLoop() =0;
  virtual void saveTimeValues(Properties* p) =0;
  virtual void mergeTimeValues(Properties* p) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

