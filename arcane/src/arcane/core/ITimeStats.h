// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeStats.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface gérant les statistiques sur les temps d'exécution.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMESTATS_H
#define ARCANE_CORE_ITIMESTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Properties;
class Timer;
class JSONWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface gérant les statistiques sur les temps d'exécution.
 *
 * Il faut appeler beginGatherStats() pour commencer à collecter les
 * informations et appeler endGatherStats() pour arrêter la collection.
 *
 * En général cette interface ne s'utilise pas directement mais par
 * l'intermédiaire des classes Timer::Phase et Timer::Action.
 *
 * Les méthodes de cette classe ne doivent être appelées que par un seul thread.
 */
class ITimeStats
{
 public:

  // Libère les ressources.
  virtual ~ITimeStats() = default;

 public:

  //! Démarre la collection des temps
  virtual void beginGatherStats() = 0;
  //! Arrête la collection des temps
  virtual void endGatherStats() = 0;

 public:

  virtual void beginAction(const String& action_name) = 0;
  virtual void endAction(const String& action_name, bool print_time) = 0;

  virtual void beginPhase(eTimePhase phase) = 0;
  virtual void endPhase(eTimePhase phase) = 0;

 public:

  /*!
   * \brief Temps réel écoulé pour la phase \a phase
   *
   * Retourne le temps réel écoulé (en seconde) pour la phase \a phase.
   */
  virtual Real elapsedTime(eTimePhase phase) = 0;

  /*!
   * \brief Temps écoulé pour une phase d'une action.
   *
   * Retourne le temps réel écoulé (en seconde) pour la phase \a phase
   * de l'action \a action. Le temps retourné est celui de l'action et
   * de chacune de ses filles.
   */
  virtual Real elapsedTime(eTimePhase phase, const String& action) = 0;

  /*!
   * \brief Affiche les statistiques d'une action.
   *
   * Affiche les statistiques de l'action \a name ainsi que ces sous-actions
   * pour l'itération courante.
   */
  virtual void dumpCurrentStats(const String& name) = 0;

 public:

  /*!
   * \brief Affiche les statistiques sur les temps d'exécution.
   *
   * Il est possible de spécifier une valeur pour avoir un temps
   * par itération ou par entité. Si \a use_elapsed_time est vrai,
   * utilise le temps horloge, sinon utilise le temps CPU.
   */
  virtual void dumpStats(std::ostream& ostr, bool is_verbose, Real nb,
                         const String& name, bool use_elapsed_time = false) = 0;

  /*!
   * \brief Affiche la date actuelle et la mémoire consommée.
   *
   * Cette opération est collective sur \a pm.
   *
   * Cette opération affiche la mémoire consommée pour le sous-domaine
   * courant ainsi que le min et le max pour tous les sous-domaines.
   */
  virtual void dumpTimeAndMemoryUsage(IParallelMng* pm) = 0;

  /*!
   * \brief Indique si les statistiques sont actives.
   *
   * Les statistiques sont actives entre l'appel à beginGatherStats()
   * et endGatherStats().
   */
  virtual bool isGathering() const = 0;

  //! Sérialise dans l'écrivain \a writer les statistiques temporelles.
  virtual void dumpStatsJSON(JSONWriter& writer) = 0;

  //! Interface de collection associée
  virtual ITimeMetricCollector* metricCollector() = 0;

  /*!
   * \brief Notifie qu'on commence une nouvelle itération de la boucle de calcul.
   *
   * Cette information est utilisée pour calculer les temps par itération.
   */
  virtual void notifyNewIterationLoop() = 0;
  virtual void saveTimeValues(Properties* p) = 0;
  virtual void mergeTimeValues(Properties* p) = 0;

  /*
   * \brief Remet à zéro les statistiques courantes une action est ses sous-actions
   *
   * Remet à zéro les statistiques pour l'action \a action_name est ses
   * sous-actions. Si aucune action de nom \a action_name n'existe, ne fait rien.
   *
   * Cette méthode est réservée pour les tests et ne doit pas être utilisée
   * en dehors de cette configuration pour éviter de rendre invalides les
   * statistiques temporelles.
   */
  virtual void resetStats(const String& action_name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

