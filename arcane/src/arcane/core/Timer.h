// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Timer.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Gestion d'un timer.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TIMER_H
#define ARCANE_CORE_TIMER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimerMng;
class ITimeStats;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'un timer.

 * Une instance de cette classe permet de mesurer le temps passé entre son
 * activation par la méthode start() et son arrêt par la méthode stop().

 * Le timer peut être utilisé plusieurs fois et il est possible de
 * connaître à la fois le nombre d'activation (nbActivated()) et le temps
 * total passé dans ses activations successives (totalTime()).

 * Il existe deux modes de fonctionnement:
 * <ul>
 * <li>#TimerVirtual: le timer utilise le temps CPU du processus. Ce temps
 * est constant quelle que soit la charge de la machine;</li>
 * <li>#TimerReal: le timer utilise le temps réel. La résolution de ce timer est
 * en général meilleure qu'avec le type précédent mais elle n'est significative
 * que lorsque la machine est dédiée au processus.</li>
 * </ul>
 *
 * \note Depuis la version 3.6 de %Arcane, le #TimerVirtual est obsolète et
 * la valeur retournée sera équivalent à #TimerReal.
 *
* La résolution du timer dépend de la machine. Elle est de l'ordre de la
 * milliseconde pour les timers utilisant le temps CPU et de l'ordre de
 * la microseconde pour les timers utilisant le temps réel.
 */
class ARCANE_CORE_EXPORT Timer
{
 public:

  //! Type du timer
  enum eTimerType
  {
    /*!
     * \brief Timer utilisant le temps CPU (obsolète).
     *
     * \deprecated Ce timer n'est plus utilisé et se comporte comme
     * le temps horloge (TimerReal).
     */
    TimerVirtual,
    //! Timer utilisant le temps réel
    TimerReal
  };

 public:
  
  /*!
   * \brief Sentinelle pour le timer.
   * La sentinelle associée à un timer permet de déclancher celui-ci
   * au moment de sa construction et de l'arrêter au moment de sa
   * destruction. Cela assure que le timer sera bien arrêté en cas
   * d'exception par exemple.
   */
  class ARCANE_CORE_EXPORT Sentry
  {
   public:
    //! Associe le timer \a t et le démarre
    Sentry(Timer* t) : m_timer(t)
      { m_timer->start(); }
    //! Stoppe le timer associé
    ~Sentry()
      { m_timer->stop(); }
   private:
    Timer* m_timer; //!< Timer associé
  };

  /*!
   * \brief Postionne le nom de l'action en cours d'exécution.
   *
   * Le nom d'une action peut-être n'importe quoi. Il est
   * juste utilisé pour différencier les différentes partie d'une
   * exécution et connaître le temps de chacune d'elle.
   * Les actions doivent s'imbriquent les unes dans les autres
   */
  class ARCANE_CORE_EXPORT Action
  {
   public:
    Action(ISubDomain* sub_domain,const String& action_name,bool print_time=false);
    Action(ITimeStats* stats,const String& action_name,bool print_time=false);
    ~Action();
   public:
   private:
    ITimeStats* m_stats;
    String m_action_name;
    bool m_print_time;
   private:
    void _init();
  };

  /*!
   * \brief Positionne la phase de l'action en cours d'exécution.
   */
  class ARCANE_CORE_EXPORT Phase
  {
   public:
   public:
    Phase(ISubDomain* sub_domain,eTimePhase pt);
    Phase(ITimeStats* stats,eTimePhase pt);
    ~Phase();
   public:
   private:
    ITimeStats* m_stats; //!< Gestionnaire de sous-domaine
    eTimePhase m_phase_type;
   private:
    void _init();
  };

  /*!
   * \brief Affiche le temps passé entre l'appel au constructeur et le destructeur.
   *
   * Cette classe permet de simplement afficher au moment du destructeur,
   * le temps réel écoulé depuis l'appel au constructeur. L'affichage se fait
   * via la méthode info() du ITraceMng.
   * \code
   * {
   *   Timer::SimplePrinter sp(traceMng(),"myFunction");
   *   myFunction();
   * }
   * \endcode
   */
  class ARCANE_CORE_EXPORT SimplePrinter
  {
   public:
    SimplePrinter(ITraceMng* tm,const String& msg);
    SimplePrinter(ITraceMng* tm,const String& msg,bool is_active);
    ~SimplePrinter();
   private:
    ITraceMng* m_trace_mng;
    Real m_begin_time;
    bool m_is_active;
    String m_message;
   private:
    void _init();
  };

 public:

  /*!
   * \brief Construit un timer.
   *
   * Construit un timer lié au sous-domaine \a sd, de nom \a name et de
   * type \a type.
   */
  Timer(ISubDomain* sd,const String& name,eTimerType type);

  /*!
   * \brief Construit un timer.
   *
   * Construit un timer lié au gestionnaire \a tm, de nom \a name et de
   * type \a type.
   */
  Timer(ITimerMng* tm,const String& name,eTimerType type);

  ~Timer(); //!< Libère les ressources

 public:
	
  /*!
   * \brief Active le timer.
   *
   * Si le timer est déjà actif, cette méthode ne fait rien.
   */
  void start();

  /*!
   * \brief Désactive le timer.
   *
   * Si le timer n'est pas actif au moment de l'appel, cette méthode ne
   * fait rien.
   *
   * \return le temps écoulé (en secondes) depuis la dernière activation.
   */
  Real stop();

  //! Retourne l'état d'activation du timer
  bool isActivated() const { return m_is_activated; }

  //! Retourne le nom du timer
  const String& name() const { return m_name; }

  //! Retourne le temps total (en secondes) passé dans le timer
  Real totalTime() const { return m_total_time; }

  //! Retourne le temps (en secondes) passé lors de la dernière activation du timer
  Real lastActivationTime() const { return m_activation_time; }

  //! Retourne le nombre de fois que le timer a été activé
  Integer nbActivated() const { return m_nb_activated; }

  //! Retourne le type du temps utilisé
  eTimerType type() const { return m_type; }

  //! Remet à zéro les compteurs de temps
  void reset();

  //! Gestionnaire associé à ce timer.
  ITimerMng* timerMng() const { return m_timer_mng; }
 public:
  static TimeMetricAction phaseAction(ITimeStats* s,eTimePhase phase);
 public:
  //! \internal
  void _setStartTime(Real t) { m_start_time = t; }
  //! \internal
  Real _startTime() const { return m_start_time; }
 private:

  ITimerMng* m_timer_mng; //!< Gestionnaire de timer
  eTimerType m_type; //!< Type du timer
  Integer m_nb_activated; //!< Nombre de fois que le timer a été activé
  bool m_is_activated; //!< \a true si le timer est actif
  Real m_activation_time; //!< Temps passé lors de la dernière activation
  Real m_total_time; //!< Temps total passé dans le timer
  String m_name; //!< Nom du timer
  Real m_start_time; //!< Temps du début de la dernière activation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

