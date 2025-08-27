// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeLoopMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire de la boucle en temps.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMELOOPMNG_H
#define ARCANE_CORE_ITIMELOOPMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IBackwardMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class eTimeLoopEventType
{
  BeginEntryPoint,
  EndEntryPoint,
  BeginIteration,
  EndIteration
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Raison pour laquelle on arrête le code.
 */
enum class eTimeLoopStopReason
{
  //! Indique qu'on n'est pas encore en mode d'arrête du code.
  NoStop =0,
  //! Pas de raison spécifique
  NoReason =1,
  //! Arrêt sur une erreur
  Error =2,
  //! Arrêt car temps final atteint
  FinalTimeReached =3,
  //! Arrêt car nombre d'itération maximal spécifié atteint.
  MaxIterationReached =4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de la boucle en temps.
 *
 La boucle en temps est composée de trois parties, exécutées dans l'ordre
 suivant:
 \arg l'initialisation (Init)
 \arg la boucle de calcul (ComputeLoop)
 \arg la terminaison (Exit)

 L'initialisation et la terminaison ne sont appelés qu'une seule fois.
 Par contre, la boucle de calcul est exécutée tant que personne ne demande
 l'arrêt explicite par la méthode stopComputeLoop()
 */
class ITimeLoopMng
{
 public:

  virtual ~ITimeLoopMng() {} //!< Libère les ressources.

 public:

  virtual void build() =0;

  //virtual void initialize() =0;

 public:

  //!< Retourne le gestionnaire du sous-domaine
  virtual ISubDomain* subDomain() const =0;
  
  //! Exécute les points d'entrée de terminaison
  virtual void execExitEntryPoints() =0;

  //! Exécute les points d'entrée de construction
  virtual void execBuildEntryPoints() =0;

  /*! \brief Exécute les points d'entrée d'initialisation.
   * \param is_continue est vrai si on est en reprise */
  virtual void execInitEntryPoints(bool is_continue) =0;

  //! Exécute les points d'entrée après rééquilibrage
  //virtual void execLoadBalanceEntryPoints() =0;

  //! Exécute les points d'entrée après rééquilibrage
  virtual void execOnMeshChangedEntryPoints() =0;

  //! Exécute les points d'entrée après raffinement
  virtual void execOnMeshRefinementEntryPoints() =0;

  /*!
   * \brief Indique que la boucle de calcul doit s'interrompre.
   *
   * Si \a is_final_time est vrai, cela indique que le temps final est atteint.
   * Si \a has_error est vrai, cela indique que le calcul est arrété suite à une
   * erreur. Dans ce cas, le code de retour de l'application sera différent de 0.
   */
  virtual void stopComputeLoop(bool is_final_time,bool has_error=false) =0;

  //! Retourne \a true si le temps final est atteint.
  virtual bool finalTimeReached() const =0;

  //! Retourne le temps CPU utilisé en secondes.
  virtual Real cpuTimeUsed() const =0;

  //! Retourne la liste des points d'entrée de type 'ComputeLoop' de la boucle en temps.
  virtual EntryPointCollection loopEntryPoints() =0;
      
  //! Liste de tous les points d'entrée pour la boucle en temps actuelle.
  virtual EntryPointCollection usedTimeLoopEntryPoints() =0;

  /*!
   * Exécute le point d'entrée suivant.
   *
   * Retourne dans \a is_last \e true si le point d'entrée qui vient d'être
   * exécuté est le dernier de l'itération.
   */
  virtual void doExecNextEntryPoint(bool& is_last) =0;

  //! Retourne le prochain point d'entrée à exécuter ou 0 s'il n'y en a pas
  virtual IEntryPoint* nextEntryPoint() =0;

  //! Retourne le point d'entrée en cours d'exécution ou 0 s'il n'y en a pas
  virtual IEntryPoint* currentEntryPoint() =0;

  /*!
   * \brief Lance l'exécution d'une itération de la boucle de calcul.
   *
   * \retval 0 si le code doit continuer.
   * \retval >0 si le calcul s'arrête normalement.
   * \retval <0 si le calcul s'arrête suite à une erreur.
   */
  virtual int doOneIteration() =0;

  /*!
   * \brief Exécute la boucle de calcul.
   *
   * La boucle de calcul est exécutée jusqu'à l'appel à la méthode
   * stopComputeLoop() ou que le nombre de boucles effectuées est égal
   * à \a max_loop si \a max_loop est différent de 0.
   * \retval 1 si le code s'arrête normalement suite au temps final atteint
   * \retval 2 si le code s'arrête normalement suite à \a max_loop atteint
   * \retval <0 si le calcul s'arrête suite à une erreur.
   */
  virtual int doComputeLoop(Integer max_loop=0) =0;

  //@{
  //! Enregistrement et choix de la boucle en temps.
  /*!
   * \brief Enregistre une boucle en temps.
   * Enregistre la boucle en temps \a time_loop.
   *
   * Si une boucle en temps de même nom que \a time_loop est déjà référencée,
   * la nouvelle remplace l'ancienne.
   */
  virtual void registerTimeLoop(ITimeLoop* time_loop) =0;

  /*! \brief Positionne la boucle en temps à exécuter.
   * Sélectionne la boucle en temps de nom \a name comme celle qui sera
   * exécutée. Cette méthode effectue les opérations suivantes:
   * <ul>
   * <li>à partir du nom \a name, recherche la boucle en temps à utiliser.
   * Cette boucle en temps doit avoir été référencée par l'appel à
   * registerTimeLoop()</li>
   * <li>pour chaque nom de point d'entrée de la boucle en temps,
   * recherche le point d'entrée (IEntryPoint) correspondant enregistré dans
   * l'architecture</li>
   * <li>contruit la liste des points d'entrée à appeler lors de
   * l'initialisation, dans la boucle de calcul et lors de la terminaison
   * en prenant en compte les points d'entrée qui sont chargés automatiquement.</li>
   * <li>détermine la liste des modules utilisés en considérant qu'un module
   * est utilisé si et seulement si l'un de ses points d'entrée est utilisé</li>
   * </ul>
   *
   * L'opération est un échec et provoque une erreur fatal dans l'un
   * des cas suivants:
   * \arg cette méthode a déjà été appelée,
   * \arg aucune boucle en temps de nom \a name n'est enregistrée,
   * \arg un des noms des points d'entrée de la liste ne correspondant à
   * aucun point d'entrée référencé.
   *
   * Si \a name est nulle, la boucle en temps utilisée est la boucle par
   * défaut qui ne contient aucune point d'entrée explicite. Elle contient
   * uniquement les points d'entrée automatiquement enregistrés.
   *
   * \retval true en cas d'erreur,
   * \retval false sinon.
   */
  virtual void setUsedTimeLoop(const String& name) =0;
  //@}

  //! Retourne la boucle en temps utilisée
  virtual ITimeLoop* usedTimeLoop() const =0;

  virtual void setBackwardMng(IBackwardMng* backward_mng) = 0;

  virtual IBackwardMng * getBackwardMng() const = 0;

  /*!
   * \brief Effectue un retour arrière.
   *
   * Cette méthode positionne juste un marqueur. Le retour arrière a
   * effectivement lieu lorsque le point d'entrée actuellement en cours
   * d'exécution se termine.
   *
   * Après retour-arrière, les points d'entrée de retour-arrière sont
   * appelés.
   *
   * \warning Lors d'une exécution parallèle, cette méthode doit être
   * appelée par tous les sous-domaines.
   */
  virtual void goBackward() =0;

  /*! \brief Vrai si on est actuellement dans un retour-arrière.
   *
   * Un retour arrière est actif tant que le temps physique est inférieur
   * au temps physique atteint avant le déclechement du retout-arrière.
   */
  virtual bool isDoingBackward() =0;

  /*!
   * \brief Programme un repartitionnement du maillage avec l'outil
   * de partition \a mesh_partitioner.
   *
   * Cette méthode positionne juste un marqueur. Le repartitionnement a
   * effectivement lieu lorsque le dernier point d'entrée de la boucle
   * de calcul est terminé (fin d'une itération).
   *
   * Après partitionnement, les points d'entrée de changement de maillage sont
   * appelés.
   *
   * \warning Lors d'une exécution parallèle, cette méthode doit être
   * appelée par tous les sous-domaines.
   */
  virtual void registerActionMeshPartition(IMeshPartitionerBase* mesh_partitioner) =0;

  /*!
   * \brief Positionne la période entre deux sauvegarde pour le retour arrière.
   * Si cette valeur est nulle, le retour arrière est désactivé.
   */
  virtual void setBackwardSavePeriod(Integer n) =0;

  /*!
   * \brief Positionne l'état du mode de vérification
   */
  virtual void setVerificationActive(bool is_active) =0;

  /*!
   * \brief Effectue une vérification.
   *
   * Cette opération est collective.
   *
   * Cette opération permet d'effectuer manuellement une opération de
   * vérification, dont le nom est \a name. Ce nom \a name doit être
   * unique pour une itération donnée.
   */
  virtual void doVerification(const String& name) =0;

  /*!
   * \brief Retourne dans \a names la liste des noms des boucles en temps.
   */
  virtual void timeLoopsName(StringCollection& names) const =0;

  //! Retourne dans \a time_loops la liste des boucles en temps.
  virtual void timeLoops(TimeLoopCollection& time_loops) const =0;

  //! Crée une boucle en temps de nom \a name.
  virtual ITimeLoop* createTimeLoop(const String& name) =0;

  //! Nombre de boucles de calcul (ComputeLoop) effectuées.
  virtual Integer nbLoop() const =0;

  /*!
   * \brief Observable sur l'instance.
   *
   * Le type de l'observable est donné par \a type
   */
  virtual IObservable* observable(eTimeLoopEventType type) =0;

  //! Positionne la raison pour laquelle on arrête le code
  virtual void setStopReason(eTimeLoopStopReason reason) =0;

  /*!
   * \brief Raison pour laquelle on arrête le code.
   *
   * Si la valeur est eTimeLoopStopReason::NoStop, alors le code
   * n'est pas en arrêt.
   */
  virtual eTimeLoopStopReason stopReason() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

