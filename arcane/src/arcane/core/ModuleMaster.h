// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleMaster.h                                              (C) 2000-2012 */
/*                                                                           */
/* Module maître.                                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MODULEMASTER_H
#define ARCANE_MODULEMASTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"

#include "arcane/IModuleMaster.h"
#include "arcane/AbstractModule.h"
#include "arcane/VariableTypedef.h"
#include "arcane/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseOptionsMain;
class ITimeLoopService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Module principal.
 *
 * Ce module est toujours chargé en premier pour que ces points d'entrée encadrent tous ceux des autres modules.
 * Il contient les variables globales du cas, comme le nom de fichier ou le numéro de l'itération.
 */
class ARCANE_CORE_EXPORT  ModuleMaster
: public AbstractModule
, public CommonVariables
, public IModuleMaster
{
 public:

  //! Constructeur
  ModuleMaster(const ModuleBuildInfo&);

  //! Destructeur
  virtual ~ModuleMaster();

 public:

  //! Version du module
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,0); }

 public:

  //! Accès aux options du module
  virtual CaseOptionsMain* caseoptions() { return m_case_options_main; }

  //! Conversion en \a IModule
  virtual IModule* toModule() { return this; }

  //! Accès aux variables 'communes' partagées entre tout service et module
  virtual CommonVariables* commonVariables() { return this; }

  virtual void addTimeLoopService(ITimeLoopService* tls);

  //! Sort les courbes classiques (CPUTime, ElapsedTime, ...)
  virtual void dumpStandardCurves();

  //! Point d'entrée auto-chargé en début d'itération de la boucle de calcul
  /*!
    <ul>
    <li>Au cas où le temps courant est strictement supérieur au temps limite, demande l'arrêt du calcul</li>
    <li>Ajoute au temps courant le deltat calculé au pas de temps précédent</li>
    </ul>
  */
  void timeLoopBegin();

  //! Point d'entrée auto-chargé en fin d'itération de la boucle de calcul
  /*!
   <ul>
   <li>Incrémente le compteur d'itération</li>
   </ul>
  */
  void timeLoopEnd();

  //! Point d'entrée auto-chargé en début d'initialisation
  void masterInit();

  //! Point d'entrée auto-chargé en début d'initialisation d'un nouveau cas
  /*! N'est pas appelé en cas d'initialisation sur une reprise */
  void masterStartInit();

  //! Point d'entrée auto-chargé en début de reprise d'un nouveau cas
  void masterContinueInit();

 protected:

  //! Incrémentation du pas de temps surchargeable
  // nb: IFPEN a une notion d'évènement. On peut connaître le prochain
  // temps et pas de temps. Si on applique l'incrémentation par défaut,
  // on a des erreurs d'arrondi...
  virtual void timeIncrementation();

  //! Affichage surchargeable des informations du pas de temps
  // nb: IFPEN souhaite des affichages paramétrables par application
  virtual void timeStepInformation();

  void _masterBeginLoop();
  void _masterEndLoop();
  void _masterStartInit();
  void _masterContinueInit();
  void _masterLoopExit();
  void _masterMeshChanged();
  void _masterRestore();
  
 protected:

  //! Instance des options du module
  CaseOptionsMain* m_case_options_main;

  //! Nombre de boucles de calcul effectuées
  Integer m_nb_loop;

  //! Valeur du temps CPU à la derniere itération
  Real m_old_cpu_time;

  //! Valeur du temps horloge à la derniere itération
  Real m_old_elapsed_time;

  //! Liste des serviecs de boucle en temps
  UniqueArray<ITimeLoopService*> m_timeloop_services;

  //! Indique si on est dans la première itération de l'exécution
  bool m_is_first_loop;

  Real m_thm_mem_used;
  Real m_thm_diff_cpu;
  Real m_thm_global_cpu_time;
  Real m_thm_diff_elapsed;
  Real m_thm_global_elapsed_time;
  Real m_thm_global_time;
  bool m_has_thm_dump_at_iteration;

 private:

  void _dumpTimeInfo();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

