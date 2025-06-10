// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleMaster.h                                              (C) 2000-2025 */
/*                                                                           */
/* Module maître.                                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MODULEMASTER_H
#define ARCANE_CORE_MODULEMASTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"

#include "arcane/core/IModuleMaster.h"
#include "arcane/core/AbstractModule.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Module principal.
 *
 * Ce module est toujours chargé en premier pour que ces points d'entrée encadrent tous ceux des autres modules.
 * Il contient les variables globales du cas, comme le nom de fichier ou le numéro de l'itération.
 */
class ARCANE_CORE_EXPORT ModuleMaster
: public AbstractModule
, public CommonVariables
, public IModuleMaster
{
 public:

  //! Constructeur
  explicit ModuleMaster(const ModuleBuildInfo&);

  //! Destructeur
  ~ModuleMaster() override;

 public:

  //! Version du module
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

  //! Accès aux options du module
  CaseOptionsMain* caseoptions() override { return m_case_options_main; }

  //! Conversion en \a IModule
  IModule* toModule() override { return this; }

  //! Accès aux variables 'communes' partagées entre tout service et module
  CommonVariables* commonVariables() override { return this; }

  void addTimeLoopService(ITimeLoopService* tls) override;

  //! Sort les courbes classiques (CPUTime, ElapsedTime, ...)
  void dumpStandardCurves() override;

 public:

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
  CaseOptionsMain* m_case_options_main = nullptr;

  //! Nombre de boucles de calcul effectuées
  Integer m_nb_loop = 0;

  //! Valeur du temps CPU à la dernière itération
  Real m_old_cpu_time = 0.0;

  //! Valeur du temps horloge à la dernière itération
  Real m_old_elapsed_time = 0.0;

  //! Liste des services de boucle en temps
  UniqueArray<ITimeLoopService*> m_timeloop_services;

  //! Indique si on est dans la première itération de l'exécution
  bool m_is_first_loop = true;

  Real m_thm_mem_used = 0.0;
  Real m_thm_diff_cpu = 0.0;
  Real m_thm_global_cpu_time = 0.0;
  Real m_thm_diff_elapsed = 0.0;
  Real m_thm_global_elapsed_time = 0.0;
  Real m_thm_global_time = 0.0;

  bool m_has_thm_dump_at_iteration = false;

 private:

  void _dumpTimeInfo();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

