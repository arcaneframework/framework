// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProfilingService.h                                         (C) 2000-2023 */
/*                                                                           */
/* Interface d'un service de profiling.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IPROFILINGSERVICE_H
#define ARCANE_UTILS_IPROFILINGSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimerMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service de profiling.
 *
 * Il faut appeler initialize() avant d'utiliser l'instance. On peut
 * ensuite appeler startProfiling()/stopProfiling() pour démarrer et
 * arrêter le profiling.
 *
 * Lorsque le profiling est arrêté, on peut appeler printInfos() pour
 * afficher les informations de profiling. La méthode reset() permet de
 * remettre à zéro les informations de profiling.
 */
class ARCANE_UTILS_EXPORT IProfilingService
{
 public:

  virtual ~IProfilingService() = default;

 public:

  /*!
   * \brief Initialise le service de profiling.
   *
   * Cette méthode ne peut être appelée qu'une seule fois.
   */
  virtual void initialize() = 0;

  //! Indique si initialize() a déjà été appelé
  virtual bool isInitialized() const { return false; }

  //! Démarre un profiling
  virtual void startProfiling() = 0;

  virtual void switchEvent() = 0;

  //! Stoppe le profiling
  virtual void stopProfiling() = 0;

  /*!
   * \brief Affiche les infos de profiling.
   *
   * Le profiling doit être arrêté.
   * Si \a dump_file est vrai, des sorties fichiers contenant les infos
   * sont générées ce qui peut prendre du temps.
   */
  virtual void printInfos(bool dump_file = false) = 0;

  virtual void getInfos(Int64Array&) = 0;

  //! Ecrit les infos de profiling dans l'écrivain \a writer.
  virtual void dumpJSON(JSONWriter& writer) = 0;

  /*!
   * \brief Remet à zéro les compteurs.
   *
   * Le profiling doit être arrêté pour cela.
   */
  virtual void reset() = 0;

  //! Timer utilisant les fonctionnalités de ce service si elles existent. Peut être nul.
  virtual ITimerMng* timerMng() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant de démarrer et arrêter automatiquement un service.
 */
class ARCANE_UTILS_EXPORT ProfilingSentry
{
 public:

  explicit ProfilingSentry(IProfilingService* s)
  : m_service(s)
  {
    if (m_service)
      m_service->startProfiling();
  }
  ~ProfilingSentry()
  {
    if (m_service)
      m_service->stopProfiling();
  }

 public:

  IProfilingService* service() { return m_service; }

 private:

  IProfilingService* m_service;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant de démarrer et arrêter automatiquement un service.
 *
 * Le service est initialisé si nécessaire.
 */
class ARCANE_UTILS_EXPORT ProfilingSentryWithInitialize
{
 public:

  /*!
   * \brief Construit une instance associée au service \a s.
   *
   * Si \a s est \a null, alors l'instance ne fait rien.
   */
  explicit ProfilingSentryWithInitialize(IProfilingService* s)
  : m_service(s)
  {
    if (m_service) {
      if (!m_service->isInitialized())
        m_service->initialize();
      m_service->startProfiling();
    }
  }

  ~ProfilingSentryWithInitialize()
  {
    if (m_service) {
      m_service->stopProfiling();
      if (m_print_at_end)
        m_service->printInfos(false);
    }
  }

 public:

  IProfilingService* service() { return m_service; }
  //! Indique si on imprime les résultats à la fin du profiling
  void setPrintAtEnd(bool v) { m_print_at_end = v; }

 private:

  IProfilingService* m_service = nullptr;
  bool m_print_at_end = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
