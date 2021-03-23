// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProfilingService.h                                         (C) 2000-2020 */
/*                                                                           */
/* Interface d'un service de profiling.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPROFILINGSERVICE_H
#define ARCANE_IPROFILINGSERVICE_H
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
 * \note En cours de développement
 */
class ARCANE_UTILS_EXPORT IProfilingService
{
 public:

  virtual ~IProfilingService() = default;

 public:

  //! Initialise le service de profiling.
  virtual void initialize() =0;

  //! Démarre un profiling
  virtual void startProfiling() =0;

  virtual void switchEvent() =0;

  //! Stoppe le profiling
  virtual void stopProfiling() =0;

  /*!
   * \brief Affiche les infos de profiling.
   * Le profiling doit être arrété.
   * Si \a dump_file est vrai, des sorties fichiers contenant les infos
   * sont généréées ce qui peut prendre du temps.
   */
  virtual void printInfos(bool dump_file=false) =0;
  virtual void getInfos(Int64Array &)=0;

  //! Ecrit les infos de profiling dans l'écrivain \a writer.
  virtual void dumpJSON(JSONWriter& writer) =0;

  /*!
   * \brief Remet à zéro les compteurs.
   * Le profiling doit être arrété pour cela.
   */
  virtual void reset() =0;

  //! Timer utilisant les fonctionalité de ce service si elles existent. Peut être nul.
  virtual ITimerMng* timerMng() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT ProfilingSentry
{
 public:
  explicit ProfilingSentry(IProfilingService* s) : m_service(s)
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
