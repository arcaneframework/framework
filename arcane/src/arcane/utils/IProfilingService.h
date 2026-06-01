// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProfilingService.h                                         (C) 2000-2023 */
/*                                                                           */
/* Interface of a profiling service.                                         */
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
 * \brief Interface of a profiling service.
 *
 * initialize() must be called before using the instance. You can
 * then call startProfiling()/stopProfiling() to start and
 * stop profiling.
 *
 * When profiling is stopped, you can call printInfos() to
 * display the profiling information. The reset() method allows you to
 * reset the profiling information.
 */
class ARCANE_UTILS_EXPORT IProfilingService
{
 public:

  virtual ~IProfilingService() = default;

 public:

  /*!
   * \brief Initializes the profiling service.
   *
   * This method can only be called once.
   */
  virtual void initialize() = 0;

  //! Indicates if initialize() has already been called
  virtual bool isInitialized() const { return false; }

  //! Starts profiling
  virtual void startProfiling() = 0;

  virtual void switchEvent() = 0;

  //! Stops profiling
  virtual void stopProfiling() = 0;

  /*!
   * \brief Displays profiling information.
   *
   * Profiling must be stopped.
   * If \a dump_file is true, file outputs containing the information
   * are generated, which may take time.
   */
  virtual void printInfos(bool dump_file = false) = 0;

  virtual void getInfos(Int64Array&) = 0;

  //! Writes the profiling information to the writer \a writer.
  virtual void dumpJSON(JSONWriter& writer) = 0;

  /*!
   * \brief Resets the counters.
   *
   * Profiling must be stopped for this.
   */
  virtual void reset() = 0;

  //! Timer using the features of this service if they exist. Can be null.
  virtual ITimerMng* timerMng() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing automatic start and stop of a service.
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
 * \brief Class allowing automatic start and stop of a service.
 *
 * The service is initialized if necessary.
 */
class ARCANE_UTILS_EXPORT ProfilingSentryWithInitialize
{
 public:

  /*!
   * \brief Constructs an instance associated with the service \a s.
   *
   * If \a s is \a null, the instance does nothing.
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
  //! Indicates if results are printed at the end of profiling
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
