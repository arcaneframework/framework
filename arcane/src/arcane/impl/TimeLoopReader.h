// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoopReader.h                                            (C) 2000-2006 */
/*                                                                           */
/* Loading a time loop.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MAIN_TIMELOOPREADER_H
#define ARCANE_MAIN_TIMELOOPREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeLoop;
class IApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Time loop loading functor.
 *
 * Based on the dataset and general options, it reads the name of the
 * time loop and indicates it to the manager #m_mng.
 */
class ARCANE_IMPL_EXPORT TimeLoopReader
: public TraceAccessor
{
 public:

  //! Creates an instance associated with the manager \a sm
  TimeLoopReader(IApplication* sm);
  ~TimeLoopReader(); //!< Frees resources

 public:

  //! Performs the reading of available time loops.
  void readTimeLoops();

  //! Registers the list of time loops in the manager \a sd
  void registerTimeLoops(ISubDomain* sd);

  //! Positions the used time loop in the manager \a sd
  void setUsedTimeLoop(ISubDomain* sd);

  //! name of the time loop to execute.
  const String& timeLoopName() const { return m_time_loop_name; }

  //! List of read time loops
  TimeLoopCollection timeLoops() const { return m_time_loops; }

 private:

  IApplication* m_application; //!< Supervisor.
  TimeLoopList m_time_loops;
  String m_time_loop_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
