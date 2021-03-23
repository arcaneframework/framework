// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SignalException.h                                           (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'un signal survient.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIGNALEXCEPTION_H
#define ARCANE_UTILS_SIGNALEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception lorsqu'un signal survient.
 */
class ARCANE_UTILS_EXPORT SignalException
: public Exception
{
 public:
  enum eSignalType
  {
    ST_Unknown,
    ST_FloatingException,
    ST_SegmentationFault,
    ST_BusError,
    ST_Alarm
  };
 public:
	
  SignalException(const String& where,eSignalType st,int signal_number);
  SignalException(const String& where,const StackTrace& stack_trace,
                  eSignalType st,int signal_number);
  SignalException(const SignalException& ex);
  ~SignalException() ARCANE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;
  eSignalType signalType() const;
  int signalNumber() const;

 private:

	String m_message;
  eSignalType m_signal_type;
  int m_signal_number;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

