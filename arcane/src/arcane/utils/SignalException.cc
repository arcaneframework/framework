// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SignalException.cc                                          (C) 2000-2016 */
/*                                                                           */
/* Exception lorsqu'un signal survient.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/SignalException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SignalException::
SignalException(const String& where,eSignalType st,int signal_number)
: Exception("Signal",where)
, m_signal_type(st)
, m_signal_number(signal_number)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SignalException::
SignalException(const String& where,const StackTrace& stack_trace,
                eSignalType st,int signal_number)
: Exception("Signal",where,stack_trace)
, m_signal_type(st)
, m_signal_number(signal_number)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SignalException::
SignalException(const SignalException& ex)
: Exception(ex)
, m_message(ex.m_message)
, m_signal_type(ex.m_signal_type)
, m_signal_number(ex.m_signal_number)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SignalException::
explain(std::ostream& m) const
{
  if (!m_message.null())
		m << "Message: " << m_message << '\n';
	
	m << "A fatal signal has occurred: ";
	switch(m_signal_type){
  case ST_FloatingException:
    m << "Floating Exception";
    break;
  case ST_SegmentationFault:
    m << "Segmentation Violation";
    break;
  case ST_BusError:
    m << "Bus Error";
    break; 
  case ST_Alarm:
    m << "Sigalarm";
    break;
  case ST_Unknown:
    m << "Unknown";
    break;
  }
  m << " (signal number is: " << m_signal_number << ")\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SignalException::eSignalType SignalException::
signalType() const
{
  return m_signal_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int SignalException::
signalNumber() const
{
  return m_signal_number;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

