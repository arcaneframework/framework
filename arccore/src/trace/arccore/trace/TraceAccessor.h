// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceAccessor.h                                             (C) 2000-2025 */
/*                                                                           */
/* Access to traces.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TRACEACCESSOR_H
#define ARCCORE_TRACE_TRACEACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceMessage.h"
#include "arccore/base/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITraceMng;
class TraceMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Trace access class.
 * \ingroup Core
 */
class ARCCORE_TRACE_EXPORT TraceAccessor
{
 public:
  // NOTE: The 'default' versions of the constructors,
  // destructors and copy operators must not be used unless 'ITraceMng' is known
  // because of the use of 'Ref' and we do not want to include
  // 'ITraceMng.h' if it is not used directly.

  //! Constructs an accessor via the trace manager \a m.
  explicit TraceAccessor(ITraceMng* m);
  //! Copy constructor
  TraceAccessor(const TraceAccessor& rhs);
  //! Copy assignment operator
  TraceAccessor& operator=(const TraceAccessor& rhs);
  virtual ~TraceAccessor(); //!< Frees resources

 public:

  //! Trace manager
  ITraceMng* traceMng() const;

  //! Flow for an information message
  TraceMessage info() const;

  /*! \brief Flow for a parallel information message.
   *
   * Unlike info(), all processors write this
   * message to standard output.
   */
  TraceMessage pinfo() const;

  //! Flow for an information message of a given category
  TraceMessage info(char category) const;

  //! Flow for a parallel information message of a given category
  TraceMessage pinfo(char category) const;

  /*!
   * \brief Flow for an information message.
   *
   * If \a v is \a false, the message will not be displayed.
   */
  TraceMessage info(bool v) const;

  //! Flow for a warning message
  TraceMessage warning() const;

  /*! Flow for a parallel warning message
   *
   * Unlike warning(), only the master processor writes this message.
   */
  TraceMessage pwarning() const;

  //! Flow for an error message
  TraceMessage error() const;

  /*! Flow for a parallel error message
   *
   * Unlike error(), only the master processor writes this message.
   */
  TraceMessage perror() const;

  //! Flow for a log message
  TraceMessage log() const;

  //! Flow for a log message
  TraceMessage plog() const;

  //! Flow for a log message preceded by the date
  TraceMessage logdate() const;

  //! Flow for a fatal error message
  TraceMessage fatal() const;

  //! Flow for a parallel fatal error message
  TraceMessage pfatal() const;

#ifdef ARCCORE_DEBUG
  //! Flow for a debug message
  TraceMessageDbg debug(Trace::eDebugLevel =Trace::Medium) const;
#else
  //! Flow for a debug message
  TraceMessageDbg debug(Trace::eDebugLevel =Trace::Medium) const
  { return TraceMessageDbg(); }
#endif

  //! Debug level of the configuration file
  Trace::eDebugLevel configDbgLevel() const;

  //! Flow for an information message of a given level
  TraceMessage info(Int32 verbose_level) const;

  //! Flow for an information message with the local information level of this instance.
  TraceMessage linfo() const
  {
    return info(m_local_verbose_level);
  }

  //! Flow for an information message with the local information level of this instance.
  TraceMessage linfo(Int32 relative_level) const
  {
    return info(m_local_verbose_level+relative_level);
  }

  void fatalMessage [[noreturn]] (const StandaloneTraceMessage& o) const;

 protected:
  
  void _setLocalVerboseLevel(Int32 v) { m_local_verbose_level = v; }
  Int32 _localVerboseLevel() const { return m_local_verbose_level; }

 private:

  Ref<ITraceMng> m_trace;
  Int32 m_local_verbose_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
