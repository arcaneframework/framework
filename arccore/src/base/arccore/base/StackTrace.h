// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StackTrace.h                                                (C) 2000-2025 */
/*                                                                           */
/* Information about a call stack.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STACKTRACE_H
#define ARCCORE_BASE_STACKTRACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Stores the addresses corresponding to a call stack.
 * This class is internal and should not be used outside of Arccore.
 * \todo add support for windows.
 */
class StackFrame
{
 public:

  explicit StackFrame(intptr_t v)
  : m_address(v)
  {}
  StackFrame()
  : m_address(0)
  {}

 public:

  intptr_t address() const { return m_address; }

 private:

  intptr_t m_address;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Stores a fixed maximum size list of StackFrame.
 */
class FixedStackFrameArray
{
 public:

  static const int MAX_FRAME = 32;
  FixedStackFrameArray()
  : m_nb_frame(0)
  {}

 public:

  ConstArrayView<StackFrame> view() const
  {
    return ConstArrayView<StackFrame>(m_nb_frame, m_addresses);
  }

  /*!
   * \brief Adds a frame to the list of frames. If nbFrame() is greater
   * than or equal to MAX_FRAME, no operation is performed.
   */
  void addFrame(const StackFrame& frame)
  {
    if (m_nb_frame < MAX_FRAME) {
      m_addresses[m_nb_frame] = frame;
      ++m_nb_frame;
    }
  }
  Integer nbFrame() const { return m_nb_frame; }

 private:

  //! List of call stack addresses. Stores up to 32 calls.
  StackFrame m_addresses[MAX_FRAME];
  Integer m_nb_frame;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Information about function call stacks.
 */
class ARCCORE_BASE_EXPORT StackTrace
{
 public:

  StackTrace() {}
  StackTrace(const FixedStackFrameArray& stack_frames)
  : m_stack_frames(stack_frames)
  {}
  StackTrace(const String& msg)
  : m_stack_trace_string(msg)
  {}
  StackTrace(const FixedStackFrameArray& stack_frames, const String& msg)
  : m_stack_frames(stack_frames)
  , m_stack_trace_string(msg)
  {}

 public:

  //! String indicating the call stack.
  const String& toString() const { return m_stack_trace_string; }

  //! Call stack in the form of addresses.
  ConstArrayView<StackFrame> stackFrames() const { return m_stack_frames.view(); }

 private:

  FixedStackFrameArray m_stack_frames;
  String m_stack_trace_string;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Stream insertion operator for a StackTrace
ARCCORE_BASE_EXPORT std::ostream& operator<<(std::ostream& o, const StackTrace&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
