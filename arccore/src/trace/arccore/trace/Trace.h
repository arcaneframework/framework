// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Trace.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Traces.                                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TRACE_H
#define ARCCORE_TRACE_TRACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Namespace containing types related to traces.
 */
namespace Trace
{
  //! Default verbosity level.
  static const int DEFAULT_VERBOSITY_LEVEL = 3;

  //! Unspecified verbosity level.
  static const int UNSPECIFIED_VERBOSITY_LEVEL = -2;

  //! Stream on which messages are sent.
  enum eMessageType
  {
    Normal=0,
    Info=1,
    Warning=2,
    Error=3,
    Log=4,
    Fatal=5,
    ParallelFatal=6,
    Debug=7,
    Null=8
  };

  //! Trace debug level
  enum eDebugLevel
  {
    None = 0, //! No debug information
    Lowest = 1, //!< Lowest level
    Low = 2, //!< Low level
    Medium = 3, //!< Medium level (default)
    High = 4, //!< High level
    Highest = 5 //!< Highest level
  };

  //! Display parameters.
  enum ePrintFlags
  {
    PF_Default = 0,
    PF_NoClassName = 1 << 1, //!< Display or not the message class.
    PF_ElapsedTime = 1 << 2, //!< Display of elapsed time
  };

  /*!
   * \brief Formatting the stream by length.
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Width
  {
   public:
    //! Formats the stream to a length of \a v characters.
    Width(Integer v) : m_width(v) {}
   public:
    Integer m_width; //!< Formatting length.
  };
  /*!
   * \brief Formatting real numbers with a given precision.
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Precision
  {
   public:
    //! Prints the value \a value with \a n significant figures.
    Precision(Integer n,Real value)
    : m_precision(n), m_value(value), m_scientific(false) {}
    /*! Prints the value \a value with \n n significant figures in scientific mode if
     * \a scientific is true
     */
    Precision(Integer n,Real value,bool scientific)
    : m_precision(n), m_value(value), m_scientific(scientific) {}
   public:
    Integer m_precision; //!< Number of significant figures.
    Real m_value; //!< Value to output
    bool m_scientific; //! True if output is in scientific mode
  };

  /*!
   * \brief Sets a color for the message
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Color
  {
   public:
    // List of colors.
    // NOTE: any change here must be reported in TraceMng.cc
    enum { Black = 0,
           DarkRed, DarkGreen, DarkYellow, DarkBlue, DarkMagenta, DarkCyan, DarkGrey,
           Red, Green, Yellow, Blue, Magenta, Cyan, Grey };

    static const int LAST_COLOR = Grey;
   public:
    Color(int color) : m_color(color){}
   public:
    static Color darkRed() { return Color(DarkRed); }
    static Color darkGreen() { return Color(DarkGreen); }
    static Color darkYellow() { return Color(DarkYellow); }
    static Color darkBlue() { return Color(DarkBlue); }
    static Color darkMagenta() { return Color(DarkMagenta); }
    static Color darkCyan() { return Color(DarkCyan); }
    static Color darkGrey() { return Color(DarkGrey); }

    static Color red() { return Color(Red); }
    static Color green() { return Color(Green); }
    static Color yellow() { return Color(Yellow); }
    static Color blue() { return Color(Blue); }
    static Color magenta() { return Color(Magenta); }
    static Color cyan() { return Color(Cyan); }
    static Color grey() { return Color(Grey); }
   public:
    int m_color;
  };


  /*!
   * \brief Sets a message class.
   *
   * Sets the message class \a name upon construction
   * of the instance and restores the previous one upon destruction.
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Setter
  {
   public:
    //! Sets the class \a name for the manager \a msg
    Setter(ITraceMng* msg,const String& name);
    //! Releases the instance and restores the previous message class in \a m_msg
    ~Setter();
   public:
   private:
    ITraceMng* m_msg; //!< Message manager.
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
