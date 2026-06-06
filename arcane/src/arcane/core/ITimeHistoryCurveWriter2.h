// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryCurveWriter2.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface for a history curve writer (Version 2).                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYCURVEWRITER2_H
#define ARCANE_CORE_ITIMEHISTORYCURVEWRITER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

/*
 * \brief Indicates whether private member access is allowed.
 * By default (March 2016), access is kept for compatibility reasons,
 * but it will need to be removed.
 */
#define ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS 1
#ifdef SWIG
#undef ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for writing a curve.
 */
class TimeHistoryCurveInfo
{
 public:

  TimeHistoryCurveInfo(const String& aname, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size)
  : m_name(aname)
  , m_support()
  , m_has_support(false)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(-1)
  {}

  TimeHistoryCurveInfo(const String& aname, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size, Integer sub_domain)
  : m_name(aname)
  , m_support()
  , m_has_support(false)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(sub_domain)
  {}

  TimeHistoryCurveInfo(const String& aname, const String& asupport, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size)
  : m_name(aname)
  , m_support(asupport)
  , m_has_support(true)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(-1)
  {}

  TimeHistoryCurveInfo(const String& aname, const String& asupport, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size, Integer sub_domain)
  : m_name(aname)
  , m_support(asupport)
  , m_has_support(true)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(sub_domain)
  {}

 public:

  //! Curve name
  const String& name() const { return m_name; }
  const String& support() const { return m_support; }
  bool hasSupport() const { return m_has_support; }
  //! List of iterations
  Int32ConstArrayView iterations() const { return m_iterations; }
  //! List of curve values
  RealConstArrayView values() const { return m_values; }
  //! Number of values per time step
  Integer subSize() const { return m_sub_size; }
  // TODO not a great name
  Integer subDomain() const { return m_sub_domain; }

#if ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS
 public:

#else
 private:
#endif
  String m_name;
  String m_support;
  bool m_has_support = false;
  Int32ConstArrayView m_iterations;
  RealConstArrayView m_values;
  Integer m_sub_size = 0;
  Integer m_sub_domain = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about writing curves.
 */
class TimeHistoryCurveWriterInfo
{
 public:

  TimeHistoryCurveWriterInfo(const String& apath, RealConstArrayView atimes)
  : m_path(apath)
  , m_times(atimes)
  {}

 public:

  /*!
   * \brief Path to write the data (unless specifically overridden
   * by the service via ITimeHistoryCurveWriter2::setOutputPath())
   */
  String path() const { return m_path; }
  //! List of times
  RealConstArrayView times() const { return m_times; }

#if ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS
 public:

#else
 private:
#endif
  String m_path;
  RealConstArrayView m_times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface for a curve writer.
 *
 * When writing curves, the instance will be called as follows:
 * \code
 * ITimeHistoryCurveWriter2* instance = ...;
 * instance->beginWrite();
 * for( const TimeHistoryCurveInfo& curveinfo : all_curves )
 *   instance->writeCurve(curveinfo);
 * instance->endWrite()
 * \endcode
 */
class ARCANE_CORE_EXPORT ITimeHistoryCurveWriter2
{
 public:

  //! Release resources
  virtual ~ITimeHistoryCurveWriter2() = default;

 public:

  virtual void build() = 0;

  /*!
   * \brief Notify the start of writing.
   */
  virtual void beginWrite(const TimeHistoryCurveWriterInfo& infos) = 0;

  /*!
   * \brief Notify the end of writing.
   */
  virtual void endWrite() = 0;

  /*!
   * \brief Write a curve.
   *
   * Curve info is provided by \a infos
   * Values are in the array \a values. \a times and \a iterations
   * contain respectively the time and the iteration number for
   * each value.
   * \a path contains the directory where the curves will be written
   */
  virtual void writeCurve(const TimeHistoryCurveInfo& infos) = 0;

  //! Writer name
  virtual String name() const = 0;

  /*!
   * \brief Base directory where curves will be written.
   *
   * If null, the directory specified during beginWrite()
   * will be used.
   */
  virtual void setOutputPath(const String& path) = 0;

  //! Base directory where curves will be written.
  virtual String outputPath() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
