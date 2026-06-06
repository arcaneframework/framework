// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface of the class managing a history of values.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYMNG_H
#define ARCANE_CORE_ITIMEHISTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Class containing the arguments for the user methods 'addValue'.
 */
class TimeHistoryAddValueArg
{
 public:

  /*!
   * \brief Constructor with three parameters.
   *
   * \param name The name of the curve.
   * \param end_time Should the value be written at our iteration or at our iteration-1?
   * \param subdomain_id The ID of the subdomain that must save the value (-1 for global).
   */
  TimeHistoryAddValueArg(const String& name, bool end_time, Integer subdomain_id)
  : m_name(name)
  , m_end_time(end_time)
  , m_subdomain_id(subdomain_id)
  {}

  /*!
   * \brief Constructor with two parameters.
   *
   * The value will be saved globally, and not on a specific subdomain.
   *
   * \param name The name of the curve.
   * \param end_time Should the value be written at our iteration or at our iteration-1?
   */
  TimeHistoryAddValueArg(const String& name, bool end_time)
  : TimeHistoryAddValueArg(name, end_time, NULL_SUB_DOMAIN_ID)
  {}

  /*!
   * \brief Constructor with one parameter.
   *
   * The value will be saved globally, and not on a specific subdomain.
   * The value will be saved at our iteration.
   *
   * \param name The name of the curve.
   */
  explicit TimeHistoryAddValueArg(const String& name)
  : TimeHistoryAddValueArg(name, true)
  {}

 public:

  const String& name() const { return m_name; }
  bool endTime() const { return m_end_time; }
  bool isLocal() const { return m_subdomain_id != NULL_SUB_DOMAIN_ID; }
  Integer localSubDomainId() const { return m_subdomain_id; }

 private:

  String m_name;
  bool m_end_time;
  Integer m_subdomain_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryCurveWriter;
class ITimeHistoryCurveWriter2;
class ITimeHistoryTransformer;
class ITimeHistoryMngInternal;
class ITimeHistoryAdder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class managing a history of values.
 *
 The history manager manages the history of a set of values over time.
 
 The history is based on iterations (VariablesCommon::globalIteration()).
 For each iteration, it is possible to save a value using the addValue()
 methods. It is not mandatory to have a value for every iteration. When
 several addValue() calls are made for the same history at the same
 iteration, only the last value is taken into account.

 Each history is associated with a name, which is the name of the file
 where the list of values will be saved.

 Only the instance associated with the subdomain where
 parallelMng()->isMasterIO() is true saves the values. For others, calls
 to addValue() have no effect.

 Values are only saved if active() is true. It is possible to change the
 activation status by calling isActive().

 In debug mode, all histories are saved at every time step.
 In normal execution, this set is saved every \a n iterations, \a n being
 given by the dataset option <module-main/time-history-iteration-step>.
 In any case, an output is performed at the end of the execution.

 The format of these files depends on the implementation.

 \since 0.4.38
 */
class ITimeHistoryMng
{
 public:

  virtual ~ITimeHistoryMng() = default; //!< Frees resources

 public:

  // TODO Deprecated
  /*! \brief Adds the value \a value to the history \a name.
   *
   * \deprecated This method is deprecated and is replaced by using the
   * GlobalTimeHistoryAdder object.
   *
   * The value is that at the end time of the iteration if \a end_time is true,
   * at the beginning otherwise.
   * the boolean is_local indicates whether the curve is specific to the
   * process or not, in order to be able to write curves even by non io_master
   * procs when the ARCANE_ENABLE_NON_IO_MASTER_CURVES variable is set.
   */
  virtual void addValue(const String& name, Real value, bool end_time = true, bool is_local = false) = 0;
  /*! \brief Adds the value \a value to the history \a name.
   *
   * \deprecated This method is deprecated and is replaced by using the
   * GlobalTimeHistoryAdder object.
   *
   * The value is that at the end time of the iteration if \a end_time is true,
   * at the beginning otherwise.
   * the boolean is_local indicates whether the curve is specific to the
   * process or not, in order to be able to write curves even by non io_master
   * procs when the ARCANE_ENABLE_NON_IO_MASTER_CURVES variable is set.
   */
  virtual void addValue(const String& name, Int32 value, bool end_time = true, bool is_local = false) = 0;
  /*! Adds the value \a value to the history \a name.
   *
   * \deprecated This method is deprecated and is replaced by using the
   * GlobalTimeHistoryAdder object.
   *
   * The value is that at the end time of the iteration if \a end_time is true,
   * at the beginning otherwise.
   * the boolean is_local indicates whether the curve is specific to the process
   * or not, in order to be able to write curves even by non io_master procs
   * when the ARCANE_ENABLE_NON_IO_MASTER_CURVES variable is set.
   */
  virtual void addValue(const String& name, Int64 value, bool end_time = true, bool is_local = false) = 0;
  /*! \brief Adds the value \a value to the history \a name.
   *
   * \deprecated This method is deprecated and is replaced by using the
   * GlobalTimeHistoryAdder object.
   *
   * The number of elements of \a value must be constant over time.
   * The value is that at the end time of the iteration if \a end_time is true,
   * at the beginning otherwise.
   * the boolean is_local indicates whether the curve is specific to the process
   * or not, in order to be able to write curves even by non io_master procs when
   * the ARCANE_ENABLE_NON_IO_MASTER_CURVES variable is set.
   */
  virtual void addValue(const String& name, RealConstArrayView value, bool end_time = true, bool is_local = false) = 0;
  /*! \brief Adds the value \a value to the history \a name.
   *
   * \deprecated This method is deprecated and is replaced by using the
   * GlobalTimeHistoryAdder object.
   *
   * The number of elements of \a value must be constant over time.
   * The value is that at the end time of the iteration if \a end_time is true,
   * at the beginning otherwise.
   * the boolean is_local indicates whether the curve is specific to the process
   * or not, in order to be able to write curves even by non io_master procs when
   * the ARCANE_ENABLE_NON_IO_MASTER_CURVES variable is set.
   */
  virtual void addValue(const String& name, Int32ConstArrayView value, bool end_time = true, bool is_local = false) = 0;
  /*! Adds the value \a value to the history \a name.
   *
   * \deprecated This method is deprecated and is replaced by using the
   * GlobalTimeHistoryAdder object.
   *
   * The number of elements of \a value must be constant over time.
   * The value is that at the end time of the iteration if \a end_time is true,
   * at the beginning otherwise.
   * the boolean is_local indicates whether the curve is specific to the process
   * or not, in order to be able to write curves even by non io_master procs when
   * the ARCANE_ENABLE_NON_IO_MASTER_CURVES variable is set.
   */
  virtual void addValue(const String& name, Int64ConstArrayView value, bool end_time = true, bool is_local = false) = 0;

 public:

  virtual void timeHistoryBegin() = 0;
  virtual void timeHistoryEnd() = 0;
  virtual void timeHistoryInit() = 0;
  virtual void timeHistoryStartInit() = 0;
  virtual void timeHistoryContinueInit() = 0;
  virtual void timeHistoryRestore() = 0;

 public:

  //! Adds a writer
  virtual ARCANE_DEPRECATED void addCurveWriter(ITimeHistoryCurveWriter* writer)
  {
    ARCANE_UNUSED(writer);
    ARCANE_FATAL("No longer supported. Use 'ITimeHistoryCurveWriter2' interface");
  }

  //! Removes a writer
  virtual ARCANE_DEPRECATED void removeCurveWriter(ITimeHistoryCurveWriter* writer)
  {
    ARCANE_UNUSED(writer);
    ARCANE_FATAL("No longer supported. Use 'ITimeHistoryCurveWriter2' interface");
  }

  //! Adds a writer
  virtual void addCurveWriter(ITimeHistoryCurveWriter2* writer) = 0;

  //! Removes a writer
  virtual void removeCurveWriter(ITimeHistoryCurveWriter2* writer) = 0;

  //! Removes the writer with name \a name
  virtual void removeCurveWriter(const String& name) = 0;

 public:

  /*!
   * \internal
   * \brief Saves the history.
   *
   * This consists of calling dumpCurves() for each registered writer.
   */
  virtual void dumpHistory(bool is_verbose) = 0;

  /*!
   * \brief Uses the writer \a writer to output all curves.
   *
   * The output path is the current directory.
   */
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer) = 0;

  /*!
   * \brief Indicates the activation status.
   *
   * The addValue() functions are only considered if the instance
   * is active. Otherwise, calls to addValue() are
   * ignored.
   */
  virtual bool active() const = 0;

  /*!
   * \brief Sets the activation status.
   * \sa active().
   */
  virtual void setActive(bool is_active) = 0;

  /*!
   * \brief Applies the transformation \a v to all curves.
   */
  virtual void applyTransformation(ITimeHistoryTransformer* v) = 0;

  /*!
   * \brief Indicates the output activation status.
   *
   * The dumpHistory() function is inactive
   * if isDumpActive() is false.
   */
  virtual bool isDumpActive() const = 0;

  /*!
   * \brief Sets the output activation status.
   */
  virtual void setDumpActive(bool is_active) = 0;

  /*!
   * \brief Returns a boolean indicating if the history is compressed
   */
  virtual bool isShrinkActive() const = 0;

  /*!
   * \brief Sets the boolean indicating if the history is compressed
   */
  virtual void setShrinkActive(bool is_active) = 0;

 public:

  //! Internal Arcane API
  virtual ITimeHistoryMngInternal* _internalApi() { ARCANE_FATAL("Invalid usage"); };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
