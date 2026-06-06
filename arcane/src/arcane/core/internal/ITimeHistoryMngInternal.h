// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryMngInternal.h                                   (C) 2000-2025 */
/*                                                                           */
/* Internal class interface managing a history of values.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ITIMEHISTORYMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_ITIMEHISTORYMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/IPropertyMng.h"
#include "arcane/core/Directory.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryTransformer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class extending the arguments when adding a value
 * to a value history.
 */
class ARCANE_CORE_EXPORT TimeHistoryAddValueArgInternal
{
 public:

  explicit TimeHistoryAddValueArgInternal(const TimeHistoryAddValueArg& thp)
  : m_thp(thp)
  , m_mesh_handle()
  {}

  TimeHistoryAddValueArgInternal(const TimeHistoryAddValueArg& thp, const MeshHandle& mesh_handle)
  : m_thp(thp)
  , m_mesh_handle(mesh_handle)
  {}

  TimeHistoryAddValueArgInternal(const String& name, bool end_time, Integer subdomain_id)
  : m_thp(name, end_time, subdomain_id)
  , m_mesh_handle()
  {}

 public:

  const TimeHistoryAddValueArg& timeHistoryAddValueArg() const { return m_thp; }
  const MeshHandle& meshHandle() const { return m_mesh_handle; }

 private:

  TimeHistoryAddValueArg m_thp;
  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for the internal part of a value history manager.
 */
class ARCANE_CORE_EXPORT ITimeHistoryMngInternal
{
 public:

  virtual ~ITimeHistoryMngInternal() = default; //!< Releases resources

 public:

  /*!
   * \brief Method allowing a value to be added to a history.
   *
   * \param thpi History parameters.
   * \param value The value to add.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Real value) = 0;

  /*!
   * \brief Method allowing a value to be added to a history.
   *
   * \param thpi History parameters.
   * \param value The value to add.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32 value) = 0;

  /*!
   * \brief Method allowing a value to be added to a history.
   *
   * \param thpi History parameters.
   * \param value The value to add.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64 value) = 0;

  /*!
   * \brief Method allowing values to be added to a history.
   *
   * \param thpi History parameters.
   * \param value The values to add.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, RealConstArrayView values) = 0;

  /*!
   * \brief Method allowing values to be added to a history.
   *
   * \param thpi History parameters.
   * \param value The values to add.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32ConstArrayView values) = 0;

  /*!
   * \brief Method allowing values to be added to a history.
   *
   * \param thpi History parameters.
   * \param value The values to add.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64ConstArrayView values) = 0;

  /*!
   * \brief Method allowing the current GlobalTime to be added to the GlobalTimes array.
   */
  virtual void addNowInGlobalTime() = 0;

  /*!
   * \brief Method allowing the GlobalTime array to be copied into the global GlobalTime variable.
   */
  virtual void updateGlobalTimeCurve() = 0;

  /*!
   * \brief Method allowing the value arrays to be resized after a restart.
   */
  virtual void resizeArrayAfterRestore() = 0;

  /*!
   * \brief Method allowing curves to be written using the provided writer.
   *
   * \param writer The writer with which the curves must be written.
   * \param master_only If all histories must be transferred to
   *                    the masterIO before copying.
   */
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer) = 0;

  /*!
   * \brief Method allowing all curves to be written using all registered writers.
   */
  virtual void dumpHistory() = 0;

  /*!
   * \brief Method allowing the curve metadata to be updated.
   */
  virtual void updateMetaData() = 0;

  /*!
   * \brief Method allowing previously written curves to be retrieved during a restart.
   *
   * \param mesh_mng A pointer to a meshMng.
   * \param default_mesh A pointer to the default mesh (only necessary for
   *                     retrieving old checkpoints).
   */
  virtual void readVariables(IMeshMng* mesh_mng, IMesh* default_mesh) = 0;

  /*!
   * \brief Method allowing a writer to be added for curve output.
   *
   * \param writer A reference to the writer.
   */
  virtual void addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer) = 0;

  /*!
   * \brief Method allowing a writer to be removed.
   *
   * \param writer The name of the writer.
   */
  virtual void removeCurveWriter(const String& name) = 0;

  /*!
   * \brief Applies the transformation \a v to all curves.
   *
   * \param v The transformation to apply.
   */
  virtual void applyTransformation(ITimeHistoryTransformer* v) = 0;

  /*!
   * \brief Returns a boolean indicating if the history is compressed
   */
  virtual bool isShrinkActive() const = 0;
  /*!
   * \brief Sets the boolean indicating if the history is compressed
   */
  virtual void setShrinkActive(bool is_active) = 0;

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
   * \brief Method allowing to know if our process is the writer.
   * \return True if we are the writer.
   */
  virtual bool isMasterIO() = 0;

  /*!
   * \brief Method allowing to know if our process is the writer for our subdomain.
   * In the case where replication is enabled, only one process among the replicas can
   * write (and only if isNonIOMasterCurvesEnabled() == true).
   *
   * The environment variable ARCANE_ENABLE_ALL_REPLICATS_WRITE_CURVES allows bypassing this
   * protection and allows all processes to write.
   *
   * \return True if we are the writer for our subdomain.
   */
  virtual bool isMasterIOOfSubDomain() = 0;

  /*!
   * \brief Method allowing to know if all processes can have a value history.
   */
  virtual bool isNonIOMasterCurvesEnabled() = 0;

  /*!
   * \brief Method allowing to know if only the master process calls the writers.
   *
   * \return true if yes
   */
  virtual bool isIOMasterWriteOnly() = 0;

  /*!
   * \brief Method allowing to define if only the master process calls the writers.
   *
   * \param is_active True if yes.
   */
  virtual void setIOMasterWriteOnly(bool is_active) = 0;

  /*!
   * \brief Method allowing observers saving the history before a checkpoint to be added.
   *
   * \param prop_mng A pointer to an IPropertyMng.
   */
  virtual void addObservers(IPropertyMng* prop_mng) = 0;

  /*!
   * \brief Method allowing the curve output directory to be changed.
   *
   * Note that the directory will be created if it does not exist.
   *
   * \param directory The new output directory.
   */
  virtual void editOutputPath(const Directory& directory) = 0;

  /*!
   * \brief Method allowing the iterations and values of a history to be outputted.
   *
   * Useful method for debug/test. Caution in domain replication mode: only the masterRank of the subdomains possess the values.
   *
   * \param thpi Information necessary for retrieving the history.
   * \param iterations [OUT] The iterations where each value was retrieved.
   * \param values [OUT] The retrieved values.
   */
  virtual void iterationsAndValues(const TimeHistoryAddValueArgInternal& thpi, UniqueArray<Int32>& iterations, UniqueArray<Real>& values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
