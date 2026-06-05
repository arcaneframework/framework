// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMngInternal.h                                    (C) 2000-2025 */
/*                                                                           */
/* Internal class managing a history of values.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_TIMEHISTORYMNGINTERNAL_H
#define ARCANE_IMPL_INTERNAL_TIMEHISTORYMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/core/IIOMng.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/Directory.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ITimeHistoryCurveWriter2.h"
#include "arcane/core/ITimeHistoryTransformer.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IParallelReplication.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/core/internal/ITimeHistoryMngInternal.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*!
 * \brief Base class for a value history.
 *
 * A history contains a set of values for certain iterations.
 * It is characterized by a name.
 */
class TimeHistoryValue
{
 public:

  TimeHistoryValue(const TimeHistoryAddValueArgInternal& thpi, eDataType dt, Integer index, Integer sub_size)
  : m_data_type(dt)
  , m_index(index)
  , m_sub_size(sub_size)
  , m_thpi(thpi)
  {}

  virtual ~TimeHistoryValue() = default; //!< Frees resources

 public:

  /*!
   * \brief Method allowing the conversion of variables from old saves
   * to the new format.
   *
   * \param vm The VariableMng.
   * \param default_mesh The default mesh.
   */
  virtual void fromOldToNewVariables(IVariableMng* vm, IMesh* default_mesh) = 0;

  //! Prints the history values using the writer \a writer
  virtual void dumpValues(ITraceMng* msg,
                          ITimeHistoryCurveWriter2* writer,
                          const TimeHistoryCurveWriterInfo& infos) const = 0;

  /*!
   * \brief Method allowing the retrieval of iterations and values from a value history.
   *
   * \param iterations [OUT] The iterations where each value was retrieved.
   * \param values [OUT] The retrieved values.
   * \param infos The information necessary to retrieve the history.
   */
  virtual void arrayToWrite(UniqueArray<Int32>& iterations,
                            UniqueArray<Real>& values,
                            const TimeHistoryCurveWriterInfo& infos) const = 0;

  /*!
   * \brief Method allowing the application of a transformation on the values
   * of the value history.
   *
   * \param msg The traceMng where messages should be written.
   * \param v The transformer.
   */
  virtual void applyTransformation(ITraceMng* msg,
                                   ITimeHistoryTransformer* v) = 0;

  /*!
   * \brief Method allowing the retrieval of the number of recorded values.
   *
   * \return The number of recorded values.
   */
  virtual Integer size() const = 0;

  /*!
   * \brief Method allowing the removal of all values after a certain iteration.
   *
   * \param last_iteration The desired last iteration.
   */
  virtual void removeAfterIteration(Integer last_iteration) = 0;

  //! History name
  const String& name() const { return m_thpi.timeHistoryAddValueArg().name(); }

  //! History data type
  eDataType dataType() const { return m_data_type; }

  //! History index in the list
  Integer index() const { return m_index; }

  Integer subSize() const { return m_sub_size; }

  /*!
   * \brief Method allowing the retrieval of the registered MeshHandle.
   *
   * Note: For global histories, this MeshHandle is null!
   *
   * \return The MeshHandle.
   */
  const MeshHandle& meshHandle() const { return m_thpi.meshHandle(); }

  /*!
   * \brief Method allowing determination if it is a global history or local to a subdomain.
   *
   * \sa localProcId()
   * \return true if it is a local history.
   */
  bool isLocal() const { return m_thpi.timeHistoryAddValueArg().isLocal(); }

  /*!
   * \brief Method allowing the retrieval of the subdomain ID to which this history belongs.
   *
   * \return The subdomain ID.
   */
  Integer localSubDomainId() const { return m_thpi.timeHistoryAddValueArg().localSubDomainId(); }

 private:

  eDataType m_data_type; //!< Data type
  Integer m_index; //!< History index in the list
  Integer m_sub_size;
  TimeHistoryAddValueArgInternal m_thpi;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Value history of type \a T.
 *
 * Currently, only three types of values are supported: Real, \a Int32
 * and \a Int64.
 *
 * A history is composed of an array of pairs (x,y) where \a x is the
 * iteration number and \a y is the history value.
 *
 * Histories must be sorted in ascending order of iteration.
 */
template <typename DataType>
class TimeHistoryValueT
: public TimeHistoryValue
{
  /*
   * WARNING: WHAT IS DESCRIBED IN THIS COMMENT IS NOT YET OPERATIONAL
   * When there are many curves and the number of iterations
   * becomes significant, storage consumes memory. To avoid
   * this, it is possible to compress the array of iterations.
   * If this is the case and the iterations are consecutive, only the first
   * and the last are kept. In this case,
   * m_iterations has 3 values: [0] = COMPRESSED_TAG, [1] = first
   * and [2] = last.
   */
 public:

  typedef VariableRefArrayT<DataType> ValueList;
  typedef VariableRefArrayT<Int32> IterationList;
  static const Integer COMPRESSED_TAG = -15;

 public:

  const int VAR_BUILD_FLAGS = IVariable::PNoRestore | IVariable::PExecutionDepend | IVariable::PNoReplicaSync;

 public:

  /*!
   * \brief Constructor for building a value history not linked to a mesh.
   *
   * \param vm The variableMng to create the variables.
   * \param thpi The arguments to create the history.
   * \param index The index of the global variables.
   * \param nb_element The number of values per iteration.
   * \param shrink If there is compression.
   */
  TimeHistoryValueT(IVariableMng* vm, const TimeHistoryAddValueArgInternal& thpi, Integer index, Integer nb_element, bool shrink)
  : TimeHistoryValue(thpi, DataTypeTraitsT<DataType>::type(), index, nb_element)
  , m_values(VariableBuildInfo(vm, String("TimeHistoryMngValues") + index, VAR_BUILD_FLAGS))
  , m_iterations(VariableBuildInfo(vm, String("TimeHistoryMngIterations") + index, VAR_BUILD_FLAGS))
  , m_use_compression(false)
  , m_shrink_history(shrink)
  {
  }

  /*!
   * \brief Constructor for building a value history linked to a mesh.
   *
   * \param thpi The arguments to create the history.
   * \param index The index of the global variables.
   * \param nb_element The number of values per iteration.
   * \param shrink If there is compression.
   */
  TimeHistoryValueT(const TimeHistoryAddValueArgInternal& thpi, Integer index, Integer nb_element, bool shrink)
  : TimeHistoryValue(thpi, DataTypeTraitsT<DataType>::type(), index, nb_element)
  , m_values(VariableBuildInfo(thpi.meshHandle(), String("TimeHistoryMngValues") + index, VAR_BUILD_FLAGS))
  , m_iterations(VariableBuildInfo(thpi.meshHandle(), String("TimeHistoryMngIterations") + index, VAR_BUILD_FLAGS))
  , m_use_compression(false)
  , m_shrink_history(shrink)
  {
  }

 public:

  void fromOldToNewVariables(IVariableMng* vm, IMesh* default_mesh) override
  {
    IVariable* ptr_old_values = vm->findMeshVariable(default_mesh, String("TimeHistory_Values_") + index());
    IVariable* ptr_old_iterations = vm->findMeshVariable(default_mesh, String("TimeHistory_Iterations_") + index());
    if (ptr_old_values == nullptr || ptr_old_iterations == nullptr)
      ARCANE_FATAL("Unknown old variable");

    ValueList old_values(ptr_old_values);
    IterationList old_iterations(ptr_old_iterations);

    m_values.resize(old_values.size());
    m_values.copy(old_values);

    m_iterations.resize(old_iterations.size());
    m_iterations.copy(old_iterations);

    old_values.resize(0);
    old_iterations.resize(0);
  }

  Integer size() const override
  {
    return m_iterations.size();
  }

  /*!
   * \brief Method allowing the addition of values to an iteration.
   *
   * \param values The values to add.
   * \param iteration The iteration linked to the values.
   */
  void addValue(ConstArrayView<DataType> values, Integer iteration)
  {
    Integer nb_iteration = m_iterations.size();
    Integer nb_value = m_values.size();
    Integer sub_size = values.size();
    if (nb_iteration != 0)
      if (m_iterations[nb_iteration - 1] == iteration) {
        // Replace the value
        for (Integer i = 0; i < sub_size; ++i)
          m_values[nb_value - sub_size + i] = values[i];
        return;
      }
    Integer add_nb_iter = math::max(128, nb_iteration / 20);
    Integer add_nb_value = math::max(1024, nb_value / 20);
    m_iterations.resizeWithReserve(nb_iteration + 1, add_nb_iter);
    m_values.resizeWithReserve(nb_value + sub_size, add_nb_value);
    m_iterations[nb_iteration] = iteration;
    for (Integer i = 0; i < sub_size; ++i)
      m_values[nb_value + i] = values[i];
  }

  void removeAfterIteration(Integer last_iteration) override
  {
    Integer size = m_iterations.size();
    Integer last_elem = size;
    for (Integer i = 0; i < size; ++i)
      if (m_iterations[i] >= last_iteration) {
        last_elem = i;
        break;
      }
    if (last_elem != size) {
      m_iterations.resize(last_elem);
      m_values.resize(last_elem * subSize());
    }
  }

  // Writing a curve for writers version 2.
  void dumpValues(ITraceMng* msg,
                  ITimeHistoryCurveWriter2* writer,
                  const TimeHistoryCurveWriterInfo& infos) const override
  {
    ARCANE_UNUSED(msg);

    // For now, we do nothing
    if (m_shrink_history)
      return;

    UniqueArray<Real> values_to_write;
    UniqueArray<Int32> iterations_to_write;

    arrayToWrite(iterations_to_write, values_to_write, infos);

    Integer sd = localSubDomainId();
    if (!meshHandle().isNull()) {
      TimeHistoryCurveInfo curve_info(name(), meshHandle().meshName(), iterations_to_write, values_to_write, subSize(), sd);
      writer->writeCurve(curve_info);
    }
    else {
      TimeHistoryCurveInfo curve_info(name(), iterations_to_write, values_to_write, subSize(), sd);
      writer->writeCurve(curve_info);
    }
  }

  void applyTransformation(ITraceMng* msg, ITimeHistoryTransformer* v) override
  {
    ITimeHistoryTransformer::CommonInfo ci;
    ci.name = name();
    SharedArray<Int32> iterations(m_iterations.asArray());
    ci.iterations = iterations;
    Integer sub_size = subSize();
    ci.sub_size = subSize();

    SharedArray<DataType> values(m_values.asArray());

    v->transform(ci, values);

    Integer nb_iteration = iterations.size();
    Integer nb_value = values.size();
    if (nb_iteration * sub_size != nb_value) {
      msg->warning() << "Bad size after history transformation";
      return;
    }

    m_iterations.resize(nb_iteration);
    for (Integer i = 0; i < nb_iteration; ++i)
      m_iterations[i] = iterations[i];

    m_values.resize(nb_value);
    for (Integer i = 0; i < nb_value; ++i)
      m_values[i] = values[i];
  }

  void arrayToWrite(UniqueArray<Int32>& iterations, UniqueArray<Real>& values, const TimeHistoryCurveWriterInfo& infos) const override
  {
    // To check that we do not save more iterations than there are
    // currently (which can happen in case of rollback).
    Integer max_iter = infos.times().size();
    Integer nb_iteration = m_iterations.size();
    Integer sub_size = subSize();
    iterations.clear();
    iterations.reserve(nb_iteration);
    values.clear();
    values.reserve(nb_iteration * sub_size);
    for (Integer i = 0, is = nb_iteration; i < is; ++i) {
      Integer iter = m_iterations[i];
      if (iter < max_iter) {
        for (Integer z = 0; z < sub_size; ++z) {
          values.add(Convert::toReal(m_values[(i * sub_size) + z]));
        }
        iterations.add(iter);
      }
    }
  }

  const ValueList& values() const { return m_values; }
  const IterationList& iterations() const { return m_iterations; }

 private:

  ValueList m_values;
  IterationList m_iterations;
  bool m_use_compression;
  bool m_shrink_history;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TimeHistoryMngInternal
: public ITimeHistoryMngInternal
{
 public:

  explicit TimeHistoryMngInternal(IVariableMng* vm, const Ref<Properties>& properties)
  : m_variable_mng(vm)
  , m_trace_mng(m_variable_mng->traceMng())
  , m_parallel_mng(m_variable_mng->parallelMng())
  , m_common_variables(m_variable_mng)
  , m_is_active(true)
  , m_is_shrink_active(false)
  , m_is_dump_active(true)
  , m_io_master_write_only(false)
  , m_need_comm(false)
  , m_th_meta_data(VariableBuildInfo(m_variable_mng, "TimeHistoryMngMetaData"))
  , m_th_global_time(VariableBuildInfo(m_variable_mng, "TimeHistoryMngGlobalTime"))
  , m_properties(properties)
  , m_version(2)
  {
    // TODO AH: With the new API, this variable should always be true
    // (thanks to m_need_comm). Keep for IFPEN.
    m_enable_non_io_master_curves = !platform::getEnvironmentVariable("ARCANE_ENABLE_NON_IO_MASTER_CURVES").null();

    bool enable_all_replicats_write = !platform::getEnvironmentVariable("ARCANE_ENABLE_ALL_REPLICATS_WRITE_CURVES").null();

    // Only the IO master subdomain activates time history.
    IParallelReplication* pr = m_parallel_mng->replication();
    if (pr->hasReplication()) {
      m_is_master_io = (pr->isMasterRank() && m_parallel_mng->isMasterIO());
      m_is_master_io_of_sd = (enable_all_replicats_write || pr->isMasterRank());
    }
    else {
      m_is_master_io = m_parallel_mng->isMasterIO();
      m_is_master_io_of_sd = true;
    }
  }

  ~TimeHistoryMngInternal() override
  {
    arcaneCallFunctionAndCatchException([&]() { _destroyAll(); });

    m_curve_writers2.clear();
    m_history_list.clear();
  }

  typedef std::map<String, TimeHistoryValue*> HistoryList;
  typedef std::set<Ref<ITimeHistoryCurveWriter2>> CurveWriter2List;
  typedef HistoryList::value_type HistoryValueType;

 public:

  void addValue(const TimeHistoryAddValueArgInternal& thpi, Real value) override
  {
    RealConstArrayView values(1, &value);
    _addHistoryValue(thpi, values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64 value) override
  {
    Int64ConstArrayView values(1, &value);
    _addHistoryValue(thpi, values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32 value) override
  {
    Int32ConstArrayView values(1, &value);
    _addHistoryValue(thpi, values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi, RealConstArrayView values) override
  {
    _addHistoryValue(thpi, values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32ConstArrayView values) override
  {
    _addHistoryValue(thpi, values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64ConstArrayView values) override
  {
    _addHistoryValue(thpi, values);
  }

 public:

  void addNowInGlobalTime() override;
  void updateGlobalTimeCurve() override;
  void resizeArrayAfterRestore() override;
  void dumpCurves(ITimeHistoryCurveWriter2* writer) override;
  void dumpHistory() override;
  void updateMetaData() override;
  void readVariables(IMeshMng* mesh_mng, IMesh* default_mesh) override;

  void addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer) override;
  void removeCurveWriter(const String& name) override;
  void applyTransformation(ITimeHistoryTransformer* v) override;

  void addObservers(IPropertyMng* prop_mng) override;
  void editOutputPath(const Directory& directory) override;
  void iterationsAndValues(const TimeHistoryAddValueArgInternal& thpi, UniqueArray<Int32>& iterations, UniqueArray<Real>& values) override;

 public:

  bool isShrinkActive() const override { return m_is_shrink_active; }
  void setShrinkActive(bool is_active) override { m_is_shrink_active = is_active; }
  bool active() const override { return m_is_active; }
  void setActive(bool is_active) override { m_is_active = is_active; }
  bool isDumpActive() const override { return m_is_dump_active; }
  void setDumpActive(bool is_active) override { m_is_dump_active = is_active; }
  bool isMasterIO() override { return m_is_master_io; }
  bool isMasterIOOfSubDomain() override { return m_is_master_io_of_sd; }
  bool isNonIOMasterCurvesEnabled() override { return m_enable_non_io_master_curves; }
  bool isIOMasterWriteOnly() override { return m_io_master_write_only; }
  void setIOMasterWriteOnly(bool is_active) override { m_io_master_write_only = is_active; }

 private:

  /*!
 * \brief Method allowing values to be added to a value history.
 *
 * \tparam DataType The values to be added.
 * \param thpi The parameters for adding values.
 * \param values The values to be added.
 */
  template <class DataType>
  void _addHistoryValue(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<DataType> values);

  /*!
   * \brief Destructor.
   */
  void _destroyAll();

  /*!
   * \brief Method allowing all curves to be dumped with all writers.
   */
  void _dumpCurvesAllWriters();

  /*!
   * \brief Method allowing an XML file to be dumped with the name of
   * each curve output in GNUPLOT format.
   */
  void _dumpSummaryOfCurvesLegacy();

  /*!
 * \brief Method allowing a JSON file to be dumped with the name of
   * each curve output in GNUPLOT format as well as several other
   * information.
   */
  void _dumpSummaryOfCurves();

  /*!
   * \brief Method allowing conversion from the old format to the new.
   *
   * \param default_mesh The default mesh on which the old values are linked.
   */
  void _fromLegacyFormat(IMesh* default_mesh);

  /*!
   * \brief Method allowing saving the properties of the metadata.
   */
  void _saveProperties();

  /*!
   * \brief Method allowing removal of a writer.
   * \param writer The reference of the writer.
   */
  void _removeCurveWriter(const Ref<ITimeHistoryCurveWriter2>& writer);

 private:

  IVariableMng* m_variable_mng;
  ITraceMng* m_trace_mng;
  IParallelMng* m_parallel_mng;
  CommonVariables m_common_variables;
  Directory m_directory;

  bool m_is_master_io; //!< True if I am the IO manager
  bool m_is_master_io_of_sd; //!< True if I am the IO manager for my subdomain.
  bool m_enable_non_io_master_curves; //!< Indicates if curve writing by non-io_master procs is possible
  bool m_is_active; //!< Indicates if the service is active.
  bool m_is_shrink_active; //!< Indicates if history compression is active
  bool m_is_dump_active; //!< Indicates if dumps are active
  bool m_io_master_write_only; //!< Indicates if writers must be called by all processes.
  bool m_need_comm; //!< Indicates if at least one curve is non-local (thus requiring communications).

  String m_output_path;
  ObserverPool m_observer_pool;
  HistoryList m_history_list; //!< List of histories
  VariableScalarString m_th_meta_data; //!< History info
  VariableArrayReal m_th_global_time; //!< Array of time instants
  RealUniqueArray m_global_times; //!< List of global times
  CurveWriter2List m_curve_writers2;
  Ref<Properties> m_properties;
  Integer m_version;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
