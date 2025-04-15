// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMngInternal.h                                    (C) 2000-2025 */
/*                                                                           */
/* Classe interne gérant un historique de valeurs.                           */
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
 * \brief Classe de base d'un historique de valeurs.
 *
 * Un historique contient un ensemble de valeurs pour certaines itérations.
 * Il est caractérisé par un nom.
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

  virtual ~TimeHistoryValue() = default; //!< Libére les ressources

 public:

  /*!
   * \brief Méthode permettant de convertir les variables d'anciennes sauvegardes
   * vers le nouveau format.
   *
   * \param vm Le VariableMng.
   * \param default_mesh Le maillage par défaut.
   */
  virtual void fromOldToNewVariables(IVariableMng* vm, IMesh* default_mesh) = 0;

  //! Imprime les valeurs de l'historique avec l'écrivain \a writer
  virtual void dumpValues(ITraceMng* msg,
                          ITimeHistoryCurveWriter2* writer,
                          const TimeHistoryCurveWriterInfo& infos) const = 0;

  /*!
   * \brief Méthode permettant de récupérer les itérations et les valeurs d'un historique de valeur.
   *
   * \param iterations [OUT] Les itérations où ont été récupérer chaque valeur.
   * \param values [OUT] Les valeurs récupérées.
   * \param infos Les informations nécessaire à la récupération de l'historique.
   */
  virtual void arrayToWrite(UniqueArray<Int32>& iterations,
                            UniqueArray<Real>& values,
                            const TimeHistoryCurveWriterInfo& infos) const = 0;

  /*!
   * \brief Méthode permettant d'appliquer une transformation sur les valeurs
   * de l'historique de valeur.
   *
   * \param msg Le traceMng où écrire les messages.
   * \param v Le transformer.
   */
  virtual void applyTransformation(ITraceMng* msg,
                                   ITimeHistoryTransformer* v) = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de valeurs enregistrées.
   *
   * \return Le nombre de valeurs enregistrées.
   */
  virtual Integer size() const = 0;

  /*!
   * \brief Méthode permettant de retirer toutes les valeurs après une certaine itération.
   *
   * \param last_iteration La dernière itération voulu.
   */
  virtual void removeAfterIteration(Integer last_iteration) = 0;

  //! Nom de l'historique
  const String& name() const { return m_thpi.timeHistoryAddValueArg().name(); }

  //! Type de données de l'historique
  eDataType dataType() const { return m_data_type; }

  //! Index de l'historique dans la liste
  Integer index() const { return m_index; }

  Integer subSize() const { return m_sub_size; }

  /*!
   * \brief Méthode permettant de récupérer le MeshHandle enregistré.
   *
   * Attention, pour les historiques globaux, ce MeshHandle est null !
   *
   * \return Le MeshHandle.
   */
  const MeshHandle& meshHandle() const { return m_thpi.meshHandle(); }

  /*!
   * \brief Méthode permettant de savoir si c'est un historique global ou local à un sous-domaine.
   *
   * \sa localProcId()
   * \return true si c'est un historique local.
   */
  bool isLocal() const { return m_thpi.timeHistoryAddValueArg().isLocal(); }

  /*!
   * \brief Méthode permettant de récupérer l'id du sous-domaine à qui appartient cet historique.
   *
   * \return L'id du sous-domaine.
   */
  Integer localSubDomainId() const { return m_thpi.timeHistoryAddValueArg().localSubDomainId(); }

 private:

  eDataType m_data_type; //!< Type de la donnée
  Integer m_index; //!< Index de l'historique dans la liste
  Integer m_sub_size;
  TimeHistoryAddValueArgInternal m_thpi;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Historique de valeurs du type \a T.
 *
 * Actuellement, on ne support que trois types de valeurs: Real, \a Int32
 * et \a Int64.
 *
 * Un historique est composé d'un tableau de couples (x,y) avec \a x le
 * numéro de l'itération et \a y la valeur de l'historique.
 *
 * Les historiques doivent être rangées par ordre croissant d'itération.
 */
template <typename DataType>
class TimeHistoryValueT
: public TimeHistoryValue
{
  /*
   * ATTENTION CE QUI EST DECRIT DANS CE COMMENTAIRE N'EST PAS ENCORE OPERATIONNEL
   * Lorsqu'il y a beaucoup de courbes et que le nombre d'itérations
   * devient important, le stockage consomme de la mémoire. Pour éviter
   * cela, il est possible de compresser le tableau des itérations.
   * Si c'est le cas et que les itérations sont consécutives, on
   * conserve uniquement la première et la dernière. Dans ce cas,
   * m_iterations à 3 valeurs: [0] = COMPRESSED_TAG, [1] = première
   * et [2] = dernière.
   */
 public:

  typedef VariableRefArrayT<DataType> ValueList;
  typedef VariableRefArrayT<Int32> IterationList;
  static const Integer COMPRESSED_TAG = -15;

 public:

  const int VAR_BUILD_FLAGS = IVariable::PNoRestore | IVariable::PExecutionDepend | IVariable::PNoReplicaSync;

 public:

  /*!
   * \brief Constructeur permettant de construire un historique de valeur non lié à un maillage.
   *
   * \param vm Le variableMng pour créer les variables.
   * \param thpi Les arguments pour créer l'historique.
   * \param index L'index des variables globales.
   * \param nb_element Le nombre de valeurs par itération.
   * \param shrink S'il y a compression.
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
   * \brief Constructeur permettant de construire un historique de valeur lié à un maillage.
   *
   * \param thpi Les arguments pour créer l'historique.
   * \param index L'index des variables globales.
   * \param nb_element Le nombre de valeurs par itération.
   * \param shrink S'il y a compression.
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
   * \brief Méthode permettant d'ajouter des valeurs à une itération.
   *
   * \param values Les valeurs à ajouter.
   * \param iteration L'itération liée aux valeurs.
   */
  void addValue(ConstArrayView<DataType> values, Integer iteration)
  {
    Integer nb_iteration = m_iterations.size();
    Integer nb_value = m_values.size();
    Integer sub_size = values.size();
    if (nb_iteration != 0)
      if (m_iterations[nb_iteration - 1] == iteration) {
        // Remplace la valeur
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

  // Ecriture d'une courbe pour les écrivains version 2.
  void dumpValues(ITraceMng* msg,
                  ITimeHistoryCurveWriter2* writer,
                  const TimeHistoryCurveWriterInfo& infos) const override
  {
    ARCANE_UNUSED(msg);

    // Pour l'instant, on ne fait rien
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
    // Pour vérifier qu'on ne sauve pas plus d'itérations qu'il y en
    // a actuellement (ce qui peut arriver en cas de retour arrière).
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
    // TODO AH : Avec la nouvelle API, cette variable devrait pouvoir être toujours true (grâce à m_need_comm). À garder pour l'IFPEN.
    m_enable_non_io_master_curves = !platform::getEnvironmentVariable("ARCANE_ENABLE_NON_IO_MASTER_CURVES").null();

    bool enable_all_replicats_write = !platform::getEnvironmentVariable("ARCANE_ENABLE_ALL_REPLICATS_WRITE_CURVES").null();

    // Seul le sous-domaine maître des IO rend actif les time history.
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
 * \brief Méthode permettant d'ajouter des valeurs à un historique de valeurs.
 *
 * \tparam DataType Les valeurs à ajouter.
 * \param thpi Les paramètres pour ajouter les valeurs.
 * \param values Les valeurs à ajouter.
 */
  template <class DataType>
  void _addHistoryValue(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<DataType> values);

  /*!
   * \brief Destructeur.
   */
  void _destroyAll();

  /*!
   * \brief Méthode permettant de sortir toutes les courbes avec tous les writers.
   */
  void _dumpCurvesAllWriters();

  /*!
   * \brief Méthode permettant de sortir un fichier XML avec le nom de
   * chaque courbe sortie en format GNUPLOT.
   */
  void _dumpSummaryOfCurvesLegacy();

  /*!
 * \brief Méthode permettant de sortir un fichier JSON avec le nom de
   * chaque courbe sortie en format GNUPLOT ainsi que plusieurs autres
   * informations.
   */
  void _dumpSummaryOfCurves();

  /*!
   * \brief Méthode permettant de convertir l'ancien format vers le nouveau.
   *
   * \param default_mesh Le maillage par défaut sur lequel les anciennes valeurs sont liées.
   */
  void _fromLegacyFormat(IMesh* default_mesh);

  /*!
   * \brief Méthode permettant de sauver les propriétés des metadatas.
   */
  void _saveProperties();

  /*!
   * \brief Méthode permettant de retirer un écrivain.
   * \param writer La reference de l'écrivain.
   */
  void _removeCurveWriter(const Ref<ITimeHistoryCurveWriter2>& writer);

 private:

  IVariableMng* m_variable_mng;
  ITraceMng* m_trace_mng;
  IParallelMng* m_parallel_mng;
  CommonVariables m_common_variables;
  Directory m_directory;

  bool m_is_master_io; //!< True si je suis le gestionnaire des IO
  bool m_is_master_io_of_sd; //!< True si je suis le gestionnaire des IO pour mon sous-domaine.
  bool m_enable_non_io_master_curves; //!< Indique si l'ecriture  de courbes par des procs non io_master est possible
  bool m_is_active; //!< Indique si le service est actif.
  bool m_is_shrink_active; //!< Indique si la compression de l'historique est active
  bool m_is_dump_active; //!< Indique si les dump sont actifs
  bool m_io_master_write_only; //!< Indique si les writers doivent être appelé par tous les processus.
  bool m_need_comm; //!< Indique si au moins une courbe est non local (donc nécessite des communications).

  String m_output_path;
  ObserverPool m_observer_pool;
  HistoryList m_history_list; //!< Liste des historiques
  VariableScalarString m_th_meta_data; //!< Infos des historiques
  VariableArrayReal m_th_global_time; //!< Tableau des instants de temps
  RealUniqueArray m_global_times; //!< Liste des temps globaux
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
