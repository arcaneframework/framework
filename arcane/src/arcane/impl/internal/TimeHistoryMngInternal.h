// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMngInternal.h                                    (C) 2000-2024 */
/*                                                                           */
/* Classe interne gérant un historique de valeurs.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_TIMEHISTORYMNGINTERNAL_H
#define ARCANE_IMPL_INTERNAL_TIMEHISTORYMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/IIOMng.h"
#include "arcane/CommonVariables.h"
#include "arcane/ISubDomain.h"
#include "arcane/Directory.h"
#include "arcane/AbstractModule.h"
#include "arcane/EntryPoint.h"
#include "arcane/ObserverPool.h"
#include "arcane/IVariableMng.h"
#include "arcane/CaseOptionsMain.h"
#include "arcane/IParallelMng.h"
#include "arcane/ITimeHistoryCurveWriter2.h"
#include "arcane/ITimeHistoryTransformer.h"
#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/Properties.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/core/internal/ITimeHistoryMngInternal.h"

#include <map>
#include <set>
#include <variant>

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

 public:

  TimeHistoryValue(const String& name, eDataType dt, Integer index, Integer sub_size)
  : m_name(name)
  , m_data_type(dt)
  , m_index(index)
  , m_sub_size(sub_size)
  , m_mesh_handle()
  {}

  TimeHistoryValue(const String& name, const MeshHandle& mesh_handle, eDataType dt, Integer index, Integer sub_size)
  : m_name(name)
  , m_data_type(dt)
  , m_index(index)
  , m_sub_size(sub_size)
  , m_mesh_handle(mesh_handle)
  {}

  virtual ~TimeHistoryValue() = default; //!< Libére les ressources

 public:

  virtual void fromOldToNewVariables(ISubDomain* sd) = 0;

  //! Imprime les valeurs de l'historique avec l'écrivain \a writer
  virtual void dumpValues(ITraceMng* msg,
                          ITimeHistoryCurveWriter2* writer,
                          const TimeHistoryCurveWriterInfo& infos) const =0;

  virtual void applyTransformation(ITraceMng* msg,
                                   ITimeHistoryTransformer* v) =0;

  //! Retourne le nombre d'éléments dans le tableau.
  virtual Integer size() const =0;

  /*!
   * \brief Supprime les valeurs des historiques dont l'itération
   * est supérieur ou égal à \a last_iteration.
   */
  virtual void removeAfterIteration(Integer last_iteration) =0;

  //! Nom de l'historique
  const String& name() const { return m_name; }

  //! type de données de l'historique
  eDataType dataType() const { return m_data_type; }

  //! index de l'historique dans la liste
  Integer index() const { return m_index; }

  Integer subSize() const { return m_sub_size; }

  const MeshHandle& meshHandle() const { return m_mesh_handle; }

 private:

  String m_name; //!< Nom de l'historique
  eDataType m_data_type; //!< Type de la donnée
  Integer m_index; //!< Index de l'historique dans la liste
  Integer m_sub_size;
  MeshHandle m_mesh_handle;
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
template<typename DataType>
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
  const int VAR_BUILD_FLAGS = IVariable::PNoRestore|IVariable::PExecutionDepend | IVariable::PNoReplicaSync;
 public:

  TimeHistoryValueT(ISubDomain* sd, const String& name, Integer index, Integer nb_element, bool shrink)
  : TimeHistoryValue(name,DataTypeTraitsT<DataType>::type(),index,nb_element)
  , m_values(VariableBuildInfo(sd,String("TimeHistoryMngValues")+index,VAR_BUILD_FLAGS))
  , m_iterations(VariableBuildInfo(sd,String("TimeHistoryMngIterations")+index,VAR_BUILD_FLAGS))
  , m_use_compression(false)
  , m_shrink_history(shrink)
  {
  }

  TimeHistoryValueT(const MeshHandle& mesh_handle, const String& name, Integer index, Integer nb_element, bool shrink)
  : TimeHistoryValue(name, mesh_handle, DataTypeTraitsT<DataType>::type(), index, nb_element)
  , m_values(VariableBuildInfo(mesh_handle, String("TimeHistoryMngValues")+index,VAR_BUILD_FLAGS))
  , m_iterations(VariableBuildInfo(mesh_handle, String("TimeHistoryMngIterations")+index,VAR_BUILD_FLAGS))
  , m_use_compression(false)
  , m_shrink_history(shrink)
  {
  }

 public:

  void fromOldToNewVariables(ISubDomain* sd) override
  {
    IMesh* mesh0 = sd->defaultMesh();
    IVariable* ptr_old_values = sd->variableMng()->findMeshVariable(mesh0, String("TimeHistory_Values_")+index());
    IVariable* ptr_old_iterations = sd->variableMng()->findMeshVariable(mesh0, String("TimeHistory_Iterations_")+index());
    if(ptr_old_values == nullptr || ptr_old_iterations == nullptr)
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

  void addValue(ConstArrayView<DataType> values,Integer iteration)
  {
    Integer nb_iteration = m_iterations.size();
    Integer nb_value = m_values.size();
    Integer sub_size = values.size();
    if (nb_iteration!=0)
      if (m_iterations[nb_iteration-1]==iteration){
        // Remplace la valeur
        for( Integer i=0; i<sub_size; ++i )
          m_values[nb_value-sub_size+i] = values[i];
        return;
      }
    Integer add_nb_iter = math::max(128,nb_iteration/20);
    Integer add_nb_value = math::max(1024,nb_value/20);
    m_iterations.resizeWithReserve(nb_iteration+1,add_nb_iter);
    m_values.resizeWithReserve(nb_value+sub_size,add_nb_value);
    m_iterations[nb_iteration] = iteration;
    for( Integer i=0; i<sub_size; ++i )
      m_values[nb_value+i] = values[i];
  }

  void removeAfterIteration(Integer last_iteration) override
  {
    Integer size = m_iterations.size();
    Integer last_elem = size;
    for( Integer i=0; i<size; ++i )
      if (m_iterations[i]>=last_iteration){
        last_elem = i;
        break;
      }
    if (last_elem!=size){
      m_iterations.resize(last_elem);
      m_values.resize(last_elem*subSize());
    }
  }

  // Ecriture d'une courbe pour les écrivains version 2.
  void dumpValues(ITraceMng* msg,
                  ITimeHistoryCurveWriter2* writer,
                  const TimeHistoryCurveWriterInfo& infos) const override
  {
    ARCANE_UNUSED(msg);

    // Pour l'instant, on ne fait rien
    if (m_shrink_history==true)
      return;
    // Pour vérifier qu'on ne sauve pas plus d'itérations qu'il y en
    // a actuellement (ce qui peut arriver en cas de retour arrière).
    Integer max_iter = infos.times().size();
    RealUniqueArray values_to_write;
    Int32UniqueArray iterations_to_write;
    Integer nb_iteration = m_iterations.size();
    iterations_to_write.reserve(nb_iteration);
    Integer sub_size = subSize();
    values_to_write.reserve(nb_iteration*sub_size);
    for(Integer i=0, is=nb_iteration; i<is; ++i ){
      Integer iter = m_iterations[i];
      if (iter<max_iter){
        for(Integer z=0; z<sub_size; ++z )
          values_to_write.add(Convert::toReal(m_values[(i*sub_size)+ z]));
        iterations_to_write.add(iter);
      }
    }
    if(!meshHandle().isNull()){
      TimeHistoryCurveInfo curve_info(name(), meshHandle().meshName(), iterations_to_write, values_to_write, sub_size);
      writer->writeCurve(curve_info);
    }
    else{
      TimeHistoryCurveInfo curve_info(name(), iterations_to_write, values_to_write, sub_size);
      writer->writeCurve(curve_info);
    }
  }

  void applyTransformation(ITraceMng* msg,ITimeHistoryTransformer* v) override
  {
    ITimeHistoryTransformer::CommonInfo ci;
    ci.name = name();
    SharedArray<Int32> iterations(m_iterations.asArray());
    ci.iterations = iterations;
    Integer sub_size = subSize();
    ci.sub_size = subSize();

    SharedArray<DataType> values(m_values.asArray());

    v->transform(ci,values);

    Integer nb_iteration = iterations.size();
    Integer nb_value = values.size();
    if (nb_iteration*sub_size!=nb_value){
      msg->warning() << "Bad size after history transformation";
      return;
    }

    m_iterations.resize(nb_iteration);
    for( Integer i=0; i<nb_iteration; ++i )
      m_iterations[i] = iterations[i];

    m_values.resize(nb_value);
    for( Integer i=0; i<nb_value; ++i )
      m_values[i] = values[i];
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

class ARCANE_IMPL_EXPORT TimeHistoryMngInternal
: public ITimeHistoryMngInternal
{
 public:
  explicit TimeHistoryMngInternal(ISubDomain* sd)
  : m_sd(sd)
  , m_tmng(sd->traceMng())
  , m_th_meta_data(VariableBuildInfo(m_sd,"TimeHistoryMngMetaData"))
  , m_th_global_time(VariableBuildInfo(m_sd,"TimeHistoryMngGlobalTime"))
  , m_is_active(true)
  , m_is_shrink_active(false)
  , m_is_dump_active(true)
  , m_properties(new Properties(sd->propertyMng(), "ArcaneTimeHistoryProperties"))
  , m_version(2)
  {
    m_enable_non_io_master_curves = !platform::getEnvironmentVariable("ARCANE_ENABLE_NON_IO_MASTER_CURVES").null();
    // Seul le sous-domaine maître des IO rend actif les time history.
    m_is_master_io = sd->allReplicaParallelMng()->isMasterIO();
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
  void addValue(const TimeHistoryAddValueArgInternal& thpi,Real value) override
  {
    RealConstArrayView values(1,&value);
    _addHistoryValue(thpi,values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi,Int64 value) override
  {
    Int64ConstArrayView values(1,&value);
    _addHistoryValue(thpi,values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi,Int32 value) override
  {
    Int32ConstArrayView values(1,&value);
    _addHistoryValue(thpi,values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi,RealConstArrayView values) override
  {
    _addHistoryValue(thpi,values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi,Int32ConstArrayView values) override
  {
    _addHistoryValue(thpi,values);
  }
  void addValue(const TimeHistoryAddValueArgInternal& thpi,Int64ConstArrayView values) override
  {
    _addHistoryValue(thpi,values);
  }

  void addNowInGlobalTime() override;
  void updateGlobalTimeCurve() override;
  void resizeArrayAfterRestore() override;
  void dumpCurves(ITimeHistoryCurveWriter2* writer) override;
  void dumpHistory(bool is_verbose) override;
  void updateMetaData() override;
  void readVariables() override;

  void addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer) override;
  void removeCurveWriter(const String& name) override;
  void applyTransformation(ITimeHistoryTransformer* v) override;

  bool isShrinkActive() const override { return m_is_shrink_active; }
  void setShrinkActive(bool is_active) override { m_is_shrink_active = is_active; }
  bool active() const override { return m_is_active; }
  void setActive(bool is_active) override { m_is_active = is_active; }
  bool isDumpActive() const override { return m_is_dump_active; }
  void setDumpActive(bool is_active) override { m_is_dump_active = is_active; }
  bool isMasterIO() override { return m_is_master_io; }
  bool isNonIOMasterCurvesEnabled() override { return m_enable_non_io_master_curves; }
  void addObservers() override;


 private:
  template<class DataType> void
  _addHistoryValue(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<DataType> values);
  void _checkOutputPath();
  void _destroyAll();
  void _dumpCurvesAllWriters(bool is_verbose);
  void _dumpSummaryOfCurvesLegacy();
  void _dumpSummaryOfCurves();
  void _removeCurveWriter(const Ref<ITimeHistoryCurveWriter2>& writer)
  {
    m_curve_writers2.erase(writer);
  }

  void _fromLegacyFormat();
  void _saveProperties();

 private:
  ISubDomain* m_sd;
  ITraceMng* m_tmng;

  bool m_is_master_io; //!< True si je suis le gestionnaire actif
  bool m_enable_non_io_master_curves; //!< Indique si l'ecriture  de courbes par des procs non io_master est possible
  String m_output_path;
  ObserverPool m_observer_pool;
  HistoryList m_history_list; //!< Liste des historiques
  VariableScalarString m_th_meta_data; //!< Infos des historiques
  VariableArrayReal m_th_global_time; //!< Tableau des instants de temps
  RealUniqueArray m_global_times; //!< Liste des temps globaux
  CurveWriter2List m_curve_writers2;
  bool m_is_active; //!< Indique si le service est actif.
  bool m_is_shrink_active; //!< Indique si la compression de l'historique est active
  bool m_is_dump_active; //!< Indique si les dump sont actifs
  Properties* m_properties;
  Integer m_version;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
