﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMng2.cc                                          (C) 2000-2021 */
/*                                                                           */
/* Module gérant un historique de valeurs (Version 2).                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/StringImpl.h"

#include "arcane/ITimeHistoryMng.h"
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

#include "arcane/datatype/DataTypeTraits.h"

#include <map>
#include <set>
#include <variant>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
/*!
 * \brief Classe pour encapsuler soit un pointeur, soit une référence sur
 * un service.
 *
 * Cela permet de gérer la destruction de manière homogène sans se
 * préoccuper du type sous-jacent.
 *
 * \todo Utiliser compteur de référence pour le pointeur pour ne pas avoir
 * à appeler explicitement destroy().
 */
template<typename PointerType>
class AnyRef
{
  typedef AnyRef<PointerType> ThatClass;
 public:
  typedef PointerType* Value1;
  typedef Ref<PointerType> Value2;
 public:
  AnyRef(Value1 pt) : m_value(pt){}
  AnyRef(Value2 pt) : m_value(pt){}
 public:
  PointerType* get() const
  {
    if (std::holds_alternative<Value1>(m_value))
      return std::get<Value1>(m_value);
    if (std::holds_alternative<Value2>(m_value))
      return std::get<Value2>(m_value).get();
    ARCANE_FATAL("Bad get()");
  }
  PointerType* operator->() const { return get(); }
  void destroy()
  {
    if (std::holds_alternative<Value1>(m_value)){
      delete std::get<Value1>(m_value);
      m_value = nullptr;
    }
  }
  bool operator<(const ThatClass& ref) const { return get() < ref.get(); }
 private:
  std::variant<Value1,Value2> m_value;
};
} // End anonymous namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un historique de valeurs.
 *
 * Un historique contient un ensemble de valeurs pour certaines itérations.
 * Il est caractérisé par un nom.
 */
class TimeHistoryValue2
{
 public:

 public:
  
  TimeHistoryValue2(const String& name,
                    eDataType dt,
                    Integer index,
                    Integer sub_size)
  : m_name(name), m_data_type(dt), m_index(index), m_sub_size(sub_size) {}
  virtual ~TimeHistoryValue2(){} //!< Libére les ressources

 public:
 
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

private:

  String m_name; //!< Nom de l'historique
  eDataType m_data_type; //!< Type de la donnée
  Integer m_index; //!< Index de l'historique dans la liste
  Integer m_sub_size;
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
class TimeHistoryValue2T
: public TimeHistoryValue2
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
  TimeHistoryValue2T(IModule* module,
                     const String& name,
                     Integer index,
                     Integer nb_element,
                     bool shrink=false)
  : TimeHistoryValue2(name,DataTypeTraitsT<DataType>::type(),index,nb_element)
  , m_values(VariableBuildInfo(module,String("TimeHistory_Values_")+index,VAR_BUILD_FLAGS))
  , m_iterations(VariableBuildInfo(module,String("TimeHistory_Iterations_")+index,VAR_BUILD_FLAGS))
  , m_use_compression(false)
  , m_shrink_history(shrink)
  {
  }

 public:

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
    TimeHistoryCurveInfo curve_info(name(),iterations_to_write,values_to_write,sub_size);
    writer->writeCurve(curve_info);
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
/*!
 * \brief Ecrivain au format GNUPLOT.
 */
class GnuplotTimeHistoryCurveWriter2
: public TraceAccessor
, public ITimeHistoryCurveWriter2
{
 public:
  GnuplotTimeHistoryCurveWriter2(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:
  void build() override {}
  void beginWrite(const TimeHistoryCurveWriterInfo& infos) override
  {
    m_times = infos.times();
    String path = infos.path();
    // m_output_path surcharge les infos en argument si non vide.
    if (m_output_path.empty())
      m_output_path = path;

    m_gnuplot_path = Directory(Directory(m_output_path),"gnuplot");
    // Créé le répertoire de sortie.
    if (m_gnuplot_path.createDirectory()){
      warning() << "Can not create gnuplot curve directory '"
                << m_gnuplot_path.path() << "'";
    }
  }
  void writeCurve(const TimeHistoryCurveInfo& infos) override
  {
    String sname(m_gnuplot_path.file(infos.name()));
    FILE* ofile = fopen(sname.localstr(),"w");
    if (!ofile){
      warning() << "Can not open gnuplot curve file '" << sname << "'";
      return;
    }
    RealConstArrayView values = infos.values();
    Int32ConstArrayView iterations = infos.iterations();
    Integer nb_val = iterations.size();
    Integer sub_size = infos.subSize();
    for( Integer i=0; i<nb_val; ++i ){
      fprintf(ofile,"%.16E",Convert::toDouble(m_times[iterations[i]]));
      for( Integer z=0; z<sub_size; ++z )
        fprintf(ofile," %.16E",Convert::toDouble(values[(i*sub_size)+z]));
      fprintf(ofile,"\n");
    }
    fclose(ofile);
  }

  void endWrite() override {}

  String name() const override { return "gnuplot"; }

  void setOutputPath(const String& path) override
  {
    m_output_path = path;
  }
  String outputPath() const override
  {
    return m_output_path;
  }

 private:

  String m_output_path;
  UniqueArray<Real> m_times;
  Directory m_gnuplot_path;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire d'un historique de valeurs.
 *
 * IMPORTANT: ce module fournit une interface aux autres modules
 * par l'intermédiare de ITimeHistoryMng. Par conséquent, il faut être
 * sur de ne pas toucher aux variables de ce module pendant un appel à une
 * des méthodes de ITimeHistoryMng.
 */
class TimeHistoryMng2
: public AbstractModule
, public CommonVariables
, public ITimeHistoryMng
{
 public:

  typedef std::map<String,TimeHistoryValue2*> HistoryList;
  typedef HistoryList::value_type HistoryValueType;
  typedef HistoryList::iterator HistoryListIterator;
  typedef HistoryList::const_iterator HistoryListConstIterator;
  typedef std::set<AnyRef<ITimeHistoryCurveWriter2>> CurveWriter2List;

 public:
	
  TimeHistoryMng2(const ModuleBuildInfo& cb, bool add_entry_points=true);
  virtual ~TimeHistoryMng2();

 public:
	
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,0); }

 public:

  virtual void addValue(const String& name,Real value,bool end_time,bool is_local=false)
	{
    RealConstArrayView values(1,&value);
    _addHistoryValue(name,values,end_time,is_local);
  }
  virtual void addValue(const String& name,Int64 value,bool end_time,bool is_local=false)
	{
    Int64ConstArrayView values(1,&value);
    _addHistoryValue(name,values,end_time,is_local);
  }
  virtual void addValue(const String& name,Int32 value,bool end_time,bool is_local=false)
	{
    Int32ConstArrayView values(1,&value);
    _addHistoryValue(name,values,end_time,is_local);
  }
  virtual void addValue(const String& name,RealConstArrayView values,bool end_time=true,bool is_local=false)
  {
    _addHistoryValue(name,values,end_time,is_local);
  }
  virtual void addValue(const String& name,Int32ConstArrayView values,bool end_time=true,bool is_local=false)
  {
    _addHistoryValue(name,values,end_time,is_local);
  }
  virtual void addValue(const String& name,Int64ConstArrayView values,bool end_time=true,bool is_local=false)
  {
    _addHistoryValue(name,values,end_time,is_local);
  }

 public:

  void timeHistoryBegin();
  void timeHistoryEnd();
  void timeHistoryInit();
  void timeHistoryStartInit();
  void timeHistoryContinueInit();
  void timeHistoryStartInitEnd();
  void timeHistoryRestore();

 public:

  virtual void addCurveWriter(ITimeHistoryCurveWriter2* writer);
  virtual void removeCurveWriter(ITimeHistoryCurveWriter2* writer)
  {
    m_curve_writers2.erase(writer);
  }
  virtual void removeCurveWriter(const String& name);

 public:

  virtual void dumpHistory(bool is_verbose);
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer);

  virtual bool active() const { return m_is_active; }
  virtual void setActive(bool is_active) { m_is_active = is_active; }

  virtual bool isDumpActive() const { return m_is_dump_active; }
  virtual void setDumpActive(bool is_active) { m_is_dump_active = is_active; }

  virtual bool isShrinkActive() const { return m_is_shrink_active; }
  virtual void setShrinkActive(bool is_active) { m_is_shrink_active = is_active; }

  virtual void applyTransformation(ITimeHistoryTransformer* v);

 private:

  bool m_is_master_io; //!< True si je suis le gestionnaire actif
  bool m_enable_non_io_master_curves; //!< Indique si l'ecriture  de courbes par des procs non io_master est possible
  bool m_is_active; //!< Indique si le service est actif.
  bool m_is_dump_active; //!< Indique si les dump sont actifs
  bool m_is_shrink_active; //!< Indique si la compression de l'historique est active
  String m_output_path;
  ObserverPool m_observer_pool;
  HistoryList m_history_list; //!< Liste des historiques
  RealUniqueArray m_global_times; //!< Liste des temps globaux
  VariableScalarString m_th_meta_data; //!< Infos des historiques
  VariableArrayReal m_th_global_time; //!< Tableau des instants de temps
  CurveWriter2List m_curve_writers2;

 private:

  template<class DataType> void
  _addHistoryValue(const String& name,ConstArrayView<DataType> value,bool end_time,bool is_local);
  void _addCurveWriter(AnyRef<ITimeHistoryCurveWriter2> writer);
  void _removeCurveWriter(AnyRef<ITimeHistoryCurveWriter2> writer)
  {
    m_curve_writers2.erase(writer);
  }

  void _writeVariablesNotify();
  void _readVariables();
  void _checkOutputPath();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeHistoryMng2::
TimeHistoryMng2(const ModuleBuildInfo& mb, bool add_entry_points)
: AbstractModule(mb)
, CommonVariables(this)
, m_is_master_io(true)
, m_enable_non_io_master_curves(false)
, m_is_active(true)
, m_is_dump_active(true)
, m_is_shrink_active(false)
, m_th_meta_data(VariableBuildInfo(this,"TimeHistoryMetaData"))
, m_th_global_time(VariableBuildInfo(this,"TimeHistoryGlobalTime"))
{
  if (add_entry_points){
    addEntryPoint(this,"ArcaneTimeHistoryBegin",&TimeHistoryMng2::timeHistoryBegin,
                  IEntryPoint::WComputeLoop,IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this,"ArcaneTimeHistoryEnd",&TimeHistoryMng2::timeHistoryEnd,
                  IEntryPoint::WComputeLoop,IEntryPoint::PAutoLoadEnd);
    addEntryPoint(this,"ArcaneTimeHistoryInit",&TimeHistoryMng2::timeHistoryInit,
                  IEntryPoint::WInit,IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this,"ArcaneTimeHistoryStartInit",&TimeHistoryMng2::timeHistoryStartInit,
                  IEntryPoint::WStartInit,IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this,"ArcaneTimeHistoryContinueInit",&TimeHistoryMng2::timeHistoryContinueInit,
                  IEntryPoint::WContinueInit,IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this,"ArcaneTimeHistoryStartInitEnd",&TimeHistoryMng2::timeHistoryStartInitEnd,
                  IEntryPoint::WStartInit,IEntryPoint::PAutoLoadEnd);
    addEntryPoint(this,"ArcaneTimeHistoryRestore",&TimeHistoryMng2::timeHistoryRestore,
                  IEntryPoint::WRestore,IEntryPoint::PAutoLoadBegin);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeHistoryMng2::
~TimeHistoryMng2()
{
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue2* v = i->second;
    delete v;
  }

  for( auto cw : m_curve_writers2 )
    cw.destroy();
  m_curve_writers2.clear();

  m_history_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryStartInit()
{
  //warning() << "timeHistoryStartInit " << m_global_time() << " " << m_global_times.size();
  m_global_times.add(m_global_time());
  addValue(m_global_time.name(),m_global_time(),true);
}

/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryStartInitEnd()
{
  m_th_global_time.resize(m_global_times.size());
  m_th_global_time.copy(m_global_times);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryBegin()
{
  // Si on n'est pas actif, on ne grossit pas inutilement le m_global_times
  // qui sera copié dans la variable backupée 'm_th_global_time'
  if (isShrinkActive() && !active()){
    // On ne fait rien
  }
  else{
    //warning() << "timeHistoryBegin " << m_global_time() << " " << m_global_times.size();
    m_global_times.add(m_global_time());
    addValue(m_global_time.name(),m_global_time(),true);
  }
  
  // Regarde s'il faut imprimer les sorties temporelles
  {
    bool force_print_thm = false;
    int th_step = subDomain()->caseOptionsMain()->writeHistoryPeriod();
    if (th_step!=0){
      if ((globalIteration() % th_step)==0)
        if (parallelMng()->isMasterIO())
          force_print_thm = true;
    }
    if (subDomain()->applicationInfo().isDebug())
      force_print_thm = true;
    if (force_print_thm)
      dumpHistory(false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryEnd()
{
  m_th_global_time.resize(m_global_times.size());
  m_th_global_time.copy(m_global_times);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryInit()
{
  //warning() << "timeHistoryInit " << m_global_time() << " " << m_global_times.size();
  
  ISubDomain* sd = subDomain();
  IVariableMng* vm = sd->variableMng();
  
  // Seul le sous-domaine maître des IO rend actif les time history.
  m_is_master_io = sd->allReplicaParallelMng()->isMasterIO();
  info(4) << "TimeHistory is MasterIO ? " << m_is_master_io;
  m_enable_non_io_master_curves = ! platform::getEnvironmentVariable("ARCANE_ENABLE_NON_IO_MASTER_CURVES").null() ;

  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;

  _checkOutputPath();
  Directory out_dir(m_output_path);
  if (out_dir.createDirectory()){
    warning() << "Can't create the output directory '" << m_output_path << "'";
  }

  m_observer_pool.addObserver(this,
                              &TimeHistoryMng2::_writeVariablesNotify,
                              vm->writeObservable());
  
  if (platform::getEnvironmentVariable("ARCANE_DISABLE_GNUPLOT_CURVES").null()){
    ITimeHistoryCurveWriter2* gnuplot_curve_writer = new GnuplotTimeHistoryCurveWriter2(traceMng());
    addCurveWriter(gnuplot_curve_writer);
  }

  if (m_is_master_io || m_enable_non_io_master_curves){
    ServiceBuilder<ITimeHistoryCurveWriter2> builder(subDomain());
    auto writers = builder.createAllInstances();
    for( auto& wr_ref : writers ){
      ITimeHistoryCurveWriter2* cw = wr_ref.get();
      if (cw){
        info() << "FOUND CURVE SERVICE (V2)!";
        _addCurveWriter(wr_ref);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
addCurveWriter(ITimeHistoryCurveWriter2* writer)
{
  _addCurveWriter(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
_addCurveWriter(AnyRef<ITimeHistoryCurveWriter2> writer)
{
  info() << "Add CurveWriter2 name=" << writer->name();
  if(m_is_master_io || m_enable_non_io_master_curves)
    m_curve_writers2.insert(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
_writeVariablesNotify()
{
  OStringStream meta_data_str;

  meta_data_str() << "<?xml version='1.0' ?>\n";
  meta_data_str() << "<curves>\n";
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue2* val = i->second;
    meta_data_str() << "<curve "
                    << " name='" << val->name() << "'"
                    << " index='" << val->index() << "'"
                    << " data-type='" << dataTypeName(val->dataType()) << "'"
                    << " sub-size='" << val->subSize() << "'"
                    << "/>\n";
  }
  meta_data_str() << "</curves>\n";

  {
    String ss = meta_data_str.str();
    m_th_meta_data = ss;
    //warning() << "TimeHistoryMng MetaData: size=" << ss.len() << " v=" << ss;
  }

  m_th_global_time.resize(m_global_times.size());
  m_th_global_time.copy(m_global_times);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
_readVariables()
{
  info(4) << "_readVariables resizes m_global_time to " << m_th_global_time.size();
  m_global_times.resize(m_th_global_time.size());
  m_global_times.copy(m_th_global_time);

  info() << "Reading the values history";
  
  IIOMng* io_mng = subDomain()->ioMng();
  ScopedPtrT<IXmlDocumentHolder> doc(io_mng->parseXmlString(m_th_meta_data(),"meta_data"));
  if (!doc.get()){
    error() << " METADATA len=" << m_th_meta_data().length()
            << " str='" << m_th_meta_data() << "'";
    ARCANE_FATAL("The meta-data of TimeHistoryMng2 are invalid.");
  }
  XmlNode root_node = doc->documentNode();
  XmlNode curves_node = root_node.child(String("curves"));
  XmlNodeList curves = curves_node.children(String("curve"));
  String ustr_name("name");
  String ustr_index("index");
  String ustr_sub_size("sub-size");
  String ustr_data_type("data-type");

  for( XmlNode curve : curves ){
    String name = curve.attrValue(ustr_name);
    Integer index = curve.attr(ustr_index).valueAsInteger();
    Integer sub_size = curve.attr(ustr_sub_size).valueAsInteger();
    String data_type_str = curve.attrValue(ustr_data_type);
    eDataType dt = dataTypeFromName(data_type_str.localstr());
    if (name.null())
      ARCANE_FATAL("null name for curve");
    if (index<0)
      ARCANE_FATAL("Invalid index '{0}' for curve",index);
    TimeHistoryValue2* val = 0;
    switch(dt){
    case DT_Real:
      val = new TimeHistoryValue2T<Real>(this,name,index,sub_size,isShrinkActive());
      break;
    case DT_Int32:
      val = new TimeHistoryValue2T<Int32>(this,name,index,sub_size,isShrinkActive());
      break;
    case DT_Int64:
      val = new TimeHistoryValue2T<Int64>(this,name,index,sub_size,isShrinkActive());
      break;
    default:
      break;
    }
    if (!val)
      ARCANE_FATAL("Bad data-type");
    m_history_list.insert(HistoryValueType(name,val));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryContinueInit()
{
  if (m_is_master_io || m_enable_non_io_master_curves)
    _readVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryRestore()
{
  Integer current_iteration = m_global_iteration();
  {
    // Vérifie qu'on n'a pas plus d'éléments que d'itérations
    // dans 'm_th_global_time'. Normalement cela ne peut arriver
    // que lors d'un retour-arrière si les variables ont été sauvegardées
    // au cours du pas de temps.
    // TODO: ce test ne fonctionne pas si isShrinkActive() est vrai.
    Integer n = m_th_global_time.size();
    if (n>current_iteration){
      n = current_iteration;
      m_th_global_time.resize(n);
      info() << "TimeHistoryRestore: truncating TimeHistoryGlobalTime array to size n=" << n << "\n";
    }
  }
  m_global_times.resize(m_th_global_time.size());
  m_global_times.copy(m_th_global_time);

  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    i->second->removeAfterIteration(current_iteration);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
_checkOutputPath()
{
  if (m_output_path.empty()){
    Directory d(subDomain()->exportDirectory(),"courbes");
    m_output_path = d.path();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
dumpHistory(bool is_verbose)
{
  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;
  if (!m_is_dump_active)
    return;
  
  _checkOutputPath();

  ITraceMng* tm = traceMng();
  Directory out_dir(m_output_path);

  if (is_verbose)
    info() << "Writing of the history of values path=" << out_dir.path();
  if (m_is_master_io || m_enable_non_io_master_curves) {
    info() << "Begin output history: " << platform::getCurrentDateTime();

    // Ecriture via version 2 des curve writers
    for( auto& cw_ref : m_curve_writers2 ){
      ITimeHistoryCurveWriter2* writer = cw_ref.get();
        if (is_verbose){
         info() << "Writing curves with '" << writer->name()
                << "' date=" << platform::getCurrentDateTime();
        }
        TimeHistoryCurveWriterInfo infos(out_dir.path(),m_global_times.constView());
        writer->beginWrite(infos);
        for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
          const TimeHistoryValue2& th = *(i->second);
          th.dumpValues(tm,writer,infos);
        }
        writer->endWrite();
    }
  }

 
  // Génère un fichier xml contenant la liste des courbes de l'historique
  ISubDomain* sd = subDomain();
  IParallelMng* parallel_mng = sd->parallelMng();
  Integer master_io_rank = parallel_mng->masterIORank() ;

  if (m_is_master_io) {
    std::ofstream ofile(out_dir.file("time_history.xml").localstr());
    ofile << "<?xml version='1.0' ?>\n";
    ofile << "<curves>\n";
    for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
      const TimeHistoryValue2& th = *(i->second);
      ofile << "<curve name='" <<  th.name() << "'/>\n";
    }
    if (m_enable_non_io_master_curves) {
      for(Integer i=0;i<parallel_mng->commSize();++i)
        if(i!=master_io_rank) {
          Integer nb_curve = 0 ;
          parallel_mng->recv(ArrayView<Integer>(1,&nb_curve),i);
          for(Integer icurve=0;icurve<nb_curve;++icurve) {
            Integer length = 0 ;
            parallel_mng->recv(ArrayView<Integer>(1,&length),i) ;
            UniqueArray<char> buf(length) ;
            parallel_mng->recv(buf,i) ;
            ofile << "<curve name='" <<  buf.unguardedBasePointer() << "'/>\n";
          }
        }
    }
    ofile << "</curves>\n";
  }
  else if(m_enable_non_io_master_curves) {
    Integer nb_curve = arcaneCheckArraySize(m_history_list.size());
    parallel_mng->send(ArrayView<Integer>(1,&nb_curve),master_io_rank);
    for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
      const TimeHistoryValue2& th = *(i->second);
      String name = th.name() ;
      Integer length = arcaneCheckArraySize(name.length()+1);
      parallel_mng->send(ArrayView<Integer>(1,&length),master_io_rank) ;
      parallel_mng->send(ArrayView<char>(length,(char*)name.localstr()),master_io_rank) ;
    }
  }
  info() << "Fin sortie historique: " << platform::getCurrentDateTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
dumpCurves(ITimeHistoryCurveWriter2* writer)
{
  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;
  ITraceMng* tm = traceMng();
  TimeHistoryCurveWriterInfo infos(m_output_path,m_global_times.constView());
  writer->beginWrite(infos);
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    const TimeHistoryValue2& th = *(i->second);
    th.dumpValues(tm,writer,infos);
  }
  writer->endWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
applyTransformation(ITimeHistoryTransformer* v)
{
  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;
  ITraceMng* tm = traceMng();
  for( IterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue2& th = *(i->second);
    th.applyTransformation(tm,v);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void TimeHistoryMng2::
_addHistoryValue(const String& name,ConstArrayView<DataType> values,bool end_time,bool is_local)
{
  if (!m_is_master_io && !(m_enable_non_io_master_curves && is_local))
    return;

  if (!m_is_active)
    return;

  Integer iteration = globalIteration();
  
  if (!end_time && iteration!=0)
    --iteration;

  HistoryList::iterator hl = m_history_list.find(name);
  TimeHistoryValue2T<DataType>* th = 0;
  // Trouvé, on le retourne.
  if (hl!=m_history_list.end())
    th = dynamic_cast< TimeHistoryValue2T<DataType>* >(hl->second);
  else{
    th = new TimeHistoryValue2T<DataType>(this,name,(Integer)m_history_list.size(),
                                          values.size(),isShrinkActive());
    m_history_list.insert(HistoryValueType(name,th));
  }
  if (!th)
    return;
  if (values.size()!=th->subSize()){
    ARCANE_FATAL("Bad subsize for curve '{0}' current={1} old={2}",
                 name,values.size(),th->subSize());
  }
  th->addValue(values,iteration);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
removeCurveWriter(const String& name)
{
  for ( auto& cw : m_curve_writers2)
    if (cw->name()==name){
      _removeCurveWriter(cw);
      return;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT ITimeHistoryMng*
arcaneCreateTimeHistoryMng2(ISubDomain* mng)
{
  return new TimeHistoryMng2(ModuleBuildInfo(mng,"TimeHistoryMng"));
}
extern "C++" ARCANE_IMPL_EXPORT ITimeHistoryMng*
arcaneCreateTimeHistoryMng2(ISubDomain* mng, bool add_entry_points)
{
  return new TimeHistoryMng2(ModuleBuildInfo(mng,"TimeHistoryMng"), add_entry_points);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
