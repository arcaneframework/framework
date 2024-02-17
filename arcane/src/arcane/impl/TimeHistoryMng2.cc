// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMng2.cc                                          (C) 2000-2023 */
/*                                                                           */
/* Module gérant un historique de valeurs (Version 2).                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"

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

#include "arcane/impl/internal/TimeHistoryMngInternal.h"

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
  ~TimeHistoryMng2() override;

 public:
	
  VersionInfo versionInfo() const override { return VersionInfo(1,0,0); }

 public:

  void addValue(const String& name,Real value,bool end_time,bool is_local) override
  {
    RealConstArrayView values(1,&value);
    m_internal->addValue(name,values,end_time,is_local);
  }
  void addValue(const String& name,Int64 value,bool end_time,bool is_local) override
  {
    Int64ConstArrayView values(1,&value);
    m_internal->addValue(name,values,end_time,is_local);
  }
  void addValue(const String& name,Int32 value,bool end_time,bool is_local) override
  {
    Int32ConstArrayView values(1,&value);
    m_internal->addValue(name,values,end_time,is_local);
  }
  void addValue(const String& name,RealConstArrayView values,bool end_time,bool is_local) override
  {
    m_internal->addValue(name,values,end_time,is_local);
  }
  void addValue(const String& name,Int32ConstArrayView values,bool end_time,bool is_local) override
  {
    m_internal->addValue(name,values,end_time,is_local);
  }
  void addValue(const String& name,Int64ConstArrayView values,bool end_time,bool is_local) override
  {
    m_internal->addValue(name,values,end_time,is_local);
  }

 public:

  void timeHistoryBegin() override;
  void timeHistoryEnd() override;
  void timeHistoryInit() override;
  void timeHistoryStartInit() override;
  void timeHistoryContinueInit() override;
  void timeHistoryRestore() override;
  void timeHistoryStartInitEnd();

 public:

  void addCurveWriter(ITimeHistoryCurveWriter2* writer) override;
  void removeCurveWriter(ITimeHistoryCurveWriter2* writer) override
  {
    m_curve_writers2.erase(writer);
  }
  void removeCurveWriter(const String& name) override;

 public:

  void dumpHistory(bool is_verbose) override;
  void dumpCurves(ITimeHistoryCurveWriter2* writer) override;

  bool active() const override { return m_internal->active(); }
  void setActive(bool is_active) override { m_internal->setActive(is_active); }

  bool isDumpActive() const override { return m_internal->isDumpActive(); }
  void setDumpActive(bool is_active) override { m_internal->setDumpActive(is_active); }

  bool isShrinkActive() const override { return m_internal->isShrinkActive(); }
  void setShrinkActive(bool is_active) override { m_internal->setShrinkActive(is_active); }

  ITimeHistoryMngInternal* _internalApi() override {return m_internal.get();}

  void applyTransformation(ITimeHistoryTransformer* v) override;

 private:

  bool m_is_master_io; //!< True si je suis le gestionnaire actif
  bool m_enable_non_io_master_curves; //!< Indique si l'ecriture  de courbes par des procs non io_master est possible
  String m_output_path;
  ObserverPool m_observer_pool;
  HistoryList m_history_list; //!< Liste des historiques
  RealUniqueArray m_global_times; //!< Liste des temps globaux
  CurveWriter2List m_curve_writers2;
  Ref<ITimeHistoryMngInternal> m_internal;

 private:

  void _writeVariablesNotify();
  void _checkOutputPath();
  void _destroyAll();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeHistoryMng2::
TimeHistoryMng2(const ModuleBuildInfo& mb, bool add_entry_points)
: AbstractModule(mb)
, CommonVariables(this)
, m_is_master_io(true)
, m_enable_non_io_master_curves(false)
, m_internal(makeRef(new TimeHistoryMngInternal(subDomain())))
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
  arcaneCallFunctionAndCatchException([&]() { _destroyAll(); });

  m_curve_writers2.clear();
  m_history_list.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
_destroyAll()
{
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue2* v = i->second;
    delete v;
  }

  for( auto& c : m_curve_writers2 ) {
    auto cw = c;
    cw.destroy();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryStartInit()
{
  //warning() << "timeHistoryStartInit " << m_global_time() << " " << m_global_times.size();
  m_internal->addNowInGlobalTime();
}

/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryStartInitEnd()
{
  m_internal->updateThGlobalTime();
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
    m_internal->addNowInGlobalTime();
  }
  
  // Regarde s'il faut imprimer les sorties temporelles
  {
    bool force_print_thm = false;
    int th_step = subDomain()->caseOptionsMain()->writeHistoryPeriod();
    if (th_step!=0){
      if ((globalIteration() % th_step)==0)
        if (parallelMng()->isMasterIO() || m_enable_non_io_master_curves)
          force_print_thm = true;
    }
    if (subDomain()->applicationInfo().isDebug())
      force_print_thm = true;
    if (force_print_thm)
      m_internal->dumpHistory(false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryEnd()
{
  m_internal->updateThGlobalTime();
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
    m_internal->addCurveWriter(makeRef(gnuplot_curve_writer));
  }

  if (m_is_master_io || m_enable_non_io_master_curves){
    ServiceBuilder<ITimeHistoryCurveWriter2> builder(subDomain());
    auto writers = builder.createAllInstances();
    for( auto& wr_ref : writers ){
      ITimeHistoryCurveWriter2* cw = wr_ref.get();
      if (cw){
        info() << "FOUND CURVE SERVICE (V2)!";
        m_internal->addCurveWriter(wr_ref);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
addCurveWriter(ITimeHistoryCurveWriter2* writer)
{
  // TODO bof
  m_internal->addCurveWriter(makeRef(writer));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
_writeVariablesNotify()
{
  m_internal->updateMetaData();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryContinueInit()
{
  if (m_is_master_io || m_enable_non_io_master_curves)
    m_internal->_readVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryRestore()
{
  m_internal->timeHistoryRestore();
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
  m_internal->dumpHistory(is_verbose);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
dumpCurves(ITimeHistoryCurveWriter2* writer)
{
  m_internal->dumpCurves(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
applyTransformation(ITimeHistoryTransformer* v)
{
  m_internal->applyTransformation(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
removeCurveWriter(const String& name)
{
  m_internal->removeCurveWriter(name);
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
