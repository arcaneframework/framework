// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMngInternal.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Classe interne gérant un historique de valeurs.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/TimeHistoryMngInternal.h"
#include "arcane/core/IMeshMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer)
{
  m_tmng->info() << "Add CurveWriter2 name=" << writer->name();
  if(m_is_master_io || m_enable_non_io_master_curves)
    m_curve_writers2.insert(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
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

void TimeHistoryMngInternal::
updateMetaData()
{
  OStringStream meta_data_str;

  meta_data_str() << "<?xml version='1.0' ?>\n";
  meta_data_str() << "<curves version='1'>\n";
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue* val = i->second;
    if(val->meshHandle().isNull()){
      meta_data_str() << "<curve "
                      << " name='" << val->name() << "'"
                      << " index='" << val->index() << "'"
                      << " data-type='" << dataTypeName(val->dataType()) << "'"
                      << " sub-size='" << val->subSize() << "'"
                      << "/>\n";
    }
    else{
      meta_data_str() << "<curve "
                      << " name='" << val->name() << "'"
                      << " index='" << val->index() << "'"
                      << " data-type='" << dataTypeName(val->dataType()) << "'"
                      << " sub-size='" << val->subSize() << "'"
                      << " support='" << val->meshHandle().meshName() << "'"
                      << "/>\n";
    }
  }
  meta_data_str() << "</curves>\n";

  {
    String ss = meta_data_str.str();
    m_th_meta_data = ss;
    //warning() << "TimeHistoryMng MetaData: size=" << ss.len() << " v=" << ss;
  }

  updateGlobalTimeCurve();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
addNowInGlobalTime()
{
  m_global_times.add(m_sd->commonVariables().globalTime());
  TimeHistoryAddValueArgInternal thpi(m_sd->commonVariables().m_global_time.name(), true, false);
  addValue(thpi, m_sd->commonVariables().globalTime());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
updateGlobalTimeCurve()
{
  m_th_global_time.resize(m_global_times.size());
  m_th_global_time.copy(m_global_times);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
dumpCurves(ITimeHistoryCurveWriter2* writer)
{
  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;

  TimeHistoryCurveWriterInfo infos(m_output_path, m_global_times.constView());
  writer->beginWrite(infos);
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    const TimeHistoryValue& th = *(i->second);
    th.dumpValues(m_tmng,writer,infos);
  }
  writer->endWrite();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
dumpHistory(bool is_verbose)
{
  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;
  if (!m_is_dump_active)
    return;

  _checkOutputPath();
  _dumpCurvesAllWriters(is_verbose);
  _dumpSummaryOfCurvesLegacy();

  m_tmng->info() << "Fin sortie historique: " << platform::getCurrentDateTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
applyTransformation(ITimeHistoryTransformer* v)
{
  if (!m_is_master_io && !m_enable_non_io_master_curves)
    return;
  for( IterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue& th = *(i->second);
    th.applyTransformation(m_tmng,v);
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
readVariables()
{
  bool need_update = _fromOldFormat();

  m_tmng->info(4) << "readVariables resizes m_global_time to " << m_th_global_time.size();
  m_global_times.resize(m_th_global_time.size());
  m_global_times.copy(m_th_global_time);

  m_tmng->info() << "Reading the values history";

  IIOMng* io_mng = m_sd->ioMng();
  ScopedPtrT<IXmlDocumentHolder> doc(io_mng->parseXmlString(m_th_meta_data(),"meta_data"));
  if (!doc.get()){
    m_tmng->error() << " METADATA len=" << m_th_meta_data().length()
                    << " str='" << m_th_meta_data() << "'";
    ARCANE_FATAL("The meta-data of TimeHistoryMng2 are invalid.");
  }
  XmlNode root_node = doc->documentNode();
  XmlNode curves_node = root_node.child(String("curves"));
  //Integer version = curves_node.attr("version").valueAsInteger();
  XmlNodeList curves = curves_node.children(String("curve"));
  String ustr_name("name");
  String ustr_index("index");
  String ustr_sub_size("sub-size");
  String ustr_data_type("data-type");
  String ustr_support("support");

  for( XmlNode curve : curves ){
    String name = curve.attrValue(ustr_name);
    Integer index = curve.attr(ustr_index).valueAsInteger();
    Integer sub_size = curve.attr(ustr_sub_size).valueAsInteger();
    String data_type_str = curve.attrValue(ustr_data_type);
    eDataType dt = dataTypeFromName(data_type_str.localstr());
    String support_str = curve.attrValue(ustr_support, false);

    if (name.null())
      ARCANE_FATAL("null name for curve");
    if (index<0)
      ARCANE_FATAL("Invalid index '{0}' for curve",index);

    TimeHistoryValue* val = nullptr;
    if(support_str.null()){
      switch(dt){
      case DT_Real:
        val = new TimeHistoryValueT<Real>(m_sd,name,index,sub_size,isShrinkActive());
        break;
      case DT_Int32:
        val = new TimeHistoryValueT<Int32>(m_sd,name,index,sub_size,isShrinkActive());
        break;
      case DT_Int64:
        val = new TimeHistoryValueT<Int64>(m_sd,name,index,sub_size,isShrinkActive());
        break;
      default:
        break;
      }
    }
    else{
      MeshHandle mh = m_sd->meshMng()->findMeshHandle(support_str);
      switch(dt){
      case DT_Real:
        val = new TimeHistoryValueT<Real>(mh,name,index,sub_size,isShrinkActive());
        break;
      case DT_Int32:
        val = new TimeHistoryValueT<Int32>(mh,name,index,sub_size,isShrinkActive());
        break;
      case DT_Int64:
        val = new TimeHistoryValueT<Int64>(mh,name,index,sub_size,isShrinkActive());
        break;
      default:
        break;
      }
      // Important dans le cas où on a deux historiques de même nom pour deux maillages différents,
      // ou le même nom qu'un historique "globale".
      name = name + "_" + mh.meshName();
    }
    if(need_update){
      val->fromOldToNewVariables(m_sd);
    }
    if (!val)
      ARCANE_FATAL("Bad data-type");
    m_history_list.insert(HistoryValueType(name,val));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
resizeArrayAfterRestore()
{
  Integer current_iteration = m_sd->commonVariables().globalIteration();
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
      m_tmng->info() << "TimeHistoryRestore: truncating TimeHistoryGlobalTime array to size n=" << n << "\n";
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

void TimeHistoryMngInternal::
_checkOutputPath()
{
  if (m_output_path.empty()){
    Directory out_dir(m_sd->exportDirectory(),"courbes");
    m_output_path = out_dir.path();
    if (out_dir.createDirectory()){
      m_tmng->warning() << "Can't create the output directory '" << m_output_path << "'";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_dumpCurvesAllWriters(bool is_verbose)
{
  if (is_verbose) {
    Directory out_dir(m_output_path);
    m_tmng->info() << "Writing of the history of values path=" << out_dir.path();
  }
  if (m_is_master_io || m_enable_non_io_master_curves) {
    m_tmng->info() << "Begin output history: " << platform::getCurrentDateTime();

    // Ecriture via version 2 des curve writers
    for( auto& cw_ref : m_curve_writers2 ){
      ITimeHistoryCurveWriter2* writer = cw_ref.get();
      if (is_verbose){
        m_tmng->info() << "Writing curves with '" << writer->name()
                       << "' date=" << platform::getCurrentDateTime();
      }
      dumpCurves(writer);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_dumpSummaryOfCurvesLegacy()
{
  // Génère un fichier xml contenant la liste des courbes de l'historique
  Directory out_dir(m_output_path);
  IParallelMng* parallel_mng = m_sd->parallelMng();
  Integer master_io_rank = parallel_mng->masterIORank() ;
  if (m_is_master_io) {
    std::ofstream ofile(out_dir.file("time_history.xml").localstr());
    ofile << "<?xml version='1.0' ?>\n";
    ofile << "<curves>\n";
    for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
      const TimeHistoryValue& th = *(i->second);
      if(!th.meshHandle().isNull()){
        ofile << "<curve name='" <<  th.name() << "_" << th.meshHandle().meshName() << "'/>\n";
      }
      else{
        ofile << "<curve name='" <<  th.name() << "'/>\n";
      }
    }
    if (m_enable_non_io_master_curves) {
      for(Integer i=0;i<parallel_mng->commSize();++i)
        if(i!=master_io_rank) {
          Integer nb_curve = 0 ;
          parallel_mng->recv(ArrayView<Integer>(1,&nb_curve),i);
          for(Integer icurve=0;icurve<nb_curve;++icurve) {
            UniqueArray<Int32> length(2);
            parallel_mng->recv(length,i);
            if(length[1] == 0){
              UniqueArray<char> buf(length[0]);
              parallel_mng->recv(buf,i);
              ofile << "<curve name='" << buf.unguardedBasePointer() << "'/>\n";
            }
            else{
              UniqueArray<char> buf(length[0]);
              UniqueArray<char> buf2(length[1]);
              parallel_mng->recv(buf,i);
              parallel_mng->recv(buf2,i);
              ofile << "<curve name='" <<  buf.unguardedBasePointer() << "_" << buf2.unguardedBasePointer() << "'/>\n";
            }
          }
        }
    }
    ofile << "</curves>\n";
  }
  else if(m_enable_non_io_master_curves) {
    Integer nb_curve = arcaneCheckArraySize(m_history_list.size());
    parallel_mng->send(ArrayView<Integer>(1,&nb_curve),master_io_rank);
    for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
      const TimeHistoryValue& th = *(i->second);
      String name = th.name();
      UniqueArray<Int32> length(2);
      length[0] = arcaneCheckArraySize(name.length()+1);
      if(th.meshHandle().isNull()){
        length[1] = 0;
        parallel_mng->send(length, master_io_rank);
        parallel_mng->send(ConstArrayView<char>(length[0],static_cast<const char*>(name.localstr())), master_io_rank);
      }
      else{
        String mesh_name = th.meshHandle().meshName();
        length[1] = arcaneCheckArraySize(mesh_name.length()+1);
        parallel_mng->send(length, master_io_rank);
        parallel_mng->send(ConstArrayView<char>(length[0],static_cast<const char*>(name.localstr())), master_io_rank);
        parallel_mng->send(ConstArrayView<char>(length[1],static_cast<const char*>(mesh_name.localstr())), master_io_rank);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
void TimeHistoryMngInternal::
_addHistoryValue(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<DataType> values)
{
  if (!m_is_master_io && !(m_enable_non_io_master_curves && thpi.thp().isLocal()))
    return;

  if (!m_is_active)
    return;

  String name_to_find = thpi.thp().name().clone();
  if(!thpi.meshHandle().isNull()){
    // Important dans le cas où on a deux historiques de même nom pour deux maillages différents,
    // ou le même nom qu'un historique "globale".
    name_to_find = name_to_find + "_" + thpi.meshHandle().meshName();
  }

  Integer iteration = m_sd->commonVariables().globalIteration();

  if (!thpi.thp().endTime() && iteration!=0)
    --iteration;

  auto hl = m_history_list.find(name_to_find);
  TimeHistoryValueT<DataType>* th = nullptr;
  // Trouvé, on le retourne.
  if (hl!=m_history_list.end())
    th = dynamic_cast<TimeHistoryValueT<DataType>* >(hl->second);
  else{
    if(!thpi.meshHandle().isNull()) {
      th = new TimeHistoryValueT<DataType>(thpi.meshHandle(), thpi.thp().name(), (Integer)m_history_list.size(), values.size(), isShrinkActive());
    }
    else{
      th = new TimeHistoryValueT<DataType>(m_sd, thpi.thp().name(), (Integer)m_history_list.size(), values.size(), isShrinkActive());
    }
    m_history_list.insert(HistoryValueType(name_to_find, th));
  }
  if (!th)
    return;
  if (values.size()!=th->subSize()){
    ARCANE_FATAL("Bad subsize for curve '{0}' current={1} old={2}",
                 name_to_find, values.size(), th->subSize());
  }
  th->addValue(values,iteration);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_destroyAll()
{
  for( ConstIterT<HistoryList> i(m_history_list); i(); ++i ){
    TimeHistoryValue* v = i->second;
    delete v;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TimeHistoryMngInternal::
_fromOldFormat()
{
  if(!m_th_global_time.empty()) return false;
  IMesh* mesh0 = m_sd->defaultMesh();

  IVariable* ptr_old_global_time = m_sd->variableMng()->findMeshVariable(mesh0, "TimeHistoryGlobalTime");
  if(ptr_old_global_time == nullptr)
    return false;

  IVariable* ptr_old_meta_data = m_sd->variableMng()->findMeshVariable(mesh0, "TimeHistoryMetaData");
  if(ptr_old_meta_data == nullptr)
    ARCANE_FATAL("TimeHistoryGlobalTime without TimeHistoryMetaData is not possible.");

  m_tmng->warning() << "Old TimeHistory variables found (TimeHistoryGlobalTime and TimeHistoryMetaData). Copying in new variables...";

  VariableArrayReal old_global_time(ptr_old_global_time);
  VariableScalarString old_meta_data(ptr_old_meta_data);

  m_th_global_time.resize(old_global_time.size());
  m_th_global_time.copy(old_global_time);
  m_th_meta_data.swapValues(old_meta_data);

  old_global_time.resize(0);
  old_meta_data.reset();

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
