// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMngInternal.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Classe interne gérant un historique de valeurs.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/TimeHistoryMngInternal.h"

#include "arcane/core/IMeshMng.h"
#include "arcane/core/IPropertyMng.h"

#include "arcane/utils/JSONWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer)
{
  m_trace_mng->info() << "Add CurveWriter2 name=" << writer->name();
  if (m_is_master_io || (m_enable_non_io_master_curves && m_is_master_io_of_sd))
    m_curve_writers2.insert(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
removeCurveWriter(const String& name)
{
  for (auto& cw : m_curve_writers2)
    if (cw->name() == name) {
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
  meta_data_str() << "<curves>\n";
  for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
    TimeHistoryValue* val = i->second;
    meta_data_str() << "<curve "
                    << " name='" << val->name() << "'"
                    << " index='" << val->index() << "'"
                    << " data-type='" << dataTypeName(val->dataType()) << "'"
                    << " sub-size='" << val->subSize() << "'";

    if (!val->meshHandle().isNull()) {
      meta_data_str() << " support='" << val->meshHandle().meshName() << "'";
    }
    if (val->isLocal()) {
      meta_data_str() << " sub-domain='" << val->localSubDomainId() << "'";
    }

    meta_data_str() << "/>\n";
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
addObservers(IPropertyMng* prop_mng)
{
  m_observer_pool.addObserver(this,
                              &TimeHistoryMngInternal::_saveProperties,
                              prop_mng->writeObservable());

  m_observer_pool.addObserver(this,
                              &TimeHistoryMngInternal::updateMetaData,
                              m_variable_mng->writeObservable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_saveProperties()
{
  auto p = m_properties;
  p->setInt32("version", m_version);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
addNowInGlobalTime()
{
  m_global_times.add(m_common_variables.globalTime());
  TimeHistoryAddValueArgInternal thpi(m_common_variables.m_global_time.name(), true, NULL_SUB_DOMAIN_ID);
  addValue(thpi, m_common_variables.globalTime());
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
  if (!m_is_master_io && (!m_enable_non_io_master_curves || !m_is_master_io_of_sd))
    return;

  if (!m_io_master_write_only) {
    TimeHistoryCurveWriterInfo infos(m_output_path, m_global_times.constView());
    writer->beginWrite(infos);
    for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
      const TimeHistoryValue& th = *(i->second);
      th.dumpValues(m_trace_mng, writer, infos);
    }
    writer->endWrite();
  }

  else {

#ifdef ARCANE_CHECK
    if (m_enable_non_io_master_curves) {
      bool all_need_comm = m_parallel_mng->reduce(MessagePassing::ReduceMin, m_need_comm);
      if (all_need_comm != m_need_comm) {
        ARCANE_FATAL("Internal error: m_need_comm not sync");
      }
    }
#endif

    if (m_is_master_io) {
      TimeHistoryCurveWriterInfo infos(m_output_path, m_global_times.constView());
      writer->beginWrite(infos);
      // Nos courbes
      for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
        const TimeHistoryValue& th = *(i->second);
        th.dumpValues(m_trace_mng, writer, infos);
      }

      // Les courbes reçues.
      if (m_need_comm && m_enable_non_io_master_curves) {
        Integer master_io_rank = m_parallel_mng->masterIORank();
        UniqueArray<Int32> length(5);

        for (Integer i = 0; i < m_parallel_mng->commSize(); ++i) {
          if (i != master_io_rank) {
            Integer nb_curve = 0;
            m_parallel_mng->recv(ArrayView<Integer>(1, &nb_curve), i);
            for (Integer icurve = 0; icurve < nb_curve; ++icurve) {
              m_parallel_mng->recv(length, i);

              UniqueArray<char> buf(length[0]);
              UniqueArray<Int32> iterations_to_write(length[1]);
              UniqueArray<Real> values_to_write(length[2]);

              m_parallel_mng->recv(buf, i);
              m_parallel_mng->recv(iterations_to_write, i);
              m_parallel_mng->recv(values_to_write, i);

              String name = String(buf.unguardedBasePointer());

              if (length[4] != 0) {
                UniqueArray<char> buf2(length[4]);
                m_parallel_mng->recv(buf2, i);
                String name_mesh = String(buf2.unguardedBasePointer());

                TimeHistoryCurveInfo curve_info(name, name_mesh, iterations_to_write, values_to_write, length[3], i);
                writer->writeCurve(curve_info);
              }
              else {
                TimeHistoryCurveInfo curve_info(name, iterations_to_write, values_to_write, length[3], i);
                writer->writeCurve(curve_info);
              }
            }
          }
        }
      }

      writer->endWrite();
    }
    else if (m_need_comm) {
      TimeHistoryCurveWriterInfo infos(m_output_path, m_global_times.constView());
      Integer master_io_rank = m_parallel_mng->masterIORank();
      Integer nb_curve = arcaneCheckArraySize(m_history_list.size());
      UniqueArray<Int32> length(5);
      m_parallel_mng->send(ArrayView<Integer>(1, &nb_curve), master_io_rank);
      for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
        const TimeHistoryValue& th = *(i->second);
        String name = th.name();
        UniqueArray<Int32> iterations_to_write;
        UniqueArray<Real> values_to_write;
        th.arrayToWrite(iterations_to_write, values_to_write, infos);

        length[0] = arcaneCheckArraySize(name.length() + 1);
        length[1] = iterations_to_write.size();
        length[2] = values_to_write.size();
        length[3] = th.subSize();
        if (!th.meshHandle().isNull()) {
          length[4] = arcaneCheckArraySize(th.meshHandle().meshName().length() + 1);
        }
        else {
          length[4] = 0;
        }

        m_parallel_mng->send(length, master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[0], static_cast<const char*>(name.localstr())), master_io_rank);
        m_parallel_mng->send(iterations_to_write, master_io_rank);
        m_parallel_mng->send(values_to_write, master_io_rank);
        if (!th.meshHandle().isNull()) {
          m_parallel_mng->send(ConstArrayView<char>(length[4], static_cast<const char*>(th.meshHandle().meshName().localstr())), master_io_rank);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
dumpHistory()
{
  if (!m_is_master_io && (!m_enable_non_io_master_curves || !m_is_master_io_of_sd))
    return;
  if (!m_is_dump_active)
    return;

  _dumpCurvesAllWriters();
  _dumpSummaryOfCurvesLegacy();
  _dumpSummaryOfCurves();

  m_trace_mng->info() << "Fin sortie historique: " << platform::getCurrentDateTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
applyTransformation(ITimeHistoryTransformer* v)
{
  if (!m_is_master_io && (!m_enable_non_io_master_curves || !m_is_master_io_of_sd))
    return;
  for (IterT<HistoryList> i(m_history_list); i(); ++i) {
    TimeHistoryValue& th = *(i->second);
    th.applyTransformation(m_trace_mng, v);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
readVariables(IMeshMng* mesh_mng, IMesh* default_mesh)
{
  bool need_update = false;

  auto p = m_properties;

  Int32 version = 0;
  if (!p->get("version", version)) {
    version = 1;
    m_trace_mng->info() << "The checkpoint contains legacy format of TimeHistory variables, updating...";
    _fromLegacyFormat(default_mesh);
    need_update = true;
  }
  else if (version == 2) {
    m_trace_mng->info() << "TimeHistory Variables version 2";
  }
  else {
    ARCANE_FATAL("Unknown TimeHistory Variables format -- Found version: {0}", version);
  }

  m_trace_mng->info(4) << "readVariables resizes m_global_time to " << m_th_global_time.size();
  m_global_times.resize(m_th_global_time.size());
  m_global_times.copy(m_th_global_time);

  m_trace_mng->info() << "Reading the values history";

  IIOMng* io_mng = m_parallel_mng->ioMng();
  ScopedPtrT<IXmlDocumentHolder> doc(io_mng->parseXmlString(m_th_meta_data(), "meta_data"));
  if (!doc.get()) {
    m_trace_mng->error() << " METADATA len=" << m_th_meta_data().length()
                         << " str='" << m_th_meta_data() << "'";
    ARCANE_FATAL("The meta-data of TimeHistoryMng2 are invalid.");
  }
  XmlNode root_node = doc->documentNode();
  XmlNode curves_node = root_node.child(String("curves"));
  XmlNodeList curves = curves_node.children(String("curve"));

  // v1
  String ustr_name("name");
  String ustr_index("index");
  String ustr_sub_size("sub-size");
  String ustr_data_type("data-type");

  // v2
  String ustr_support("support");
  String ustr_sub_domain("sub-domain");

  for (XmlNode curve : curves) {
    String name = curve.attrValue(ustr_name);
    Integer index = curve.attr(ustr_index).valueAsInteger();
    Integer sub_size = curve.attr(ustr_sub_size).valueAsInteger();
    String data_type_str = curve.attrValue(ustr_data_type);
    eDataType dt = dataTypeFromName(data_type_str.localstr());
    String support_str = curve.attrValue(ustr_support, false);

    XmlNode sub_domain_node = curve.attr(ustr_sub_domain);
    Integer sub_domain = NULL_SUB_DOMAIN_ID;
    if (!sub_domain_node.null()) {
      sub_domain = sub_domain_node.valueAsInteger();
      m_need_comm = true;
    }

    if (sub_domain != NULL_SUB_DOMAIN_ID && m_parallel_mng->commRank() != sub_domain) {
      continue;
    }

    if (name.null())
      ARCANE_FATAL("null name for curve");
    if (index < 0)
      ARCANE_FATAL("Invalid index '{0}' for curve", index);

    TimeHistoryValue* val = nullptr;
    if (support_str.null()) {
      TimeHistoryAddValueArgInternal thpi(name, true, sub_domain);
      switch (dt) {
      case DT_Real:
        val = new TimeHistoryValueT<Real>(m_variable_mng, thpi, index, sub_size, isShrinkActive());
        break;
      case DT_Int32:
        val = new TimeHistoryValueT<Int32>(m_variable_mng, thpi, index, sub_size, isShrinkActive());
        break;
      case DT_Int64:
        val = new TimeHistoryValueT<Int64>(m_variable_mng, thpi, index, sub_size, isShrinkActive());
        break;
      default:
        ARCANE_FATAL("Unsupported type");
      }
      if (need_update) {
        val->fromOldToNewVariables(m_variable_mng, default_mesh);
      }
    }
    else {
      MeshHandle mh = mesh_mng->findMeshHandle(support_str);
      TimeHistoryAddValueArgInternal thpi(TimeHistoryAddValueArg(name, true, sub_domain), mh);
      switch (dt) {
      case DT_Real:
        val = new TimeHistoryValueT<Real>(thpi, index, sub_size, isShrinkActive());
        break;
      case DT_Int32:
        val = new TimeHistoryValueT<Int32>(thpi, index, sub_size, isShrinkActive());
        break;
      case DT_Int64:
        val = new TimeHistoryValueT<Int64>(thpi, index, sub_size, isShrinkActive());
        break;
      default:
        ARCANE_FATAL("Unsupported type");
      }
      // Important dans le cas où on a deux historiques de même nom pour deux maillages différents,
      // ou le même nom qu'un historique "globale".
      name = name + "_" + mh.meshName();
    }
    if (sub_domain != NULL_SUB_DOMAIN_ID) {
      name = name + "_Local";
    }
    m_history_list.insert(HistoryValueType(name, val));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
resizeArrayAfterRestore()
{
  Integer current_iteration = m_common_variables.globalIteration();
  {
    // Vérifie qu'on n'a pas plus d'éléments que d'itérations
    // dans 'm_th_global_time'. Normalement cela ne peut arriver
    // que lors d'un retour-arrière si les variables ont été sauvegardées
    // au cours du pas de temps.
    // TODO: ce test ne fonctionne pas si isShrinkActive() est vrai.
    Integer n = m_th_global_time.size();
    if (n > current_iteration) {
      n = current_iteration;
      m_th_global_time.resize(n);
      m_trace_mng->info() << "TimeHistoryRestore: truncating TimeHistoryGlobalTime array to size n=" << n << "\n";
    }
  }
  m_global_times.resize(m_th_global_time.size());
  m_global_times.copy(m_th_global_time);

  for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
    i->second->removeAfterIteration(current_iteration);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
editOutputPath(const Directory& directory)
{
  m_directory = directory;
  if (m_output_path.empty()) {
    m_output_path = m_directory.path();
    if (m_directory.createDirectory()) {
      m_trace_mng->warning() << "Can't create the output directory '" << m_output_path << "'";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
iterationsAndValues(const TimeHistoryAddValueArgInternal& thpi, UniqueArray<Int32>& iterations, UniqueArray<Real>& values)
{
  if (!m_is_master_io && (!thpi.timeHistoryAddValueArg().isLocal() || !m_enable_non_io_master_curves || !m_is_master_io_of_sd)) {
    return;
  }

  if (thpi.timeHistoryAddValueArg().isLocal() && thpi.timeHistoryAddValueArg().localSubDomainId() != m_parallel_mng->commRank()) {
    return;
  }

  if (!m_is_active) {
    return;
  }

  String name_to_find = thpi.timeHistoryAddValueArg().name().clone();
  if (!thpi.meshHandle().isNull()) {
    // Important dans le cas où on a deux historiques de même nom pour deux maillages différents,
    // ou le même nom qu'un historique "globale".
    name_to_find = name_to_find + "_" + thpi.meshHandle().meshName();
  }
  if (thpi.timeHistoryAddValueArg().isLocal()) {
    name_to_find = name_to_find + "_Local";
  }

  auto hl = m_history_list.find(name_to_find);

  if (hl != m_history_list.end()) {
    TimeHistoryCurveWriterInfo infos(m_output_path, m_global_times.constView());
    hl->second->arrayToWrite(iterations, values, infos);
  }
  else {
    iterations.clear();
    values.clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_dumpCurvesAllWriters()
{
  m_trace_mng->debug() << "Writing of the history of values path=" << m_output_path;

  if (m_is_master_io || (m_enable_non_io_master_curves && m_is_master_io_of_sd)) {
    m_trace_mng->info() << "Begin output history: " << platform::getCurrentDateTime();

    // Ecriture via version 2 des curve writers
    for (auto& cw_ref : m_curve_writers2) {
      ITimeHistoryCurveWriter2* writer = cw_ref.get();
      m_trace_mng->debug() << "Writing curves with '" << writer->name()
                           << "' date=" << platform::getCurrentDateTime();
      dumpCurves(writer);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_dumpSummaryOfCurvesLegacy()
{
  // Seul le processus master écrit.
  Integer master_io_rank = m_parallel_mng->masterIORank();
  if (m_is_master_io) {
    std::ofstream ofile(m_directory.file("time_history.xml").localstr());
    ofile << "<?xml version='1.0' ?>\n";
    ofile << "<curves>\n";

    // On écrit d'abord le nom de nos courbes.
    for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
      const TimeHistoryValue& th = *(i->second);
      ofile << "<curve name='";
      if (!th.meshHandle().isNull()) {
        ofile << th.meshHandle().meshName() << "_";
      }
      if (th.isLocal()) {
        ofile << "SD" << master_io_rank << "_";
      }
      ofile << th.name() << "'/>\n";
    }
    // Puis, si les autres processus peuvent aussi avoir des courbes, on
    // écrit aussi leurs noms.
    if (m_need_comm && m_enable_non_io_master_curves) {
      for (Integer i = 0; i < m_parallel_mng->commSize(); ++i)
        if (i != master_io_rank) {
          Integer nb_curve = 0;
          m_parallel_mng->recv(ArrayView<Integer>(1, &nb_curve), i);
          for (Integer icurve = 0; icurve < nb_curve; ++icurve) {
            UniqueArray<Int32> length(2);
            m_parallel_mng->recv(length, i);

            UniqueArray<char> buf(length[0]);
            m_parallel_mng->recv(buf, i);
            ofile << "<curve name='";

            if (length[1] != 0) {
              UniqueArray<char> buf2(length[1]);
              m_parallel_mng->recv(buf2, i);
              ofile << buf2.unguardedBasePointer() << "_";
            }
            ofile << "SD" << i << "_";
            ofile << buf.unguardedBasePointer() << "'/>\n";
          }
        }
    }
    ofile << "</curves>\n";
  }

  // Si l'on n'est pas un processus écrivain mais qu'il est possible que l'on possède des courbes.
  else if (m_need_comm && m_enable_non_io_master_curves && m_is_master_io_of_sd) {
    Integer nb_curve = arcaneCheckArraySize(m_history_list.size());
    m_parallel_mng->send(ArrayView<Integer>(1, &nb_curve), master_io_rank);
    for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
      const TimeHistoryValue& th = *(i->second);
      String name = th.name();
      UniqueArray<Int32> length(2);
      length[0] = arcaneCheckArraySize(name.length() + 1);
      if (th.meshHandle().isNull()) {
        length[1] = 0;
        m_parallel_mng->send(length, master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[0], static_cast<const char*>(name.localstr())), master_io_rank);
      }
      else {
        String mesh_name = th.meshHandle().meshName();
        length[1] = arcaneCheckArraySize(mesh_name.length() + 1);
        m_parallel_mng->send(length, master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[0], static_cast<const char*>(name.localstr())), master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[1], static_cast<const char*>(mesh_name.localstr())), master_io_rank);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_dumpSummaryOfCurves()
{
  // Seul le processus master écrit.
  if (m_is_master_io) {

    JSONWriter json_writer(JSONWriter::FormatFlags::None);
    {
      JSONWriter::Object o1(json_writer);
      {
        JSONWriter::Object o2(json_writer, "arcane-curves");
        json_writer.write("version", 1);
        {
          Integer master_io_rank = m_parallel_mng->masterIORank();
          json_writer.writeKey("curves");
          json_writer.beginArray();

          // On écrit d'abord le nom de nos courbes.
          for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
            JSONWriter::Object o4(json_writer);
            const TimeHistoryValue& th = *(i->second);
            json_writer.write("name", th.name());

            String name("");

            if (!th.meshHandle().isNull()) {
              json_writer.write("support", th.meshHandle().meshName());
              name = name + th.meshHandle().meshName() + "_";
            }

            if (th.isLocal()) {
              json_writer.write("sub-domain", master_io_rank);
              name = name + "SD" + String::fromNumber(master_io_rank) + "_";
            }

            json_writer.write("unique-name", name + th.name());
          }

          // Puis, si les autres processus peuvent aussi avoir des courbes, on
          // écrit aussi leurs noms.
          if (m_need_comm && m_enable_non_io_master_curves) {
            for (Integer i = 0; i < m_parallel_mng->commSize(); ++i) {
              if (i != master_io_rank) {
                Integer nb_curve = 0;
                m_parallel_mng->recv(ArrayView<Integer>(1, &nb_curve), i);
                for (Integer icurve = 0; icurve < nb_curve; ++icurve) {
                  JSONWriter::Object o4(json_writer);
                  UniqueArray<Int32> length(2);
                  m_parallel_mng->recv(length, i);

                  UniqueArray<char> buf(length[0]);
                  m_parallel_mng->recv(buf, i);
                  json_writer.write("name", buf.unguardedBasePointer());

                  String name("");

                  if (length[1] != 0) {
                    UniqueArray<char> buf2(length[1]);
                    m_parallel_mng->recv(buf2, i);
                    json_writer.write("support", buf2.unguardedBasePointer());
                    name = name + buf2.unguardedBasePointer() + "_";
                  }

                  name = name + "SD" + String::fromNumber(i) + "_";

                  json_writer.write("sub-domain", i);
                  json_writer.write("unique-name", name + buf.unguardedBasePointer());
                }
              }
            }
          }
          json_writer.endArray();
        }
      }
    }

    Directory out_dir(m_output_path);
    std::ofstream ofile(out_dir.file("time_history.json").localstr());
    ofile << json_writer.getBuffer();
    ofile.close();
  }

  // Si l'on n'est pas un processus écrivain mais qu'il est possible que l'on possède des courbes.
  else if (m_need_comm && m_enable_non_io_master_curves && m_is_master_io_of_sd) {

    Integer master_io_rank = m_parallel_mng->masterIORank();

    Integer nb_curve = arcaneCheckArraySize(m_history_list.size());
    m_parallel_mng->send(ArrayView<Integer>(1, &nb_curve), master_io_rank);
    for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
      const TimeHistoryValue& th = *(i->second);
      String name = th.name();
      UniqueArray<Int32> length(2);
      length[0] = arcaneCheckArraySize(name.length() + 1);
      if (th.meshHandle().isNull()) {
        length[1] = 0;
        m_parallel_mng->send(length, master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[0], static_cast<const char*>(name.localstr())), master_io_rank);
      }
      else {
        String mesh_name = th.meshHandle().meshName();
        length[1] = arcaneCheckArraySize(mesh_name.length() + 1);
        m_parallel_mng->send(length, master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[0], static_cast<const char*>(name.localstr())), master_io_rank);
        m_parallel_mng->send(ConstArrayView<char>(length[1], static_cast<const char*>(mesh_name.localstr())), master_io_rank);
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
  if (!m_is_active) {
    return;
  }

  if (thpi.timeHistoryAddValueArg().isLocal()) {
    m_need_comm = true;
  }

  if (!m_is_master_io && (!thpi.timeHistoryAddValueArg().isLocal() || !m_enable_non_io_master_curves || !m_is_master_io_of_sd)) {
    return;
  }

  if (thpi.timeHistoryAddValueArg().isLocal() && thpi.timeHistoryAddValueArg().localSubDomainId() != m_parallel_mng->commRank()) {
    return;
  }

  String name_to_find = thpi.timeHistoryAddValueArg().name().clone();
  if (!thpi.meshHandle().isNull()) {
    // Important dans le cas où on a deux historiques de même nom pour deux maillages différents,
    // ou le même nom qu'un historique "globale".
    name_to_find = name_to_find + "_" + thpi.meshHandle().meshName();
  }
  if (thpi.timeHistoryAddValueArg().isLocal()) {
    name_to_find = name_to_find + "_Local";
  }

  Integer iteration = m_common_variables.globalIteration();

  if (!thpi.timeHistoryAddValueArg().endTime() && iteration != 0)
    --iteration;

  auto hl = m_history_list.find(name_to_find);
  TimeHistoryValueT<DataType>* th = nullptr;
  // Trouvé, on le retourne.
  if (hl != m_history_list.end())
    th = dynamic_cast<TimeHistoryValueT<DataType>*>(hl->second);
  else {
    if (!thpi.meshHandle().isNull()) {
      th = new TimeHistoryValueT<DataType>(thpi, (Integer)m_history_list.size(), values.size(), isShrinkActive());
    }
    else {
      th = new TimeHistoryValueT<DataType>(m_variable_mng, thpi, (Integer)m_history_list.size(), values.size(), isShrinkActive());
    }
    m_history_list.insert(HistoryValueType(name_to_find, th));
  }
  if (!th)
    return;
  if (values.size() != th->subSize()) {
    ARCANE_FATAL("Bad subsize for curve '{0}' current={1} old={2}",
                 name_to_find, values.size(), th->subSize());
  }
  th->addValue(values, iteration);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_destroyAll()
{
  for (ConstIterT<HistoryList> i(m_history_list); i(); ++i) {
    TimeHistoryValue* v = i->second;
    delete v;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_fromLegacyFormat(IMesh* default_mesh)
{
  IVariable* ptr_old_global_time = m_variable_mng->findMeshVariable(default_mesh, "TimeHistoryGlobalTime");
  IVariable* ptr_old_meta_data = m_variable_mng->findMeshVariable(default_mesh, "TimeHistoryMetaData");
  if (ptr_old_global_time == nullptr || ptr_old_meta_data == nullptr)
    ARCANE_FATAL("TimeHistoryGlobalTime or TimeHistoryMetaData is not found.");

  VariableArrayReal old_global_time(ptr_old_global_time);
  VariableScalarString old_meta_data(ptr_old_meta_data);

  m_th_global_time.resize(old_global_time.size());
  m_th_global_time.copy(old_global_time);
  m_th_meta_data.swapValues(old_meta_data);

  old_global_time.resize(0);
  old_meta_data.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMngInternal::
_removeCurveWriter(const Ref<ITimeHistoryCurveWriter2>& writer)
{
  m_curve_writers2.erase(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template void TimeHistoryMngInternal::
_addHistoryValue<Real>(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<Real> values);

template void TimeHistoryMngInternal::
_addHistoryValue<Int32>(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<Int32> values);

template void TimeHistoryMngInternal::
_addHistoryValue<Int64>(const TimeHistoryAddValueArgInternal& thpi, ConstArrayView<Int64> values);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
