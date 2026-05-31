// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCurveWriter.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Writing curves in Arcane format.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/ITimeHistoryCurveWriter2.h"
#include "arcane/BasicService.h"
#include "arcane/FactoryService.h"
#include "arcane/IApplication.h"
#include "arcane/IRessourceMng.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Writing curves in the specific Arcane format.
 *
 * \todo write documentation on the format.
 */
class ArcaneCurveWriter
: public BasicService
, public ITimeHistoryCurveWriter2
{
 public:

  class Impl : TraceAccessor
  {
   public:

    Impl(IApplication* app, ITraceMng* tm, const String& path);

   public:

    String m_file_name;
    std::ofstream m_stream;
    ScopedPtrT<IXmlDocumentHolder> m_curves_doc;
    XmlNode m_root_element;
  };

 public:

  ArcaneCurveWriter(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  , m_version(2)
  {}
  ~ArcaneCurveWriter() {}

 public:

  virtual void build() {}
  virtual void beginWrite(const TimeHistoryCurveWriterInfo& infos);
  virtual void endWrite();
  virtual void writeCurve(const TimeHistoryCurveInfo& infos);
  virtual String name() const { return "ArcaneCurveWriter"; }
  virtual void setOutputPath(const String& path) { m_output_path = path; }
  virtual String outputPath() const { return m_output_path; }

 private:

  ScopedPtrT<Impl> m_p;
  Int32 m_version;
  String m_output_path;

 private:

  void _writeHeader();
  template <typename T>
  Int64 _write(ConstArrayView<T> values)
  {
    Int64 offset = m_p->m_stream.tellp();
    m_p->m_stream.write((const char*)values.data(), values.size() * sizeof(T));
    //info() << "OFFSET offset=" << offset;
    return offset;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCurveWriter::Impl::
Impl(IApplication* app, ITraceMng* tm, const String& path)
: TraceAccessor(tm)
, m_file_name("curves")
{
  String full_path = path + "/" + m_file_name + ".acv";
  info() << "Begin write curves full_path=" << full_path;
  m_stream.open(full_path.localstr(), std::ios::trunc);
  if (!m_stream)
    warning() << "Can not open file '" << full_path << "' for writing curves";
  m_curves_doc = app->ressourceMng()->createXmlDocument();
  XmlNode doc = m_curves_doc->documentNode();
  m_root_element = XmlElement(doc, "curves");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCurveWriter::
beginWrite(const TimeHistoryCurveWriterInfo& infos)
{
  String path = infos.path();
  // m_output_path overrides the infos argument if not empty.
  // TODO: check if the directory needs to be created
  if (!m_output_path.empty())
    m_output_path = path;

  info() << A_FUNCNAME << "Begin write curves path=" << path;
  m_p = new Impl(subDomain()->application(), traceMng(), path);

  _writeHeader();
  Int64 time_offset = _write(infos.times());
  m_p->m_root_element.setAttrValue("times-offset", String::fromNumber(time_offset));
  m_p->m_root_element.setAttrValue("times-size", String::fromNumber(infos.times().size()));
  m_p->m_root_element.setAttrValue("x", "iteration");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCurveWriter::
_writeHeader()
{
  Byte header[12];
  // First 4 bytes to indicate that it is a curve file
  // arcane
  header[0] = 'A';
  header[1] = 'C';
  header[2] = 'V';
  header[3] = (Byte)122;
  // Next 4 bytes for the version
  // Currently version 1 or 2. Version 1 only supports 32-bit sized files. The only difference in version 2 is that
  // the offsets and lengths stored at the end of the file are 64-bit instead of 32.
  header[4] = (Byte)m_version;
  header[5] = 0;
  header[6] = 0;
  header[7] = 0;
  // Next 4 bytes to indicate the endianness.
  Int32 v = 0x01020304;
  Byte* ptr = (Byte*)(&v);
  for (Integer i = 0; i < 4; ++i)
    header[8 + i] = ptr[i];
  m_p->m_stream.write((const char*)header, 12);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCurveWriter::
endWrite()
{
  ByteUniqueArray bytes;
  m_p->m_curves_doc->save(bytes);
  if (m_version == 2) {
    Int64 write_info[2];
    write_info[0] = _write(bytes.constView());
    write_info[1] = bytes.largeSize();
    // Must always be the last write in the file
    _write(Int64ConstArrayView(2, write_info));
  }
  else if (m_version == 1) {
    Int32 write_info[2];
    write_info[0] = CheckedConvert::toInt32(_write(bytes.constView()));
    write_info[1] = bytes.size();
    // Must always be the last write in the file
    _write(Int32ConstArrayView(2, write_info));
  }
  else
    ARCANE_FATAL("Invalid version {0} (valid values are '1' or '2')", m_version);

  info(4) << "End writing curves";

  // Release the pointer
  m_p = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCurveWriter::
writeCurve(const TimeHistoryCurveInfo& infos)
{
  //info() << "Writing curve name=" << infos.m_name;
  Int64 values_offset = _write(infos.values());

  Int32ConstArrayView iterations(infos.iterations());
  Integer nb_val = iterations.size();
  Int32 range_iterations[2];
  // Check if the iterations are contiguous, in which case
  // we only save the first and the last to
  // save space.
  if (nb_val > 3) {
    Int32 first_iter = iterations[0];
    Int32 last_iter = iterations[nb_val - 1];
    Int32 diff = 1 + last_iter - first_iter;
    //info() << "NB_VAL=" << nb_val << " first=" << first_iter
    //       << " last=" << last_iter << " diff=" << diff << " IS_CONTIGOUS=" << (diff==nb_val);
    if (diff == nb_val) {
      range_iterations[0] = first_iter;
      range_iterations[1] = last_iter;
      iterations = Int32ConstArrayView(2, range_iterations);
    }
  }
  Int64 iteration_offset = _write(iterations);

  XmlNode node = m_p->m_root_element.createAndAppendElement("curve");

  String name(infos.name().clone());

  if (infos.subDomain() != NULL_SUB_DOMAIN_ID) {
    name = "SD" + String::fromNumber(infos.subDomain()) + "_" + name;
  }
  if (infos.hasSupport()) {
    name = infos.support() + "_" + name;
  }

  node.setAttrValue("name", name);
  node.setAttrValue("iterations-offset", String::fromNumber(iteration_offset));
  node.setAttrValue("iterations-size", String::fromNumber(iterations.size()));
  node.setAttrValue("values-offset", String::fromNumber(values_offset));
  node.setAttrValue("values-size", String::fromNumber(infos.values().size()));
  node.setAttrValue("sub-size", String::fromNumber(infos.subSize()));
  node.setAttrValue("base-name", infos.name());
  if (infos.hasSupport()) {
    node.setAttrValue("support", infos.support());
  }
  if (infos.subDomain() != NULL_SUB_DOMAIN_ID) {
    node.setAttrValue("sub-domain", String::fromNumber(infos.subDomain()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(ArcaneCurveWriter,
                                   ITimeHistoryCurveWriter2,
                                   ArcaneCurveWriter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
