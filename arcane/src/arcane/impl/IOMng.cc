// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IOMng.cc                                                    (C) 2000-2015 */
/*                                                                           */
/* Gestionnaire des entrées-sorties.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/IApplication.h"
#include "arcane/DomUtils.h"
#include "arcane/XmlNode.h"
#include "arcane/IIOMng.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IXmlDocumentHolder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des entrées sorties.
 */
class IOMng
: public IIOMng
{
 public:
  IOMng(IParallelSuperMng* psm);
  IOMng(IParallelMng* pm);
  ~IOMng() override;

  IXmlDocumentHolder* parseXmlFile(const String& filename, const String& schemaname = String()) override;
  IXmlDocumentHolder* parseXmlFile(const String& filename, const String& schemaname, ByteConstArrayView schema_data) override;
  IXmlDocumentHolder* parseXmlBuffer(Span<const Byte> buffer, const String& name) override;
  IXmlDocumentHolder* parseXmlBuffer(Span<const std::byte> buffer,const String& name) override;
  IXmlDocumentHolder* parseXmlString(const String& str, const String& name) override;
  bool writeXmlFile(IXmlDocumentHolder* doc, const String& filename, const bool indented) override;
  bool collectiveRead(const String& filename, ByteArray& bytes) override
  {
    return collectiveRead(filename,bytes,true);
  }
  bool collectiveRead(const String& filename, ByteArray& bytes, bool is_binary) override;
  bool localRead(const String& filename, ByteArray& bytes) override
  {
    return localRead(filename, bytes, true);
  }
  bool localRead(const String& filename, ByteArray& bytes, bool is_binary) override;

 private:

  IParallelMng* m_parallel_mng;
  IParallelSuperMng* m_parallel_super_mng;
  IThreadMng* m_thread_mng;
  ITraceMng* m_trace_mng;

  template <typename ParallelMngType> bool
  _collectiveRead(ParallelMngType* pm, const String& filename, ByteArray& bytes, bool is_binary);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IIOMng*
arcaneCreateIOMng(IParallelMng* pm)
{
  return new IOMng(pm);
}

extern "C++" ARCANE_IMPL_EXPORT IIOMng*
arcaneCreateIOMng(IParallelSuperMng* psm)
{
  return new IOMng(psm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IOMng::
IOMng(IParallelMng* pm)
: m_parallel_mng(pm)
, m_parallel_super_mng(nullptr)
, m_thread_mng(pm->threadMng())
, m_trace_mng(pm->traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IOMng::
IOMng(IParallelSuperMng* psm)
: m_parallel_mng(nullptr)
, m_parallel_super_mng(psm)
, m_thread_mng(psm->threadMng())
, m_trace_mng(psm->application()->traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IOMng::
~IOMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IOMng::
writeXmlFile(IXmlDocumentHolder* doc, const String& filename, const bool to_indent)
{
  if (!doc)
    return true;
  std::ofstream ofile(filename.localstr());

  // Check if stream is OK
  if (!ofile.good())
    return true;

  Integer indented = to_indent ? 1 : -1;
  domutils::saveDocument(ofile, doc->documentNode().domNode(), indented);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IOMng::
parseXmlFile(const String& filename, const String& schemaname)
{
  return IXmlDocumentHolder::loadFromFile(filename, schemaname, m_trace_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IOMng::
parseXmlFile(const String& filename,
             const String& schemaname,
             ByteConstArrayView schema_data)
{
  dom::DOMImplementation domimp;
  // Lecture du fichier contenant les informations internes.
  return domimp._load(filename, m_trace_mng, schemaname, schema_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IOMng::
parseXmlBuffer(Span<const Byte> buffer, const String& name)
{
  return IXmlDocumentHolder::loadFromBuffer(buffer, name, m_trace_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IOMng::
parseXmlBuffer(Span<const std::byte> buffer, const String& name)
{
  return IXmlDocumentHolder::loadFromBuffer(buffer, name, m_trace_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* IOMng::
parseXmlString(const String& str, const String& name)
{
  dom::DOMImplementation domimp;
  ByteConstArrayView utf8_buf(str.utf8());
  ByteConstSpan buffer(reinterpret_cast<const std::byte*>(utf8_buf.data()), utf8_buf.size());
  // Lecture du fichier contenant les informations internes.
  return domimp._load(buffer, name, m_trace_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ParallelMngType>
bool IOMng::
_collectiveRead(ParallelMngType* pm, const String& filename, ByteArray& bytes, bool is_binary)
{
  // La lecture necessite deux broadcast: un pour la taille du fichier et
  // un pour les valeurs du fichier.
  //IParallelSuperMng* pm = m_application->parallelSuperMng();
  Integer size = 0;
  if (pm->commRank() == 0) {
    if (!localRead(filename, bytes, is_binary)) {
      // Prévient d'un bug si bytes n'était pas vidé et qu'il y a un problème de lecture
      size = bytes.size();
    }
  }
  {
    IntegerArrayView bs(1, &size);
    pm->broadcast(bs, 0);
    //m_trace_mng->info() << "IO_SIZE=" << size;
  }
  bytes.resize(size);
  if (size != 0)
    pm->broadcast(bytes, 0);
  return bytes.empty();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IOMng::
collectiveRead(const String& filename, ByteArray& bytes, bool is_binary)
{
  if (m_parallel_mng)
    return _collectiveRead(m_parallel_mng, filename, bytes, is_binary);
  return _collectiveRead(m_parallel_super_mng, filename, bytes, is_binary);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IOMng::
localRead(const String& filename, ByteArray& bytes, bool is_binary)
{
  return platform::readAllFile(filename,is_binary,bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
