// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DomLibXml2.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Encapsulation of libxml2 DOM.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/Dom.h"
#include "arcane/core/DomUtils.h"
#include "arcane/core/ISharedReference.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlException.h"

#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xmlsave.h>
#include <libxml/parserInternals.h>
#include <libxml/xinclude.h>
#include <libxml/xmlschemas.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::dom
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: FOR NamedNodeMap, there is no corresponding type in libxml2
// and since this can represent different types, a
// specific type must be created to handle this.
class LibXml2_DOMImplementation;

struct _wxmlText : _xmlNode
{};
struct _wxmlComment : _xmlNode
{};
struct _wxmlDocType : _xmlNode
{};
struct _wxmlDocumentFragment : _xmlNode
{};
struct _wxmlCDATA : _xmlNode
{};
struct _wxmlNodeList : _xmlNode
{};
struct _wxmlEntityReference : _xmlNode
{};
struct _wxmlCharacterData : _xmlNode
{};
struct _wxmlProcessingInstruction : _xmlNode
{};
struct _wxmlNamedNodeMapPtr : _xmlNode
{};

typedef _wxmlNamedNodeMapPtr* xmlNamedNodeMapPtr;
typedef _wxmlDocType* xmlDocTypePtr;
typedef _wxmlDocumentFragment* xmlDocumentFragmentPtr;
typedef _wxmlCharacterData* xmlCharacterDataPtr;
typedef _wxmlNodeList* xmlNodeListPtr;
typedef _wxmlEntityReference* xmlEntityReferencePtr;
typedef _wxmlProcessingInstruction* xmlProcessingInstructionPtr;
typedef _wxmlCDATA* xmlCDATAPtr;
typedef _wxmlText* xmlTextPtr;
typedef _wxmlComment* xmlCommentPtr;

static xmlNodePtr impl(NodePrv* p)
{
  return (xmlNodePtr)p;
}
static xmlAttrPtr impl(AttrPrv* p)
{
  return (xmlAttrPtr)p;
}
static xmlElementPtr impl(ElementPrv* p)
{
  return (xmlElementPtr)p;
}
static xmlNamedNodeMapPtr impl(NamedNodeMapPrv* p)
{
  return (xmlNamedNodeMapPtr)p;
}
static xmlDocPtr impl(DocumentPrv* p)
{
  return (xmlDocPtr)p;
}
static xmlDocTypePtr impl(DocumentTypePrv* p)
{
  return (xmlDocTypePtr)p;
}
[[maybe_unused]] static LibXml2_DOMImplementation* impl(ImplementationPrv* p)
{
  return (LibXml2_DOMImplementation*)p;
}
static xmlCharacterDataPtr impl(CharacterDataPrv* p)
{
  return (xmlCharacterDataPtr)p;
}
static xmlTextPtr impl(TextPrv* p)
{
  return (xmlTextPtr)p;
}
[[maybe_unused]] static xmlNodeListPtr impl(NodeListPrv* p)
{
  return (xmlNodeListPtr)p;
}
static xmlDocumentFragmentPtr impl(DocumentFragmentPrv* p)
{
  return (xmlDocumentFragmentPtr)p;
}
static xmlCDATAPtr impl(CDATASectionPrv* p)
{
  return (xmlCDATAPtr)p;
}
static xmlProcessingInstructionPtr impl(ProcessingInstructionPrv* p)
{
  return (xmlProcessingInstructionPtr)p;
}
static xmlEntityReferencePtr impl(EntityReferencePrv* p)
{
  return (xmlEntityReferencePtr)p;
}
[[maybe_unused]] static xmlEntityPtr impl(EntityPrv* p)
{
  return (xmlEntityPtr)p;
}
[[maybe_unused]] static xmlNotationPtr impl(NotationPrv* p)
{
  return (xmlNotationPtr)p;
}
//static ::DOMError* impl(DOMErrorPrv* p) { return (::DOMError*)p; }
//static ::DOMLocator* impl(DOMLocatorPrv* p) { return (::DOMLocator*)p; }

static NodePrv* cvt(xmlNodePtr p)
{
  return (NodePrv*)p;
}
static AttrPrv* cvt(xmlAttrPtr p)
{
  return (AttrPrv*)p;
}
static ElementPrv* cvt(xmlElementPtr p)
{
  return (ElementPrv*)p;
}
static NamedNodeMapPrv* cvt(xmlNamedNodeMapPtr p)
{
  return (NamedNodeMapPrv*)p;
}
static DocumentPrv* cvt(xmlDocPtr p)
{
  return (DocumentPrv*)p;
}
static DocumentTypePrv* cvt(xmlDocTypePtr p)
{
  return (DocumentTypePrv*)p;
}
static ImplementationPrv* cvt(LibXml2_DOMImplementation* p)
{
  return (ImplementationPrv*)p;
}
static CharacterDataPrv* cvt(xmlCharacterDataPtr p)
{
  return (CharacterDataPrv*)p;
}
static TextPrv* cvt(xmlTextPtr p)
{
  return (TextPrv*)p;
}
[[maybe_unused]] static NodeListPrv* cvt(xmlNodeListPtr p)
{
  return (NodeListPrv*)p;
}
[[maybe_unused]] static DocumentFragmentPrv* cvt(xmlDocumentFragmentPtr p)
{
  return (DocumentFragmentPrv*)p;
}
[[maybe_unused]] static CommentPrv* cvt(xmlCommentPtr p)
{
  return (CommentPrv*)p;
}
[[maybe_unused]] static CDATASectionPrv* cvt(xmlCDATAPtr p)
{
  return (CDATASectionPrv*)p;
}
static ProcessingInstructionPrv* cvt(xmlProcessingInstructionPtr p)
{
  return (ProcessingInstructionPrv*)p;
}
[[maybe_unused]] static EntityReferencePrv* cvt(xmlEntityReferencePtr p)
{
  return (EntityReferencePrv*)p;
}
static EntityPrv* cvt(xmlEntityPtr p)
{
  return (EntityPrv*)p;
}
static NotationPrv* cvt(xmlNotationPtr p)
{
  return (NotationPrv*)p;
}
//static DOMLocatorPrv* cvt(::DOMLocator* p) { return (DOMLocatorPrv*)p; }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
typedef char XMLCh;

const xmlChar*
domStringToXmlChar(const String& str)
{
  const xmlChar* ch = reinterpret_cast<const xmlChar*>(str.utf8().data());
  return ch;
}

const xmlChar*
toChar(const String& value)
{
  return domStringToXmlChar(value);
}

String fromChar(const xmlChar* value)
{
  if (!value)
    return DOMString();
  Integer len = ::xmlStrlen(value);
  // Ne pas oublier le '\0' terminal
  ByteConstArrayView bytes(len + 1, value);
  return DOMString(bytes);
}

String fromCharAndFree(xmlChar* value)
{
  if (!value)
    return DOMString();
  String s(fromChar(value));
  ::xmlFree(value);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodePrv* toNodePrv(const Node& node)
{
  return node._impl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define TNIE throw NotImplementedException(A_FUNCINFO)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LibXml2_DOMImplementation
{
 public:

  static LibXml2_DOMImplementation sDOMImplementation;
  static LibXml2_DOMImplementation* getImplementation()
  {
    return &sDOMImplementation;
  }
};

LibXml2_DOMImplementation LibXml2_DOMImplementation::sDOMImplementation;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Handling of XML reader errors.
 *
 * \note the error handler is managed by libxml2 per thread.
 * It is therefore not necessary to protect it with multi-threading.
 */
class LibXml2_ErrorHandler
{
 public:

  LibXml2_ErrorHandler()
  {
    ::xmlSetStructuredErrorFunc(this, &LibXml2_ErrorHandler::handler);
  }
  ~LibXml2_ErrorHandler()
  {
    ::xmlSetStructuredErrorFunc(nullptr, nullptr);
  }

 public:

  //! Handler to connect to libxml2.
  template <class T>
  static void XMLCDECL handler(void* user_data, T* e)
  {
    if (!e)
      return;
    auto x = reinterpret_cast<LibXml2_ErrorHandler*>(user_data);
    if (!x)
      return;
    x->addError(e);
  }

 public:

  const String& errorMessage() const { return m_error_message; }

 private:

  String m_error_message;

 public:

  void addError(const xmlError* e)
  {
    StringBuilder sb;
    if (e->level == XML_ERR_WARNING)
      sb += "(warning):";
    else if (e->level == XML_ERR_ERROR)
      sb += "(error):";
    else if (e->level == XML_ERR_FATAL)
      sb += "(fatal):";

    sb += " domain ";
    sb += e->domain;
    sb += " errcode ";
    sb += e->code;

    if (e->line > 0) {
      sb += " line ";
      sb += e->line;
      if (e->int2 > 0) {
        sb += " column ";
        sb += e->int2;
      }
    }
    sb += " : ";
    if (e->message)
      sb += e->message;
    else
      sb += "(unknown)";
    m_error_message = m_error_message + sb.toString();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ILibXml2_Reader;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class encapsulating the XML document parser.
 */
class LibXml2_Parser
{
 public:

  LibXml2_Parser(const String& file_name, ITraceMng* trace)
  : m_file_name(file_name)
  , m_trace(trace)
  {
    // Les versions après 2015 de LibXml2 limitent à 10Mo la longueur de chaque noeud XML.
    // Avec cette option, on relâche cette contrainte qui passe à 1Go.
    m_options = XML_PARSE_HUGE;
  }

 public:

  /*!
   * \brief Parses the XML content via the reader \a reader.
   *
   * Returns an XML document that must then be destroyed by
   * calling the delete operator. This document cannot be null.
   *
   * \a reader Associated reader.
   * \a schema_name Name of the file containing the XML Schema to validate. Can be null.
   * \a schema_data In-memory content of the XML Schema. Can be null.
   */
  IXmlDocumentHolder* parse(ILibXml2_Reader* reader, const String& schema_name,
                            ByteConstArrayView schema_data);

 public:

  const String& fileName() const { return m_file_name; }
  int options() const { return m_options; }

 private:

  String m_file_name;
  ITraceMng* m_trace = nullptr;
  int m_options = 0;

 private:

  void _applySchema(::xmlDocPtr doc, LibXml2_ErrorHandler& err_handler,
                    const String& schema_name, ByteConstArrayView schema_data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class encapsulating XML Schema validation of an XML document.
 */
class LibXml2_SchemaValidator
{
 public:

  LibXml2_SchemaValidator(const String& schema_file_name)
  : m_schema_file_name(schema_file_name)
  , m_schema_parser_context(nullptr)
  , m_schema_ptr(nullptr)
  , m_schema_valid_context(nullptr)
  {
  }
  ~LibXml2_SchemaValidator()
  {
    _clearMemory();
  }

 public:

  /*!
   * \brief Validates an XML document.
   *
   * Validates the document \a doc. The schema file name is given
   * by the constructor. If \a schema_data is not null, it is considered
   * to be the content of the XML Schema file.
   *
   * \a doc XML document.
   * \a schema_data In-memory content of the XML Schema. Can be null.
   */
  void validate(::xmlDocPtr doc, ByteConstArrayView schema_data);

 private:

  String m_schema_file_name;
  ::xmlSchemaParserCtxtPtr m_schema_parser_context;
  ::xmlSchemaPtr m_schema_ptr;
  ::xmlSchemaValidCtxtPtr m_schema_valid_context;

 private:

  void _clearMemory()
  {
    if (m_schema_parser_context) {
      ::xmlSchemaFreeParserCtxt(m_schema_parser_context);
      m_schema_parser_context = nullptr;
    }
    if (m_schema_ptr) {
      ::xmlSchemaFree(m_schema_ptr);
      m_schema_ptr = nullptr;
    }
    if (m_schema_valid_context) {
      ::xmlSchemaFreeValidCtxt(m_schema_valid_context);
      m_schema_valid_context = nullptr;
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ILibXml2_Reader
{
 public:

  virtual ~ILibXml2_Reader() {}

 public:

  virtual ::xmlDocPtr read(LibXml2_Parser& context) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LibXml2_MemoryReader
: public ILibXml2_Reader
{
 public:

  LibXml2_MemoryReader(ByteConstSpan buffer)
  : m_buffer(buffer)
  {}

 public:

  ::xmlDocPtr read(LibXml2_Parser& parser) override
  {
    const char* encoding = nullptr;
    int options = parser.options();
    const char* buf_base = reinterpret_cast<const char*>(m_buffer.data());
    // TODO: check if there is a 64-bit reading version
    // that also works on older versions of LibXml2
    // (for RHEL6 support)
    int buf_size = CheckedConvert::toInt32(m_buffer.size());
    while (buf_size > 0 && static_cast<char>(m_buffer[buf_size - 1]) == '\0') {
      buf_size--;
    }
    const String& name = parser.fileName();
    ::xmlParserCtxtPtr ctxt = ::xmlNewParserCtxt();
    ::xmlDocPtr doc = ::xmlCtxtReadMemory(ctxt, buf_base, buf_size,
                                          name.localstr(), encoding, options);
    ::xmlFreeParserCtxt(ctxt);
    return doc;
  }

 private:

  ByteConstSpan m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LibXml2_FileReader
: public ILibXml2_Reader
{
 public:

  LibXml2_FileReader(const String& file_name)
  : m_file_name(file_name)
  {}

 public:

  ::xmlDocPtr read(LibXml2_Parser& parser) override
  {
    const char* encoding = nullptr;
    int options = parser.options();
    const char* file_name = m_file_name.localstr();
    ::xmlParserCtxtPtr ctxt = ::xmlNewParserCtxt();
    ::xmlDocPtr doc = ::xmlCtxtReadFile(ctxt, file_name, encoding, options);
    ::xmlFreeParserCtxt(ctxt);
    return doc;
  }

 private:

  String m_file_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlDocumentHolderLibXml2
: public IXmlDocumentHolder
{
 public:

  XmlDocumentHolderLibXml2()
  : m_document(nullptr)
  , m_document_node(nullptr)
  {}
  ~XmlDocumentHolderLibXml2()
  {
    if (m_document) {
      ::xmlDocPtr doc = impl(m_document);
      ::xmlFreeDoc(doc);
    }
  }
  XmlNode documentNode() override { return XmlNode(nullptr, m_document_node); }
  IXmlDocumentHolder* clone() override { TNIE; }
  void save(ByteArray& bytes) override
  {
    dom::DOMImplementation domimp;
    domimp._save(bytes, m_document, (-1));
  }
  String save() override
  {
    //TODO verify that we always save in UTF8.
    ByteUniqueArray bytes;
    save(bytes);
    String new_s = String::fromUtf8(bytes);
    return new_s;
  }

 public:

  void assignDocument(DocumentPrv* doc)
  {
    m_document = doc;
    m_document_node = (NodePrv*)doc;
  }
  DocumentPrv* _document() const { return m_document; }

 private:

  DocumentPrv* m_document;
  NodePrv* m_document_node;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMImplementation::
DOMImplementation()
: m_p(nullptr)
{
  m_p = cvt(LibXml2_DOMImplementation::getImplementation());
}

DOMImplementation::
DOMImplementation(ImplementationPrv* prv)
: m_p(prv)
{
}

DOMImplementation::
~DOMImplementation()
{
}

void DOMImplementation::
_checkValid() const
{
  if (!m_p)
    arcaneNullPointerError();
}

ImplementationPrv* DOMImplementation::
_impl() const
{
  return m_p;
}

bool DOMImplementation::
hasFeature(const DOMString& feature, const DOMString& version) const
{
  ARCANE_UNUSED(feature);
  ARCANE_UNUSED(version);
  _checkValid();
  TNIE;
  //return impl(m_p)->hasFeature(toChar(feature),toChar(version));
}

DocumentType DOMImplementation::
createDocumentType(const DOMString& qualified_name, const DOMString& public_id,
                   const DOMString& system_id) const
{
  ARCANE_UNUSED(qualified_name);
  ARCANE_UNUSED(public_id);
  ARCANE_UNUSED(system_id);
  _checkValid();
  TNIE;
  //return cvt(impl(m_p)->createDocumentType(toChar(qualified_name),toChar(public_id),
  //                                     toChar(system_id)));
}

DOMImplementation DOMImplementation::
getInterface(const DOMString& feature) const
{
  _checkValid();
  ARCANE_UNUSED(feature);
  throw NotImplementedException(A_FUNCINFO);
}

DOMWriter DOMImplementation::
createDOMWriter() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \todo handle the arguments...
 */
Document DOMImplementation::
createDocument(const DOMString& namespace_uri, const DOMString& qualified_name,
               const DocumentType& doctype) const
{
  if (!namespace_uri.null())
    ARCANE_THROW(NotImplementedException, "non nul namespace-uri");
  if (!qualified_name.null())
    ARCANE_THROW(NotImplementedException, "non nul qualified-name");
  if (!doctype._null())
    ARCANE_THROW(NotImplementedException, "non nul doctype");
  const xmlChar* xml_version = nullptr;
  xmlDocPtr doc = ::xmlNewDoc(xml_version);
  return cvt(doc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DOMImplementation::
_implementationName() const
{
  return "libxml2";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* DOMImplementation::
_newDocument()
{
  Document _doc = createDocument(DOMString(), DOMString(), DocumentType());
  auto xml_doc = new XmlDocumentHolderLibXml2();
  xml_doc->assignDocument(_doc._impl());
  return xml_doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* DOMImplementation::
_load(const String& fname, ITraceMng* msg, const String& schemaname)
{
  return _load(fname, msg, schemaname, ByteConstArrayView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* DOMImplementation::
_load(const String& fname, ITraceMng* trace, const String& schema_name,
      ByteConstArrayView schema_data)
{
  _checkValid();
  LibXml2_FileReader reader(fname);
  LibXml2_Parser parser(fname, trace);
  auto doc_holder = parser.parse(&reader, schema_name, schema_data);
  return doc_holder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* DOMImplementation::
_load(ByteConstSpan buffer, const String& name, ITraceMng* trace)
{
  _checkValid();
  if (buffer.empty())
    return new XmlDocumentHolderLibXml2();

  LibXml2_MemoryReader reader(buffer);
  LibXml2_Parser parser(name, trace);
  auto doc_holder = parser.parse(&reader, String(), ByteConstArrayView());
  return doc_holder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DOMImplementation::
_save(ByteArray& bytes, const Document& document, int indent_level)
{
  ARCANE_UNUSED(indent_level);
  // NOTE: Recent versions of libxml2 (2.9.0) use a new
  // API for buffers via the xmlBufContent() methods (to retrieve
  // the content) and xmlBufUse() (to retrieve the used size).
  // Both these methods allow handling buffers larger
  // than 2GB.
  // However, on RHEL6, the default libxml2 version is too
  // old, so we must support the old methods.
  // The macro 'LIBXML2_NEW_BUFFER' is defined if we can use the
  // new mechanism.
  xmlDocPtr doc = impl(document._impl());
  xmlBufferPtr buf = ::xmlBufferCreate();

  int options = 0;
  if (indent_level > 0)
    options = XML_SAVE_FORMAT;
  xmlSaveCtxtPtr ctx = ::xmlSaveToBuffer(buf, nullptr, options);
  (void)::xmlSaveDoc(ctx, doc);
  (void)::xmlSaveClose(ctx);

  const xmlChar* content = ::xmlBufferContent(buf);
  size_t content_len = ::xmlBufferLength(buf);

  Integer buf_view_size = arcaneCheckArraySize(content_len);
  ByteConstArrayView buf_view(buf_view_size, (const Byte*)content);
  bytes.copy(buf_view);
  // TODO: protect the buffer from possible exceptions in bytes.copy().
  ::xmlBufferFree(buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IXmlDocumentHolder* LibXml2_Parser::
parse(ILibXml2_Reader* reader, const String& schema_name,
      ByteConstArrayView schema_data)
{
  std::unique_ptr<XmlDocumentHolderLibXml2> xml_parser(new XmlDocumentHolderLibXml2());

  m_options |= XML_PARSE_DTDLOAD | XML_PARSE_NOENT | XML_PARSE_DTDATTR;
  m_options |= XML_PARSE_XINCLUDE;

  ::xmlDocPtr doc_ptr = nullptr;

  {
    // Note: the error handler is managed by libxml2 per thread.
    // Therefore, it is not necessary to protect it in multi-threading.
    LibXml2_ErrorHandler err_handler;

    doc_ptr = reader->read(*this);

    if (!doc_ptr)
      ARCANE_THROW(XmlException, "Could not parse document '{0}'\n{1}", fileName(),
                   err_handler.errorMessage());

    // Assigns the document to ensure its release in case of an exception.
    xml_parser->assignDocument(cvt(doc_ptr));

    // Performs XInclude replacement. The ::xmlXIncludeProcess()
    // returns the number of substitutions or (-1) in case of an error.
    int nb_xinclude = ::xmlXIncludeProcess(doc_ptr);
    if (nb_xinclude == (-1))
      ARCANE_THROW(XmlException, "Could not parse xinclude for document '{0}'\n{1}", fileName(),
                   err_handler.errorMessage());

    // Even if the reading is correct, there may be
    // warning messages to display.
    String err_message = err_handler.errorMessage();
    if (m_trace && !err_message.null())
      m_trace->info() << "Info parsing document " << fileName() << " : " << err_message;
  }

  // Uncomment to debug if you want to display the read document.
  //::xmlDocDump(stdout,doc_ptr);

  {
    LibXml2_SchemaValidator validator(schema_name);
    validator.validate(doc_ptr, schema_data);
  }

  return xml_parser.release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LibXml2_SchemaValidator::
validate(::xmlDocPtr doc_ptr, ByteConstArrayView schema_data)
{
  // We must provide schema_name or schema_data or both.
  // If 'schema_data' is provided, we use it and consider that
  // 'schema_name' is the corresponding file name.
  if (m_schema_file_name.null() && schema_data.empty())
    return;
  _clearMemory();
  LibXml2_ErrorHandler err_handler;
  if (!schema_data.empty()) {
    auto base_ptr = reinterpret_cast<const char*>(schema_data.data());
    m_schema_parser_context = ::xmlSchemaNewMemParserCtxt(base_ptr, schema_data.size());
  }
  else
    m_schema_parser_context = ::xmlSchemaNewParserCtxt(m_schema_file_name.localstr());
  if (!m_schema_parser_context)
    ARCANE_THROW(XmlException, "Can not create schema parser");
  m_schema_ptr = xmlSchemaParse(m_schema_parser_context);
  if (!m_schema_ptr)
    ARCANE_THROW(XmlException, "Can not read schema file '{0}'\n{1}", m_schema_file_name,
                 err_handler.errorMessage());
  m_schema_valid_context = xmlSchemaNewValidCtxt(m_schema_ptr);
  if (!m_schema_valid_context)
    ARCANE_THROW(XmlException, "Can not create valid context for file '{0}'\n{1}",
                 m_schema_file_name, err_handler.errorMessage());
  xmlSchemaSetValidOptions(m_schema_valid_context, XML_SCHEMA_VAL_VC_I_CREATE);
  int result = xmlSchemaValidateDoc(m_schema_valid_context, doc_ptr);
  if (result != 0)
    ARCANE_THROW(XmlException, "Can not validate file '{0}'\n{1}",
                 m_schema_file_name, err_handler.errorMessage());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Node::
Node()
: m_p(0)
{}
Node::
Node(NodePrv* p)
: m_p(p)
{
}
Node::
Node(const Node& from)
: m_p(from.m_p)
{
}
const Node& Node::
operator=(const Node& from)
{
  _assign(from);
  return (*this);
}
Node::
~Node()
{}
bool Node::
_null() const
{
  return !m_p;
}
void Node::
_checkValid() const
{
  if (_null())
    arcaneNullPointerError();
}
NodePrv* Node::
_impl() const
{
  return m_p;
}

UShort Node::
nodeType() const
{
  _checkValid();
  // The type returned by libxml2 (of type xmlElementType) corresponds to that of the DOM
  // with the same values. Therefore, we just need to cast the value.
  return (UShort)(impl(m_p)->type);
}
Node Node::
firstChild() const
{
  _checkValid();
  ::xmlNodePtr first_children = impl(m_p)->children;
  return cvt(first_children);
}
Node Node::
lastChild() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(m_p)->getLastChild());
}
Node Node::
previousSibling() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(m_p)->getPreviousSibling());
}
Node Node::
nextSibling() const
{
  _checkValid();
  xmlNodePtr next_node = impl(m_p)->next;
  return cvt(next_node);
}
Node Node::
parentNode() const
{
  _checkValid();
  return cvt(impl(m_p)->parent);
}
NodeList Node::
childNodes() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(m_p)->getChildNodes());
}
DOMString Node::
nodeName() const
{
  _checkValid();
  //if (impl(m_p)->type==XML_ELEMENT_NODE)
  //std::cerr << "NODE_NAME =" << impl(m_p)->name << " content=" << impl(m_p)->content << "\n";
  return fromChar(impl(m_p)->name);
}
NamedNodeMap Node::
attributes() const
{
  _checkValid();
  ::xmlNodePtr xelement = impl(m_p);
  xmlAttrPtr p = xelement->properties;
  return cvt((xmlNamedNodeMapPtr)(p));
}
Document Node::
ownerDocument() const
{
  _checkValid();
  ::xmlNodePtr node = impl(m_p);
  return cvt(node->doc);
}
DOMString Node::
nodeValue() const
{
  _checkValid();
  xmlChar* content = ::xmlNodeGetContent(impl(m_p));
  return fromCharAndFree(content);
}
void Node::
nodeValue(const DOMString& str) const
{
  ARCANE_UNUSED(str);
  _checkValid();
  TNIE;
  //impl(m_p)->setNodeValue(toChar(str));
}
void Node::
_assign(const Node& node)
{
  m_p = node.m_p;
}
Node Node::
insertBefore(const Node& new_child, const Node& ref_child) const
{
  ARCANE_UNUSED(new_child);
  ARCANE_UNUSED(ref_child);
  _checkValid();
  TNIE;
  //return cvt(impl(m_p)->insertBefore(impl(new_child._impl()),impl(ref_child._impl())));
}
Node Node::
replaceChild(const Node& new_child, const Node& old_child) const
{
  ARCANE_UNUSED(new_child);
  ARCANE_UNUSED(old_child);
  _checkValid();
  TNIE;
  //return cvt(impl(m_p)->replaceChild(impl(new_child._impl()),impl(old_child._impl())));
}
Node Node::
removeChild(const Node& old_child) const
{
  _checkValid();
  ::xmlNodePtr xchild = impl(old_child._impl());
  // Attention, the node must then be destroyed via Node::releaseNode())
  // to free the memory.
  ::xmlUnlinkNode(xchild);
  return cvt(xchild);
}
Node Node::
appendChild(const Node& new_child) const
{
  _checkValid();
  return cvt(::xmlAddChild(impl(m_p), impl(new_child._impl())));
}
bool Node::
hasChildNodes() const
{
  _checkValid();
  TNIE;
  //return impl(m_p)->hasChildNodes();
}
Node Node::
cloneNode(bool deep) const
{
  _checkValid();
  ARCANE_UNUSED(deep);
  TNIE;
  //return cvt(impl(m_p)->cloneNode(deep));
}
DOMString Node::
prefix() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(m_p)->getPrefix());
}
void Node::
prefix(const DOMString& new_prefix) const
{
  _checkValid();
  ARCANE_UNUSED(new_prefix);
  TNIE;
  //impl(m_p)->setPrefix(toChar(new_prefix));
}
void Node::
normalize() const
{
  _checkValid();
  TNIE;
  //impl(m_p)->normalize();
}
bool Node::
isSupported(const DOMString& feature, const DOMString& version) const
{
  ARCANE_UNUSED(feature);
  ARCANE_UNUSED(version);
  _checkValid();
  TNIE;
  //return impl(m_p)->isSupported(toChar(feature),toChar(version));
}
DOMString Node::
namespaceURI() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(m_p)->getNamespaceURI());
}
DOMString Node::
localName() const
{
  _checkValid();
  return fromChar(impl(m_p)->name);
}
DOMString Node::
baseURI() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(m_p)->getBaseURI());
}
DOMString Node::
textContent() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(m_p)->getTextContent());
}
void Node::
textContent(const DOMString& value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  TNIE;
  //impl(m_p)->setTextContent(toChar(value));
}
bool Node::
isSameNode(const Node& node) const
{
  _checkValid();
  ARCANE_UNUSED(node);
  TNIE;
  //return impl(m_p)->isSameNode(impl(node.m_p));
}
bool Node::
isEqualNode(const Node& other) const
{
  _checkValid();
  ARCANE_UNUSED(other);
  TNIE;
  //return impl(m_p)->isEqualNode(impl(other._impl()));
}
bool Node::
isDefaultNamespace(const DOMString& namespace_uri) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  TNIE;
  //return impl(m_p)->isDefaultNamespace(toChar(namespace_uri));
}
DOMString Node::
lookupNamespaceURI(const DOMString& prefix) const
{
  _checkValid();
  ARCANE_UNUSED(prefix);
  TNIE;
  //return fromChar(impl(m_p)->lookupNamespaceURI(toChar(prefix)));
}
DOMObject Node::
setUserData(const DOMString& key, const DOMObject& data,
            const UserDataHandler& handler) const
{
  _checkValid();
  ARCANE_UNUSED(key);
  ARCANE_UNUSED(data);
  ARCANE_UNUSED(handler);
  throw NotImplementedException(A_FUNCINFO);
}
DOMObject Node::
getUserData(const DOMString& key) const
{
  _checkValid();
  ARCANE_UNUSED(key);
  TNIE;
  //return impl(m_p)->getUserData(toChar(key));
}
void Node::
releaseNode()
{
  ::xmlNodePtr xnode = impl(m_p);
  if (xnode)
    ::xmlFreeNode(xnode);
}
bool operator==(const Node& n1, const Node& n2)
{
  return impl(n1.m_p) == impl(n2.m_p);
}
bool operator!=(const Node& n1, const Node& n2)
{
  return !operator==(n1, n2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CharacterData::
CharacterData(CharacterDataPrv* v)
: Node(cvt((xmlNodePtr)impl(v)))
{}

CharacterData::
CharacterData()
: Node()
{}

CharacterData::
CharacterData(const CharacterData& node)
: Node(node)
{
}

CharacterData::
CharacterData(const Node& node)
: Node()
{
  TNIE;
  ARCANE_UNUSED(node);
  //NodePrv* ni= node._impl();
  //if (ni && impl(ni)->getNodeType()==CDATA_SECTION_NODE)
  //_assign(node);
}

CharacterDataPrv* CharacterData::
_impl() const
{
  TNIE;
  //return cvt((xmlCharacterDataPtr)impl(m_p));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Text::
Text(TextPrv* v)
: CharacterData(cvt((xmlCharacterDataPtr)impl(v)))
{}
Text::
Text()
: CharacterData()
{}
Text::
Text(const Text& node)
: CharacterData(node)
{
}
Text::
Text(const Node& node)
: CharacterData()
{
  TNIE;
  ARCANE_UNUSED(node);
  //NodePrv* ni= node._impl();
  //if (ni && impl(ni)->getNodeType()==TEXT_NODE)
  //_assign(node);
}
TextPrv* Text::
_impl() const
{
  return cvt((xmlTextPtr)impl(m_p));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Document::
Document()
{}
Document::
Document(DocumentPrv* p)
: Node(cvt((xmlNodePtr)impl(p)))
{}
Document::
Document(const Node& node)
: Node()
{
  NodePrv* ni = node._impl();
  if (ni && impl(ni)->type == XML_DOCUMENT_NODE)
    _assign(node);
}
DocumentPrv* Document::
_impl() const
{
  return cvt((xmlDocPtr)impl(m_p));
}
DocumentType Document::
doctype() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(_impl())->getDoctype());
}
DOMImplementation Document::
implementation() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(_impl())->getImplementation());
}
Element Document::
documentElement() const
{
  _checkValid();
  xmlDocPtr xdoc = impl(_impl());
  xmlNodePtr xnode = ::xmlDocGetRootElement(xdoc);
  return cvt((xmlElementPtr)xnode);
}
Element Document::
createElement(const DOMString& name) const
{
  _checkValid();
  xmlDocPtr xdoc = impl(_impl());
  xmlNsPtr nspace = nullptr;
  xmlChar* content = nullptr;
  xmlNodePtr xnode = ::xmlNewDocNode(xdoc, nspace, toChar(name), content);
  return cvt((xmlElementPtr)xnode);
}
DocumentFragment Document::
createDocumentFragment() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(_impl())->createDocumentFragment());
}
Text Document::
createTextNode(const DOMString& data) const
{
  _checkValid();
  xmlDocPtr xdoc = impl(_impl());
  return cvt((xmlTextPtr)::xmlNewDocText(xdoc, toChar(data)));
}
Comment Document::
createComment(const DOMString& data) const
{
  _checkValid();
  TNIE;
  ARCANE_UNUSED(data);
  //return cvt(impl(_impl())->createComment(toChar(data)));
}
CDATASection Document::
createCDATASection(const DOMString& data) const
{
  _checkValid();
  TNIE;
  ARCANE_UNUSED(data);
  //return cvt(impl(_impl())->createCDATASection(toChar(data)));
}
ProcessingInstruction Document::
createProcessingInstruction(const DOMString& target,
                            const DOMString& data) const
{
  _checkValid();
  ARCANE_UNUSED(target);
  ARCANE_UNUSED(data);
  TNIE;
  //  return cvt(impl(_impl())->createProcessingInstruction(toChar(target),toChar(data)));
}

Attr Document::
createAttribute(const DOMString& name) const
{
  _checkValid();
  ARCANE_UNUSED(name);
  TNIE;
  //return cvt(impl(_impl())->createAttribute(toChar(name)));
}

EntityReference Document::
createEntityReference(const DOMString& name) const
{
  _checkValid();
  ARCANE_UNUSED(name);
  TNIE;
  //return cvt(impl(_impl())->createEntityReference(toChar(name)));
}
NodeList Document::
getElementsByTagName(const DOMString& tagname) const
{
  _checkValid();
  ARCANE_UNUSED(tagname);
  TNIE;
  //return cvt(impl(_impl())->getElementsByTagName(toChar(tagname)));
}
Node Document::
importNode(const Node& imported_node, bool deep) const
{
  _checkValid();
  ARCANE_UNUSED(imported_node);
  ARCANE_UNUSED(deep);
  TNIE;
  //return cvt(impl(_impl())->importNode(impl(toNodePrv(imported_node)),deep));
}
Element Document::
createElementNS(const DOMString& namespace_uri, const DOMString& qualified_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(qualified_name);
  TNIE;
  //return cvt(impl(_impl())->createElementNS(toChar(namespace_uri),toChar(qualified_name)));
}
Attr Document::
createAttributeNS(const DOMString& namespace_uri, const DOMString& qualified_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(qualified_name);
  TNIE;
  //return cvt(impl(_impl())->createAttributeNS(toChar(namespace_uri),toChar(qualified_name)));
}
NodeList Document::
getElementsByTagNameNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return cvt(impl(_impl())->getElementsByTagNameNS(toChar(namespace_uri),toChar(local_name)));
}
Element Document::
getElementById(const DOMString& element_id) const
{
  _checkValid();
  TNIE;
  ARCANE_UNUSED(element_id);
  //return cvt(impl(_impl())->getElementById(toChar(element_id)));
}
DOMString Document::
actualEncoding() const
{
  _checkValid();
  TNIE;
  //throw NotImplementedException(A_FUNCINFO);
}
void Document::
actualEncoding(const DOMString& value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  throw NotImplementedException(A_FUNCINFO);
}

DOMString Document::
encoding() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

void Document::
encoding(const DOMString& value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  throw NotImplementedException(A_FUNCINFO);
}

bool Document::
standalone() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

void Document::
standalone(bool value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  throw NotImplementedException(A_FUNCINFO);
}

bool Document::
strictErrorChecking() const
{
  _checkValid();
  TNIE;
  //return impl(_impl())->getStrictErrorChecking();
}
void Document::
strictErrorChecking(bool value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  TNIE;
  //impl(_impl())->setStrictErrorChecking(value);
}
void Document::
documentURI(const DOMString& document_uri) const
{
  _checkValid();
  ARCANE_UNUSED(document_uri);
  TNIE;
  //impl(_impl())->setDocumentURI(toChar(document_uri));
}
DOMString Document::
documentURI() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getDocumentURI());
}

Node Document::
adoptNode(const Node& source) const
{
  _checkValid();
  TNIE;
  ARCANE_UNUSED(source);
  //return Node(cvt(impl(_impl())->adoptNode(impl(source._impl()))));
}
void Document::
normalizeDocument()
{
  _checkValid();
  TNIE;
  //impl(_impl())->normalizeDocument();
}
Node Document::
renameNode(const Node& node, const DOMString& namespace_uri,
           const DOMString& name)
{
  _checkValid();
  node._checkValid();
  ARCANE_UNUSED(node);
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(name);
  TNIE;
  //return Node(cvt(impl(_impl())->renameNode(impl(node._impl()),toChar(namespace_uri),toChar(name))));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DocumentFragment::
DocumentFragment()
: Node()
{}
DocumentFragment::
DocumentFragment(DocumentFragmentPrv* prv)
: Node(cvt((xmlNodePtr)impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Comment::
Comment()
: CharacterData()
{}
Comment::
Comment(CommentPrv* prv)
: CharacterData((CharacterDataPrv*)prv)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CDATASection::
CDATASection()
: Text()
{}
CDATASection::
CDATASection(CDATASectionPrv* prv)
: Text(cvt((xmlTextPtr)impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntityReference::
EntityReference()
: Node()
{}
EntityReference::
EntityReference(EntityReferencePrv* prv)
: Node(cvt((xmlNodePtr)impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeList::
NodeList()
: m_p(0)
{}
NodeList::
NodeList(NodeListPrv* prv)
: m_p(prv)
{}

void NodeList::
_checkValid() const
{
  if (!m_p)
    arcaneNullPointerError();
}
Node NodeList::
item(ULong index) const
{
  _checkValid();
  ARCANE_UNUSED(index);
  TNIE;
  //return cvt(impl(m_p)->item(index));
}
ULong NodeList::
length() const
{
  _checkValid();
  TNIE;
  //return impl(m_p)->getLength();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMString CharacterData::
data() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getData());
}
void CharacterData::
data(const DOMString& value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  TNIE;
  //impl(_impl())->setData(toChar(value));
}
ULong CharacterData::
length() const
{
  _checkValid();
  TNIE;
  //return impl(_impl())->getLength();
}
DOMString CharacterData::
substringData(ULong offset, ULong count) const
{
  _checkValid();
  ARCANE_UNUSED(offset);
  ARCANE_UNUSED(count);
  TNIE;
  //return fromChar(impl(_impl())->substringData(offset,count));
}
void CharacterData::
appendData(const DOMString& arg) const
{
  _checkValid();
  ARCANE_UNUSED(arg);
  TNIE;
  //impl(_impl())->appendData(toChar(arg));
}
void CharacterData::
insertData(ULong offset, const DOMString& arg) const
{
  _checkValid();
  ARCANE_UNUSED(offset);
  ARCANE_UNUSED(arg);
  TNIE;
  //impl(_impl())->insertData(offset,toChar(arg));
}
void CharacterData::
deleteData(ULong offset, ULong count) const
{
  _checkValid();
  ARCANE_UNUSED(offset);
  ARCANE_UNUSED(count);
  TNIE;
  //impl(_impl())->deleteData(offset,count);
}
void CharacterData::
replaceData(ULong offset, ULong count, const DOMString& arg) const
{
  _checkValid();
  TNIE;
  ARCANE_UNUSED(offset);
  ARCANE_UNUSED(count);
  ARCANE_UNUSED(arg);
  //impl(_impl())->replaceData(offset,count,toChar(arg));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Attr::
Attr(AttrPrv* p)
: Node(cvt((xmlNodePtr)impl(p)))
{
}

Attr::
Attr(const Node& node)
: Node()
{
  ARCANE_UNUSED(node);
  TNIE;
  // NodePrv* ni= node._impl();
  //if (ni && impl(ni)->getNodeType()==ATTRIBUTE_NODE)
  //_assign(node);
}

AttrPrv* Attr::
_impl() const
{
  return cvt((xmlAttrPtr)impl(m_p));
}
DOMString Attr::
name() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getName());
}
bool Attr::
specified() const
{
  _checkValid();
  TNIE;
  //return impl(_impl())->getSpecified();
}
DOMString Attr::
value() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getValue());
}
void Attr::
value(const DOMString& str) const
{
  _checkValid();
  ARCANE_UNUSED(str);
  TNIE;
  //impl(_impl())->setValue(toChar(str));
}
Element Attr::
ownerElement() const
{
  _checkValid();
  TNIE;
  //return cvt(impl(_impl())->getOwnerElement());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Element::
Element()
: Node()
{}
Element::
Element(ElementPrv* p)
: Node(cvt((xmlNodePtr)impl(p)))
{}
Element::
Element(const Node& node)
: Node()
{
  NodePrv* ni = node._impl();
  if (ni && impl(ni)->type == XML_ELEMENT_NODE)
    _assign(node);
}
Element::
Element(const Element& node)
: Node(node)
{}
ElementPrv* Element::
_impl() const
{
  return cvt((xmlElementPtr)impl(m_p));
}

DOMString Element::
tagName() const
{
  _checkValid();
  TNIE;
  //  return fromChar(impl(_impl())->getTagName());
}
String Element::
getAttribute(const DOMString& name) const
{
  _checkValid();
  Attr a = getAttributeNode(name);
  if (a._null())
    return String();
  ::xmlChar* prop = ::xmlGetProp(impl(m_p), toChar(name));
  String s = fromChar(prop);
  ::xmlFree(prop);
  return s;
}
void Element::
setAttribute(const DOMString& name, const DOMString& value) const
{
  _checkValid();
  ::xmlSetProp(impl(m_p), toChar(name), toChar(value));
}
void Element::
removeAttribute(const DOMString& name) const
{
  _checkValid();
  ARCANE_UNUSED(name);
  TNIE;
  //impl(_impl())->removeAttribute(toChar(name));
}
Attr Element::
getAttributeNode(const DOMString& name) const
{
  _checkValid();
  xmlElementPtr elem_ptr = (xmlElementPtr)(impl(_impl()));
  return cvt(::xmlHasProp((xmlNodePtr)elem_ptr, toChar(name)));
}
Attr Element::
setAttributeNode(const Attr& new_attr) const
{
  _checkValid();
  ARCANE_UNUSED(new_attr);
  TNIE;
  //return cvt(impl(_impl())->setAttributeNode(impl(new_attr._impl())));
}
Attr Element::
removeAttributeNode(const Attr& old_attr) const
{
  _checkValid();
  ARCANE_UNUSED(old_attr);
  TNIE;
  //return cvt(impl(_impl())->removeAttributeNode(impl(old_attr._impl())));
}
NodeList Element::
getElementsByTagName(const DOMString& name) const
{
  _checkValid();
  ARCANE_UNUSED(name);
  TNIE;
  //return NodeList(cvt(impl(_impl())->getElementsByTagName(toChar(name))));
}
DOMString Element::
getAttributeNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return fromChar(impl(_impl())->getAttributeNS(toChar(namespace_uri),toChar(local_name)));
}
void Element::
setAttributeNS(const DOMString& namespace_uri, const DOMString& local_name,
               const DOMString& value) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  ARCANE_UNUSED(value);
  TNIE;
  //impl(_impl())->setAttributeNS(toChar(namespace_uri),toChar(local_name),toChar(value));
}
void Element::
removeAttributeNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //impl(_impl())->removeAttributeNS(toChar(namespace_uri),toChar(local_name));
}
Attr Element::
getAttributeNodeNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return Attr(cvt(impl(_impl())->getAttributeNodeNS(toChar(namespace_uri),toChar(local_name))));
}
Attr Element::
setAttributeNodeNS(const Attr& new_attr) const
{
  _checkValid();
  new_attr._checkValid();
  TNIE;
  //return Attr(cvt(impl(_impl())->setAttributeNodeNS(impl(new_attr._impl()))));
}
NodeList Element::
getElementsByTagNameNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return NodeList(cvt(impl(_impl())->getElementsByTagNameNS(toChar(namespace_uri),toChar(local_name))));
}
bool Element::
hasAttribute(const DOMString& name) const
{
  _checkValid();
  ARCANE_UNUSED(name);
  TNIE;
  //return impl(_impl())->hasAttribute(toChar(name));
}
bool Element::
hasAttributeNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  _checkValid();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return impl(_impl())->hasAttributeNS(toChar(namespace_uri),toChar(local_name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Text Text::
splitText(ULong offset) const
{
  _checkValid();
  ARCANE_UNUSED(offset);
  TNIE;
  //return Text(cvt(impl(_impl())->splitText(offset)));
}
DOMString Text::
wholeText() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getWholeText());
}
Text Text::
replaceWholeText(const DOMString& content) const
{
  _checkValid();
  ARCANE_UNUSED(content);
  TNIE;
  //return Text(cvt(impl(_impl())->replaceWholeText(toChar(content))));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DocumentType::
DocumentType()
: Node()
{}
DocumentType::
DocumentType(DocumentTypePrv* prv)
: Node(cvt((xmlNodePtr)impl(prv)))
{}
DocumentTypePrv* DocumentType::
_impl() const
{
  return cvt((xmlDocTypePtr)impl(m_p));
}
DOMString DocumentType::
name() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getName());
}
NamedNodeMap DocumentType::
entities() const
{
  _checkValid();
  TNIE;
  //return NamedNodeMap(cvt(impl(_impl())->getEntities()));
}
NamedNodeMap DocumentType::
notations() const
{
  _checkValid();
  TNIE;
  //return NamedNodeMap(cvt(impl(_impl())->getNotations()));
}
DOMString DocumentType::
publicId() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getPublicId());
}
DOMString DocumentType::
systemId() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getSystemId());
}
DOMString DocumentType::
internalSubset() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getInternalSubset());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NotationPrv* Notation::
_impl() const
{
  return cvt((xmlNotationPtr)impl(m_p));
}

DOMString Notation::
publicId() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getPublicId());
}

DOMString Notation::
systemId() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getSystemId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EntityPrv* Entity::
_impl() const
{
  return cvt((xmlEntityPtr)impl(m_p));
}

DOMString Entity::
publicId() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getPublicId());
}
DOMString Entity::
systemId() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getSystemId());
}
DOMString Entity::
notationName() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getNotationName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProcessingInstruction::
ProcessingInstruction()
: Node()
{}
ProcessingInstruction::
ProcessingInstruction(ProcessingInstructionPrv* prv)
: Node(cvt((xmlNodePtr)impl(prv)))
{}
ProcessingInstructionPrv* ProcessingInstruction::
_impl() const
{
  return cvt((xmlProcessingInstructionPtr)impl(m_p));
}

DOMString ProcessingInstruction::
target() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getTarget());
}
DOMString ProcessingInstruction::
data() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getData());
}
void ProcessingInstruction::
data(const DOMString& value) const
{
  _checkValid();
  ARCANE_UNUSED(value);
  TNIE;
  //impl(_impl())->setData(toChar(value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NamedNodeMap::
NamedNodeMap()
: m_p(nullptr)
{}
NamedNodeMap::
NamedNodeMap(NamedNodeMapPrv* p)
: m_p(p)
{}
NamedNodeMap::
NamedNodeMap(const NamedNodeMap& from)
: m_p(from.m_p)
{}
const NamedNodeMap& NamedNodeMap::
operator=(const NamedNodeMap& from)
{
  m_p = from.m_p;
  return (*this);
}
NamedNodeMap::
~NamedNodeMap()
{}
NamedNodeMapPrv* NamedNodeMap::
_impl() const
{
  return m_p;
}
bool NamedNodeMap::
_null() const
{
  return !m_p;
}
ULong NamedNodeMap::
length() const
{
  if (_null())
    return 0;
  ULong n = 0;
  ::xmlNodePtr xnode = (::xmlNodePtr)impl(m_p);
  while (xnode) {
    ++n;
    xnode = xnode->next;
  }
  return n;
}
Node NamedNodeMap::
getNamedItem(const DOMString& name) const
{
  // NOTE: Only works for attributes. For other types (entities, ...)
  // another version would have to be made. In principle, if we cast 'xmlAttr' to
  // 'xmlNode' and only use the basic fields (name, next), it should
  // work for everything.
  // NOTE: For attributes, the name can contain a namespace prefix (for
  // example xml:lang). Therefore, we must search via the prefix + local name
  // and not just the local name.
  if (_null())
    return Node();
  ::xmlAttrPtr xattrlist = (::xmlAttrPtr)impl(m_p);
  const ::xmlChar* aname = toChar(name);
  ::xmlAttrPtr current = xattrlist;
  while (current) {
    if (current->type == XML_ATTRIBUTE_NODE && current->ns) {
      std::string full_name = (const char*)(current->ns->prefix);
      full_name += ":";
      full_name += (const char*)(current->name);
      if (xmlStrEqual(aname, (const xmlChar*)full_name.c_str()) == 1) {
        return cvt((::xmlNodePtr)current);
      }
    }
    if (xmlStrEqual(aname, current->name) == 1) {
      return cvt((::xmlNodePtr)current);
    }
    current = current->next;
  }
  return Node();
}
Node NamedNodeMap::
setNamedItem(const Node& arg) const
{
  if (_null() || arg._null())
    return Node();
  TNIE;
  //return Node(cvt(impl(_impl())->setNamedItem(impl(arg._impl()))));
}
Node NamedNodeMap::
removeNamedItem(const DOMString& name) const
{
  if (_null())
    return Node();
  ARCANE_UNUSED(name);
  TNIE;
  //return Node(cvt(impl(_impl())->removeNamedItem(toChar(name))));
}
Node NamedNodeMap::
item(ULong index) const
{
  if (_null())
    return Node();
  ULong n = 0;
  ::xmlNodePtr xnode = (::xmlNodePtr)impl(m_p);
  while (xnode) {
    if (n == index)
      return Node(cvt(xnode));
    ++n;
    xnode = xnode->next;
  }
  return Node();
}
Node NamedNodeMap::
getNamedItemNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  if (_null())
    return Node();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return Node(cvt(impl(_impl())->getNamedItemNS(toChar(namespace_uri),toChar(local_name))));
}
Node NamedNodeMap::
setNamedItemNS(const Node& arg) const
{
  if (_null())
    return Node();
  if (arg._null())
    return Node();
  TNIE;
  //return Node(cvt(impl(_impl())->setNamedItemNS(impl(arg._impl()))));
}
Node NamedNodeMap::
removeNamedItemNS(const DOMString& namespace_uri, const DOMString& local_name) const
{
  if (_null())
    return Node();
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(local_name);
  TNIE;
  //return Node(cvt(impl(_impl())->removeNamedItemNS(toChar(namespace_uri),toChar(local_name))));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMImplementation DOMImplementationSource::
getDOMImplementation(const DOMString& features) const
{
  ARCANE_UNUSED(features);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UserDataHandler::
handle(UShort operation, const DOMString& key, const DOMObject& data,
       const Node& src, const Node& dest) const
{
  ARCANE_UNUSED(operation);
  ARCANE_UNUSED(key);
  ARCANE_UNUSED(data);
  ARCANE_UNUSED(src);
  ARCANE_UNUSED(dest);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMWriter::
DOMWriter()
: m_p(0)
{}
DOMWriter::
DOMWriter(DOMWriterPrv* p)
: m_p(p)
{
}
DOMWriter::
DOMWriter(const DOMWriter& from)
: m_p(from.m_p)
{
}
const DOMWriter& DOMWriter::
operator=(const DOMWriter& from)
{
  m_p = from.m_p;
  return (*this);
}
DOMWriter::
~DOMWriter()
{}
bool DOMWriter::
_null() const
{
  return !m_p;
}
void DOMWriter::
_checkValid() const
{
  if (_null())
    arcaneNullPointerError();
}
DOMWriterPrv* DOMWriter::
_impl() const
{
  return m_p;
}

ByteUniqueArray DOMWriter::
writeNode(const Node& node) const
{
  ARCANE_UNUSED(node);
  throw NotImplementedException(A_FUNCINFO);
}

void DOMWriter::
encoding(const String& encoding)
{
  _checkValid();
  ARCANE_UNUSED(encoding);
  throw NotImplementedException(A_FUNCINFO);
}

String DOMWriter::
encoding() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMError::
DOMError()
: m_p(0)
{}
DOMError::
DOMError(DOMErrorPrv* p)
: m_p(p)
{
}
DOMError::
DOMError(const DOMError& from)
: m_p(from.m_p)
{
}
const DOMError& DOMError::
operator=(const DOMError& from)
{
  m_p = from.m_p;
  return (*this);
}
DOMError::
~DOMError()
{}
bool DOMError::
_null() const
{
  return !m_p;
}
void DOMError::
_checkValid() const
{
  if (_null())
    arcaneNullPointerError();
}
DOMErrorPrv* DOMError::
_impl() const
{
  return m_p;
}

UShort DOMError::
severity() const
{
  _checkValid();
  TNIE;
  //return impl(_impl())->getSeverity();
}
DOMString DOMError::
message() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getMessage());
}
DOMObject DOMError::
relatedException() const
{
  _checkValid();
  TNIE;
  //return DOMObject(impl(_impl())->getRelatedException());
}
DOMLocator DOMError::
location() const
{
  _checkValid();
  TNIE;
  //return DOMLocator(cvt(impl(_impl())->getLocation()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DOMErrorHandler::
handleError(const DOMError& error) const
{
  ARCANE_UNUSED(error);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMLocator::
DOMLocator()
: m_p(0)
{}
DOMLocator::
DOMLocator(DOMLocatorPrv* p)
: m_p(p)
{
}
DOMLocator::
DOMLocator(const DOMLocator& from)
: m_p(from.m_p)
{
}
const DOMLocator& DOMLocator::
operator=(const DOMLocator& from)
{
  m_p = from.m_p;
  return (*this);
}
DOMLocator::
~DOMLocator()
{}
bool DOMLocator::
_null() const
{
  return !m_p;
}
void DOMLocator::
_checkValid() const
{
  if (_null())
    arcaneNullPointerError();
}
DOMLocatorPrv* DOMLocator::
_impl() const
{
  return m_p;
}

long DOMLocator::
lineNumber() const
{
  _checkValid();
  TNIE;
  //return impl(_impl())->getLineNumber();
}

long DOMLocator::
columnNumber() const
{
  _checkValid();
  TNIE;
  //return impl(_impl())->getColumnNumber();
}

DOMString DOMLocator::
uri() const
{
  _checkValid();
  TNIE;
  //return fromChar(impl(_impl())->getURI());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XPathExpression XPathEvaluator::
createExpression(const DOMString& expression,
                 const XPathNSResolver& resolver) const
{
  ARCANE_UNUSED(expression);
  ARCANE_UNUSED(resolver);
  throw NotImplementedException(A_FUNCINFO);
}
XPathResult XPathEvaluator::
createResult() const
{
  throw NotImplementedException(A_FUNCINFO);
}
XPathNSResolver XPathEvaluator::
createNSResolver(const Node& node_resolver) const
{
  ARCANE_UNUSED(node_resolver);
  throw NotImplementedException(A_FUNCINFO);
}
XPathResult XPathEvaluator::
evaluate(const DOMString& expression, const Node& context_node,
         const XPathNSResolver& resolver, UShort type,
         const XPathResult& result) const
{
  ARCANE_UNUSED(expression);
  ARCANE_UNUSED(context_node);
  ARCANE_UNUSED(resolver);
  ARCANE_UNUSED(type);
  ARCANE_UNUSED(result);
  throw NotImplementedException(A_FUNCINFO);
}
XPathResult XPathEvaluator::
evaluateExpression(const XPathExpression& expression,
                   const Node& context_node, UShort type,
                   const XPathResult& result) const
{
  ARCANE_UNUSED(expression);
  ARCANE_UNUSED(context_node);
  ARCANE_UNUSED(type);
  ARCANE_UNUSED(result);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMString XPathNSResolver::
lookupNamespaceURI(const DOMString& prefix) const
{
  ARCANE_UNUSED(prefix);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UShort XPathResult::
resultType() const
{
  throw NotImplementedException(A_FUNCINFO);
}
double XPathResult::
numberValue() const
{
  throw NotImplementedException(A_FUNCINFO);
}
DOMString XPathResult::
stringValue() const
{
  throw NotImplementedException(A_FUNCINFO);
}
bool XPathResult::
booleanValue() const
{
  throw NotImplementedException(A_FUNCINFO);
}
Node XPathResult::
singleNodeValue() const
{
  throw NotImplementedException(A_FUNCINFO);
}
XPathSetIterator XPathResult::
getSetIterator(bool ordered) const
{
  ARCANE_UNUSED(ordered);
  throw NotImplementedException(A_FUNCINFO);
}
XPathSetSnapshot XPathResult::
getSetSnapshot(bool ordered) const
{
  ARCANE_UNUSED(ordered);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Node XPathSetIterator::
nextNode() const
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ULong XPathSetSnapshot::
length() const
{
  throw NotImplementedException(A_FUNCINFO);
}
Node XPathSetSnapshot::
item(ULong index) const
{
  ARCANE_UNUSED(index);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Element XPathNamespace::
ownerElement() const
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DOMImplementation::
initialize()
{
  // Explicitly calls xmlInitParser(). This is not theoretically
  // indispensable, but this method can generate floating point exceptions
  // because at one point there is an explicit call to a division by zero to
  // generate a Nan (in xmlXPathInit()). Since DOMImplementation::initialize()
  // is called before enabling floating point exceptions, the call to parser
  // initialization must be made explicitly here.
  ::xmlInitParser();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DOMImplementation::
terminate()
{
  // Again, this is not indispensable, but it allows global resources to be freed
  // and thus avoids potential memory leaks (which bothers tools like valgrind).
  ::xmlCleanupParser();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::dom

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
