// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

// ***********    ATTENTION     ***********

// ATTENTION: CE FICHIER N'EST PLUS UTILISÉ MAIS CONSERVÉ POUR REGARDER
// L'IMPLÉMENTATION PRÉCÉDENTE

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/StringImpl.h"
#include "arcane/utils/String.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/Dom.h"
#include "arcane/DomUtils.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNode.h"

#include <cassert>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>
#define _AMD64_


#include <list>
#include <map>
#include <string>
#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#include <assert.h>
#include <typeinfo>
#if !defined(_MSC_VER)
#include <sys/time.h>
#endif
#include "arcane/utils/Exception.h"
#include "arcane/utils/String.h"
#include <libxml/tree.h>
#include <time.h>

#ifdef ARCANE_OS_WIN32
#define USE_WCHAR
#endif

#ifdef USE_WCHAR
namespace // anonymous
{
typedef wchar_t XMLCh; // Problem wchar_t is compiler dependant...In Xerces XMLCh typedef was generated. And
                       // no wstring were used. Here massively used...
typedef unsigned char XMLByte;
} // namespace
#else /* USE_WCHAR */
namespace // anonymous
{
typedef char16_t XMLCh;
typedef unsigned char XMLByte;
} // namespace
#endif /* USE_WCHAR */

ARCANE_BEGIN_NAMESPACE
ARCANE_BEGIN_NAMESPACE_DOM
// ---------------------------------------------------------------------------
//  Constants for the Unicode characters of interest to us in an XML parser
//  We don't put these inside the class because then they could not be const
//  inline values, which would have significant performance ramifications.
//
//  We cannot use a namespace because of the requirement to support old
//  compilers.
// ---------------------------------------------------------------------------
const XMLCh chNull = 0x00;
const XMLCh chHTab = 0x09;
const XMLCh chLF = 0x0A;
const XMLCh chVTab = 0x0B;
const XMLCh chFF = 0x0C;
const XMLCh chCR = 0x0D;
const XMLCh chAmpersand = 0x26;
const XMLCh chAsterisk = 0x2A;
const XMLCh chAt = 0x40;
const XMLCh chBackSlash = 0x5C;
const XMLCh chBang = 0x21;
const XMLCh chCaret = 0x5E;
const XMLCh chCloseAngle = 0x3E;
const XMLCh chCloseCurly = 0x7D;
const XMLCh chCloseParen = 0x29;
const XMLCh chCloseSquare = 0x5D;
const XMLCh chColon = 0x3A;
const XMLCh chComma = 0x2C;
const XMLCh chDash = 0x2D;
const XMLCh chDollarSign = 0x24;
const XMLCh chDoubleQuote = 0x22;
const XMLCh chEqual = 0x3D;
const XMLCh chForwardSlash = 0x2F;
const XMLCh chGrave = 0x60;
const XMLCh chNEL = 0x85;
const XMLCh chOpenAngle = 0x3C;
const XMLCh chOpenCurly = 0x7B;
const XMLCh chOpenParen = 0x28;
const XMLCh chOpenSquare = 0x5B;
const XMLCh chPercent = 0x25;
const XMLCh chPeriod = 0x2E;
const XMLCh chPipe = 0x7C;
const XMLCh chPlus = 0x2B;
const XMLCh chPound = 0x23;
const XMLCh chQuestion = 0x3F;
const XMLCh chSingleQuote = 0x27;
const XMLCh chSpace = 0x20;
const XMLCh chSemiColon = 0x3B;
const XMLCh chTilde = 0x7E;
const XMLCh chUnderscore = 0x5F;

const XMLCh chSwappedUnicodeMarker = XMLCh(0xFFFE);
const XMLCh chUnicodeMarker = XMLCh(0xFEFF);

const XMLCh chDigit_0 = 0x30;
const XMLCh chDigit_1 = 0x31;
const XMLCh chDigit_2 = 0x32;
const XMLCh chDigit_3 = 0x33;
const XMLCh chDigit_4 = 0x34;
const XMLCh chDigit_5 = 0x35;
const XMLCh chDigit_6 = 0x36;
const XMLCh chDigit_7 = 0x37;
const XMLCh chDigit_8 = 0x38;
const XMLCh chDigit_9 = 0x39;

const XMLCh chLatin_A = 0x41;
const XMLCh chLatin_B = 0x42;
const XMLCh chLatin_C = 0x43;
const XMLCh chLatin_D = 0x44;
const XMLCh chLatin_E = 0x45;
const XMLCh chLatin_F = 0x46;
const XMLCh chLatin_G = 0x47;
const XMLCh chLatin_H = 0x48;
const XMLCh chLatin_I = 0x49;
const XMLCh chLatin_J = 0x4A;
const XMLCh chLatin_K = 0x4B;
const XMLCh chLatin_L = 0x4C;
const XMLCh chLatin_M = 0x4D;
const XMLCh chLatin_N = 0x4E;
const XMLCh chLatin_O = 0x4F;
const XMLCh chLatin_P = 0x50;
const XMLCh chLatin_Q = 0x51;
const XMLCh chLatin_R = 0x52;
const XMLCh chLatin_S = 0x53;
const XMLCh chLatin_T = 0x54;
const XMLCh chLatin_U = 0x55;
const XMLCh chLatin_V = 0x56;
const XMLCh chLatin_W = 0x57;
const XMLCh chLatin_X = 0x58;
const XMLCh chLatin_Y = 0x59;
const XMLCh chLatin_Z = 0x5A;

const XMLCh chLatin_a = 0x61;
const XMLCh chLatin_b = 0x62;
const XMLCh chLatin_c = 0x63;
const XMLCh chLatin_d = 0x64;
const XMLCh chLatin_e = 0x65;
const XMLCh chLatin_f = 0x66;
const XMLCh chLatin_g = 0x67;
const XMLCh chLatin_h = 0x68;
const XMLCh chLatin_i = 0x69;
const XMLCh chLatin_j = 0x6A;
const XMLCh chLatin_k = 0x6B;
const XMLCh chLatin_l = 0x6C;
const XMLCh chLatin_m = 0x6D;
const XMLCh chLatin_n = 0x6E;
const XMLCh chLatin_o = 0x6F;
const XMLCh chLatin_p = 0x70;
const XMLCh chLatin_q = 0x71;
const XMLCh chLatin_r = 0x72;
const XMLCh chLatin_s = 0x73;
const XMLCh chLatin_t = 0x74;
const XMLCh chLatin_u = 0x75;
const XMLCh chLatin_v = 0x76;
const XMLCh chLatin_w = 0x77;
const XMLCh chLatin_x = 0x78;
const XMLCh chLatin_y = 0x79;
const XMLCh chLatin_z = 0x7A;

const XMLCh chYenSign = 0xA5;
const XMLCh chWonSign = 0x20A9;

const XMLCh chLineSeparator = 0x2028;
const XMLCh chParagraphSeparator = 0x2029;

class XmlDocumentHolderLibXml2 : public IXmlDocumentHolder
{
 public:
  XmlDocumentHolderLibXml2(NodePrv* node)
  : m_document_node(node)
  {}
  XmlDocumentHolderLibXml2()
  : m_document_node(0)
  {}
  ~XmlDocumentHolderLibXml2();
  virtual XmlNode documentNode() { return XmlNode(0, m_document_node); }
  virtual IXmlDocumentHolder* clone();
  virtual void save(ByteArray& bytes);
  virtual String save();

 public:
  void assignNode(NodePrv* node) { m_document_node = node; }

 private:
  NodePrv* m_document_node;
};

class LIBXML2_RefCount
{
 public:
  LIBXML2_RefCount()
  : mRefcount(1)
  {}

  operator UInt32() { return mRefcount; }

  LIBXML2_RefCount& operator=(const UInt32& aValue)
  {
    mRefcount = aValue;
    return *this;
  }

  void operator++() { AtomicInt32::increment(&mRefcount); }

  bool operator--()
  {
    AtomicInt32::decrement(&mRefcount);
    return (mRefcount != 0);
  }

 private:
  Int32 mRefcount;
};

#define LIBXML2_IMPL_REFCOUNT                                                                                \
  \
private:                                                                                                     \
  LIBXML2_RefCount _libxml2_refcount;                                                                        \
  \
public:                                                                                                      \
  void add_ref() throw() { ++_libxml2_refcount; }                                                            \
  void release_ref() throw()                                                                                 \
  {                                                                                                          \
    if (!--_libxml2_refcount)                                                                                \
      delete this;                                                                                           \
  }

class LIBXML2_Element;
class LIBXML2_Document;
class LIBXML2_Attr;
class LIBXML2_Node;
class LIBXML2_NodeList;
class LIBXML2_NodeMap;
class LIBXML2_NamedNodeMap;
class LIBXML2_EmptyNamedNodeMap;
class LIBXML2_DocumentType;
class LIBXML2_Text;
class QualifiedName;

LIBXML2_Element* LIBXML2_NewElement(LIBXML2_Document* _xDoc, const String& nsURI, const String& elname);
// The construction method for document...
LIBXML2_Document* LIBXML2_NewDocument(const String& nsURI);

class LIBXML2_DOMImplementation
{
 public:
  LIBXML2_DOMImplementation()
  : mDocument(NULL)
  {}
  static LIBXML2_DOMImplementation* sDOMImplementation;
  ~LIBXML2_DOMImplementation();
  static const char* INTERFACE_NAME() { return "LIBXML2_DOMImplementation"; }
  bool hasFeature(const String& feature, const String& version);
  LIBXML2_DocumentType* createDocumentType(const String& qualifiedName, const String& publicId,
                                           const String& systemId);
  LIBXML2_Document* createDocument(const String& namespaceURI, const String& qualifiedName,
                                   LIBXML2_DocumentType* doctype);
  void SetDocument(LIBXML2_Document* _xDoc) { mDocument = _xDoc; }
  LIBXML2_Document* GetDocument() { return mDocument; }
  void ProcessXMLError(String& aErrorMessage, xmlError* aErr);
  void ProcessContextError(String& aErrorMessage, xmlParserCtxt* ctxt);
  LIBXML2_IMPL_REFCOUNT;

 private:
  LIBXML2_Document* mDocument;
};

LIBXML2_DOMImplementation* getDomImplementation();

class LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Node"; }
  LIBXML2_Node(LIBXML2_Document* aDocument);
  virtual ~LIBXML2_Node();
  static const UInt16 NODE_NODE = 0;
  static const UInt16 ELEMENT_NODE = 1;
  static const UInt16 ATTRIBUTE_NODE = 2;
  static const UInt16 TEXT_NODE = 3;
  static const UInt16 CDATA_SECTION_NODE = 4;
  static const UInt16 ENTITY_REFERENCE_NODE = 5;
  static const UInt16 ENTITY_NODE = 6;
  static const UInt16 PROCESSING_INSTRUCTION_NODE = 7;
  static const UInt16 COMMENT_NODE = 8;
  static const UInt16 DOCUMENT_NODE = 9;
  static const UInt16 DOCUMENT_TYPE_NODE = 10;
  static const UInt16 DOCUMENT_FRAGMENT_NODE = 11;
  static const UInt16 NOTATION_NODE = 12;

#ifdef DEBUG_NODELEAK
  void find_leaked();
#endif

  String nodeName();
  String nodeValue();
  void nodeValue(const String& attr);
  virtual UInt16 nodeType();
  LIBXML2_Node* parentNode();
  LIBXML2_NodeList* childNodes();
  LIBXML2_Node* firstChild();
  LIBXML2_Node* lastChild();
  LIBXML2_Node* previousSibling();
  LIBXML2_Node* nextSibling();
  LIBXML2_NamedNodeMap* attributes();
  LIBXML2_Document* ownerDocument();
  LIBXML2_Node* insertBefore(LIBXML2_Node* newChild, LIBXML2_Node* refChild);
  LIBXML2_Node* insertBeforePrivate(LIBXML2_Node* newChild, LIBXML2_Node* refChild);
  LIBXML2_Node* replaceChild(LIBXML2_Node* newChild, LIBXML2_Node* oldChild);
  LIBXML2_Node* removeChild(LIBXML2_Node* oldChild);
  LIBXML2_Node* removeChildPrivate(LIBXML2_Node* oldChild);
  LIBXML2_Node* appendChild(LIBXML2_Node* newChild);
  bool hasChildNodes();
  virtual LIBXML2_Node* shallowCloneNode(LIBXML2_Document* doc)
  {
    ARCANE_UNUSED(doc);
    throw NotImplementedException(A_FUNCINFO);
  }
  virtual LIBXML2_Node* cloneNode(bool deep);
  LIBXML2_Node* cloneNodePrivate(LIBXML2_Document* aDoc, bool deep);
  void normalize();
  bool isSupported(const String& feature, const String& version);
  String namespaceURI();
  String prefix();
  void prefix(const String& attr);
  String localName();
  bool hasAttributes() { return false; }
  void updateDocumentAncestorStatus(bool aStatus);
  void recursivelyChangeDocument(LIBXML2_Document* aNewDocument);
  LIBXML2_Element* searchForElementById(const String& elementId);

 public:
  LIBXML2_Node* mParent;
  std::list<LIBXML2_Node*>::iterator mPositionInParent;
  bool mDocumentIsAncestor;
  LIBXML2_Document* mDocument;
  String mNodeName, mLocalName, mNodeValue, mNamespaceURI;
  std::list<LIBXML2_Node*> mNodeList;
  UInt16 mNodeType;
  LIBXML2_RefCount _libxml2_refcount;

 public:
  void add_ref()
  {
    ++_libxml2_refcount;
    if (mParent != NULL)
      mParent->add_ref();
  }

  void release_ref()
  {
    if (_libxml2_refcount == 0) {
      ARCANE_ASSERT((_libxml2_refcount == 0), ("release_ref called too many times"));
    }
    bool hitZero = !--_libxml2_refcount;
    if (mParent == NULL) {
      if (hitZero) {
        delete this;
      }
    } else /* if the owner model is non-null, we will be destroyed when there are
            * no remaining references to the model.
            */
    {
      mParent->release_ref();
    }
  }
};

class LIBXML2_NodeList
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_NodeList"; }
  LIBXML2_NodeList()
  : mParent(NULL)
  {}
  LIBXML2_NodeList(LIBXML2_Node* parent)
  : mParent(parent)
  , m_hintIndex(0)
  {
    mParent->add_ref();
  }
  virtual ~LIBXML2_NodeList()
  {
    if (mParent != NULL)
      mParent->release_ref();
  }
  LIBXML2_Node* item(UInt32 index);
  UInt32 length();
  LIBXML2_Node* mParent;
  LIBXML2_IMPL_REFCOUNT;

 private:
  UInt32 m_hintIndex;
  std::list<LIBXML2_Node*>::iterator m_hintIterator;
};

class LIBXML2_NodeListDFSSearch : public LIBXML2_NodeList
{
 public:
  LIBXML2_NodeListDFSSearch(LIBXML2_Node* parent, const String& aNameFilter)
  : mParent(parent)
  , mNameFilter(aNameFilter)
  , mFilterType(LEVEL_1_NAME_FILTER)
  {
    mParent->add_ref();
  }

  LIBXML2_NodeListDFSSearch(LIBXML2_Node* parent, const String& aNamespaceFilter,
                            const String& aLocalnameFilter)
  : mParent(parent)
  , mNamespaceFilter(aNamespaceFilter)
  , mNameFilter(aLocalnameFilter)
  , mFilterType(LEVEL_2_NAME_FILTER)
  {
    mParent->add_ref();
  }

  LIBXML2_Node* item(UInt32 index);
  UInt32 length();
  virtual ~LIBXML2_NodeListDFSSearch()
  {
    if (mParent != NULL)
      mParent->release_ref();
  }

  LIBXML2_IMPL_REFCOUNT;

  LIBXML2_Node* mParent;
  String mNamespaceFilter, mNameFilter;
  enum
  {
    LEVEL_1_NAME_FILTER,
    LEVEL_2_NAME_FILTER
  } mFilterType;
};

class LIBXML2_EmptyNamedNodeMap
: NamedNodeMap
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_EmptyNamedNodeMap"; }
  LIBXML2_EmptyNamedNodeMap() {}
  virtual ~LIBXML2_EmptyNamedNodeMap() {}
  LIBXML2_Node* getNamedItem(const String& name)
  {
    ARCANE_UNUSED(name);
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_Node* setNamedItem(LIBXML2_Node* arg)
  {
    ARCANE_UNUSED(arg);
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_Node* removeNamedItem(const String& name)
  {
    ARCANE_UNUSED(name);
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_Node* item(UInt32 index)
  {
    ARCANE_UNUSED(index);
    throw NotImplementedException(A_FUNCINFO);
  }
  UInt32 length()
  {
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_Node* getNamedItemNS(const String& namespaceURI, String& localName)
  {
    ARCANE_UNUSED(namespaceURI);
    ARCANE_UNUSED(localName);
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_Node* setNamedItemNS(LIBXML2_Node* arg)
  {
    ARCANE_UNUSED(arg);
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_Node* removeNamedItemNS(const String& namespaceURI, const String& localName)
  {
    ARCANE_UNUSED(namespaceURI);
    ARCANE_UNUSED(localName);
    throw NotImplementedException(A_FUNCINFO);
  }
  LIBXML2_IMPL_REFCOUNT;
};

class LIBXML2_CharacterData : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_CharacterData"; }
  LIBXML2_CharacterData(LIBXML2_Document* aDocument)
  : LIBXML2_Node(aDocument)
  {}
  virtual ~LIBXML2_CharacterData() {}
  String Data();
  void Data(const String& attr);
  void nodeValue(const String& attr);
  UInt32 length();
  String substringdata(UInt32 offset, UInt32 count);
  void appenddata(const String& arg);
  void insertdata(UInt32 offset, const String& arg);
  void deletedata(UInt32 offset, UInt32 count);
  void replacedata(UInt32 offset, UInt32 count, const String& arg);
  LIBXML2_IMPL_REFCOUNT;
};

#define LIBXML2_IMPL_NODETYPE(type)                                                                          \
  \
UInt16                                                                                                         \
  nodeType()                                                                                                 \
  {                                                                                                          \
    return LIBXML2_Node::type##_NODE;                                                                        \
  }

class LIBXML2_Attr : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Attr"; }
  LIBXML2_Attr(LIBXML2_Document* aDocument)
  : LIBXML2_Node(aDocument)
  , mSpecified(false)
  {}
  virtual ~LIBXML2_Attr() {}
  LIBXML2_IMPL_NODETYPE(ATTRIBUTE);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  String name();
  bool specified();
  String value();
  void value(const String& attr);
  LIBXML2_Element* ownerElement();
  bool mSpecified;
};

class LIBXML2_Element : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Element"; }
  LIBXML2_Element(LIBXML2_Document* aDocument)
  : LIBXML2_Node(aDocument)
  {}
  virtual ~LIBXML2_Element();
  LIBXML2_IMPL_NODETYPE(ELEMENT);
  LIBXML2_NamedNodeMap* attributes();
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  String tagName();
  String getAttribute(const String& name);
  void setAttribute(const String& name, const String& value);
  void removeAttribute(const String& name);
  LIBXML2_Attr* getAttributeNode(const String& name);
  LIBXML2_Attr* setAttributeNode(LIBXML2_Attr* newAttr);
  LIBXML2_Attr* removeAttributeNode(LIBXML2_Attr* oldAttr);
  LIBXML2_NodeList* getElementsByTagName(const String& name);
  String getAttributeNS(const String& namespaceURI, const String& localname);
  void setAttributeNS(const String& namespaceURI, const String& qualifiedname, const String& value);
  void removeAttributeNS(const String& namespaceURI, const String& localname);
  LIBXML2_Attr* getAttributeNodeNS(const String& namespaceURI, const String& localname);
  LIBXML2_Attr* setAttributeNodeNS(LIBXML2_Attr* newAttr);
  LIBXML2_NodeList* getElementsByTagNameNS(const String& namespaceURI, const String& localname);
  bool hasAttribute(const String& name);
  bool hasAttributeNS(const String& namespaceURI, const String& localname);
  bool hasAttributes();
  LIBXML2_Element* searchForElementById(const String& elementId);

  class LocalName
  {
   public:
    LocalName(const String& aname)
    : name(aname)
    {}
    LocalName(const LocalName& ln)
    : name(ln.name)
    {}
    bool operator==(const LocalName& aCompareWith) const { return name == aCompareWith.name; }

    bool operator<(const LocalName& aCompareWith) const { return name < aCompareWith.name; }
    String name;
  };

  class QualifiedName
  {
   public:
    QualifiedName(const String& aNamespace, const String& aname)
    : ns(aNamespace)
    , name(aname)
    {}
    QualifiedName(const QualifiedName& ln)
    : ns(ln.ns)
    , name(ln.name)
    {}

    bool operator==(const QualifiedName& aCompareWith) const
    {
      return name == aCompareWith.name && ns == aCompareWith.ns;
    }

    bool operator<(const QualifiedName& aCompareWith) const { return name < aCompareWith.name; }
    String ns;
    String name;
  };
  LIBXML2_IMPL_REFCOUNT;

  std::map<QualifiedName, LIBXML2_Attr*> attributeMapNS;
  std::map<LocalName, LIBXML2_Attr*> attributeMap;
};

class LIBXML2_NamedNodeMap : public NamedNodeMap
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_NamedNodeMap"; }
  LIBXML2_NamedNodeMap()
  : mElement(NULL)
  , m_hintIndex(0)
  {}
  LIBXML2_NamedNodeMap(LIBXML2_Element* aElement);
  virtual ~LIBXML2_NamedNodeMap();
  virtual LIBXML2_Node* getNamedItem(const String& name);
  virtual LIBXML2_Node* setNamedItem(LIBXML2_Node* arg);
  virtual LIBXML2_Node* removeNamedItem(const String& name);
  virtual LIBXML2_Node* item(UInt32 index);
  virtual UInt32 length();
  virtual LIBXML2_Node* getNamedItemNS(const String& namespaceURI, const String& localname);
  virtual LIBXML2_Node* setNamedItemNS(LIBXML2_Node* arg);
  virtual LIBXML2_Node* removeNamedItemNS(const String& namespaceURI, const String& localname);
  LIBXML2_Element* mElement;
  LIBXML2_IMPL_REFCOUNT;

 private:
  UInt32 m_hintIndex;
  std::map<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>::iterator m_hintIterator;
};

class LIBXML2_NamedNodeMapDT : public LIBXML2_NamedNodeMap
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_NamedNodeDT"; }
  LIBXML2_NamedNodeMapDT(LIBXML2_DocumentType* aDocType, UInt16 aType);
  virtual ~LIBXML2_NamedNodeMapDT();
  LIBXML2_Node* getNamedItem(const String& name);
  LIBXML2_Node* setNamedItem(LIBXML2_Node* arg);
  LIBXML2_Node* removeNamedItem(const String& name);
  LIBXML2_Node* item(UInt32 index);
  UInt32 length();
  LIBXML2_Node* getNamedItemNS(const String& namespaceURI, const String& localname);
  LIBXML2_Node* setNamedItemNS(LIBXML2_Node* arg);
  LIBXML2_Node* removeNamedItemNS(const String& namespaceURI, const String& localname);
  LIBXML2_IMPL_REFCOUNT;

  LIBXML2_DocumentType* mDocType;
  UInt16 mType;
};

class LIBXML2_TextBase : public LIBXML2_CharacterData
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_TextBase"; }
  LIBXML2_TextBase(LIBXML2_Document* aDocument)
  : LIBXML2_CharacterData(aDocument)
  {}
  virtual ~LIBXML2_TextBase() {}
  LIBXML2_Text* splitText(UInt32 offset);
  LIBXML2_IMPL_REFCOUNT;
};

class LIBXML2_Text : public LIBXML2_TextBase
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Text"; }
  LIBXML2_Text(LIBXML2_Document* aDocument)
  : LIBXML2_TextBase(aDocument)
  {}
  virtual ~LIBXML2_Text() {}
  LIBXML2_IMPL_NODETYPE(TEXT);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
};

class LIBXML2_Comment : public LIBXML2_CharacterData
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Comment"; }
  LIBXML2_Comment(LIBXML2_Document* aDocument)
  : LIBXML2_CharacterData(aDocument)
  {}
  virtual ~LIBXML2_Comment() {}
  LIBXML2_IMPL_NODETYPE(COMMENT);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
};

class LIBXML2_CDATASection : public LIBXML2_TextBase
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_CDATASection"; }
  LIBXML2_CDATASection(LIBXML2_Document* aDocument)
  : LIBXML2_TextBase(aDocument)
  {}
  virtual ~LIBXML2_CDATASection() {}
  LIBXML2_IMPL_NODETYPE(CDATA_SECTION);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  LIBXML2_Text* splitText(UInt32 offset);
};

class LIBXML2_DocumentType : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_DocumentType"; }
  LIBXML2_DocumentType(LIBXML2_Document* aDocument, const String& qualifiedName, const String& publicId,
                       const String& systemId)
  : LIBXML2_Node(aDocument)
  {
    mNodeName = qualifiedName;
    mPublicId = publicId;
    mSystemId = systemId;
  }
  virtual ~LIBXML2_DocumentType() {}
  LIBXML2_IMPL_NODETYPE(DOCUMENT_TYPE);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  String name();
  LIBXML2_NamedNodeMap* entities();
  LIBXML2_NamedNodeMap* notations();
  String publicId();
  String systemId();
  String internalSubset();
  String mPublicId, mSystemId;
};

class LIBXML2_Notation : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Notation"; }
  LIBXML2_Notation(LIBXML2_Document* aDocument, const String& aPublicId, String& aSystemId)
  : LIBXML2_Node(aDocument)
  , mPublicId(aPublicId)
  , mSystemId(aSystemId)
  {}
  virtual ~LIBXML2_Notation() {}
  LIBXML2_IMPL_NODETYPE(NOTATION);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  String publicId();
  String systemId();
  String mPublicId, mSystemId;
};

class LIBXML2_Entity : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Entity"; }

  LIBXML2_Entity(LIBXML2_Document* aDocument, String& aPublicId, String& aSystemId, String& aNotationname)
  : LIBXML2_Node(aDocument)
  , mPublicId(aPublicId)
  , mSystemId(aSystemId)
  , mNotationName(aNotationname)
  {}
  virtual ~LIBXML2_Entity() {}
  LIBXML2_IMPL_NODETYPE(ENTITY);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  String publicId();
  String systemId();
  String notationName();
  String mPublicId, mSystemId, mNotationName;
};

class LIBXML2_EntityReference : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_EntityReference"; }
  LIBXML2_EntityReference(LIBXML2_Document* aDocument)
  : LIBXML2_Node(aDocument)
  {}
  virtual ~LIBXML2_EntityReference() {}
  LIBXML2_IMPL_NODETYPE(ENTITY_REFERENCE);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
};

class LIBXML2_ProcessingInstruction : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_ProcessingInstruction"; }
  LIBXML2_ProcessingInstruction(LIBXML2_Document* aDocument, String aTarget, String aData)
  : LIBXML2_Node(aDocument)
  {
    mNodeName = aTarget;
    mNodeValue = aData;
  }

  virtual ~LIBXML2_ProcessingInstruction() {}
  LIBXML2_IMPL_NODETYPE(PROCESSING_INSTRUCTION);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  String target();
  String Data();
  void Data(const String& attr);
};

class LIBXML2_DocumentFragment : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_DocumentFragment"; }
  LIBXML2_DocumentFragment(LIBXML2_Document* aDocument)
  : LIBXML2_Node(aDocument)
  {}
  virtual ~LIBXML2_DocumentFragment() {}
  LIBXML2_IMPL_NODETYPE(DOCUMENT_FRAGMENT);
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
};

class LIBXML2_Document : public LIBXML2_Node
{
 public:
  static const char* INTERFACE_NAME() { return "LIBXML2_Document"; }
  LIBXML2_Document(const String& namespaceURI, const String& qualifiedname, LIBXML2_DocumentType* doctype);
  LIBXML2_Document()
  : LIBXML2_Node(this)
  , impl_(NULL)
  , context_(NULL)
  {
    // We are our own document ancestor...
    mDocumentIsAncestor = true;
    mDocument->release_ref();
  }

  virtual ~LIBXML2_Document()
  {
    if (context_) {
      if (context_->myDoc != nullptr) {
        xmlFreeDoc(context_->myDoc);
        Impl_(nullptr);
      }
    }
    if (Impl_() != nullptr) {
      xmlFreeDoc(Impl_());
      Impl_(nullptr);
    }
  }

  LIBXML2_IMPL_NODETYPE(DOCUMENT);
  LIBXML2_Text* doctype();
  LIBXML2_DOMImplementation* implementation();
  LIBXML2_Element* documentElement();
  LIBXML2_Element* createElement(const String& tagName);
  LIBXML2_DocumentFragment* createDocumentFragment();
  LIBXML2_Text* createTextNode(const String& data);
  LIBXML2_Comment* createComment(const String& data);
  LIBXML2_CDATASection* createCDATASection(const String& data);
  LIBXML2_ProcessingInstruction* createProcessingInstruction(const String& target, const String& data);
  LIBXML2_Attr* createAttribute(const String& name);
  LIBXML2_EntityReference* createEntityReference(const String& name);
  LIBXML2_NodeList* getElementsByTagName(const String& tagName);
  LIBXML2_Node* importNode(LIBXML2_Node* importedNode, bool deep);
  LIBXML2_Element* createElementNS(const String& namespaceURI, const String& qualifiedname);
  LIBXML2_Attr* createAttributeNS(const String& namespaceURI, const String& qualifiedname);
  LIBXML2_NodeList* getElementsByTagNameNS(const String& namespaceURI, const String& localname);
  LIBXML2_Element* getElementById(const String& elementId);
  void Impl_(_xmlDoc* Doc) { impl_ = Doc; }
  _xmlDoc* Impl_(void) { return impl_; }
  void Context_(xmlParserCtxt* Context) { context_ = Context; }
  xmlParserCtxt* Context_() { return context_; }
  LIBXML2_Node* shallowCloneNode(LIBXML2_Document* aDoc);
  LIBXML2_Element* searchForElementById(const String& elementId);

 private:
  _xmlDoc* impl_;
  xmlParserCtxt* context_;
};

class LIBXML2_DOMNamespaceContext;

class LIBXML2_DOMWriter
{
 public:
  LIBXML2_DOMWriter(int indentation)
  : impl_(0)
  , context_(0)
  , indent(indentation)
  , do_indent(0)
  , DeepNode(0)
  {}

  LIBXML2_DOMWriter(_xmlDoc* xmlDoc_, xmlParserCtxt* context, int indentation)
  : impl_(xmlDoc_)
  , context_(context)
  , indent(indentation)
  , do_indent(0)
  , DeepNode(0)
  {}
  void writeNode(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Node* n, StringBuilder& appendTo);
  void writeElement(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Element* el, StringBuilder& appendTo);
  void writeAttr(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Attr* at, StringBuilder& appendTo);
  void writeText(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Text* txt, StringBuilder& appendTo);
  void writeCDATASection(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_CDATASection* cds,
                         StringBuilder& appendTo);
  void writeEntity(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Entity* er, StringBuilder& appendTo);
  void writeEntityReference(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_EntityReference* er,
                            StringBuilder& appendTo);
  void writeProcessingInstruction(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_ProcessingInstruction* pi,
                                  StringBuilder& appendTo);
  void writeComment(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Comment* comment, StringBuilder& appendTo);
  void writeDocument(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Document* doc, StringBuilder& appendTo);
  void writeDocumentType(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_DocumentType* dt, StringBuilder& appendTo);
  void writeDocumentFragment(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_DocumentFragment* df,
                             StringBuilder& appendTo);
  void writeNotation(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Notation* nt, StringBuilder& appendTo);

  static const char* get_encoding_or_utf8(const String& encoding)
  {
    if (encoding.empty()) {
      // If we don't specify this to the xmlDocDump* functions (using nullptr instead),
      // then some other encoding is used, causing them to fail on non-ASCII characters.
      return "UTF-8";
    } else
      return (const char*)encoding.localstr();
  }

 private:
  void DoIndentation(bool StartEnd, StringBuilder& appendTo);
  _xmlDoc* impl_;
  xmlParserCtxt* context_;

  LIBXML2_IMPL_REFCOUNT;

 public:
  int indent;
  int do_indent;
  int DeepNode;
};

class LIBXML2_DOMNamespaceContext
{
 public:
  LIBXML2_DOMNamespaceContext(LIBXML2_DOMNamespaceContext* aParent);
  void setDefaultNamespace(const String& newns);
  void recordPrefix(const String& prefix, const String& ns);
  String getDefaultNamespace();
  String findNamespaceForPrefix(const String& prefix);
  String findPrefixForNamespace(const String& ns);
  void possiblyInventPrefix(const String& prefix);
  void resolveOrInventPrefixes();
  void writeXMLNS(StringBuilder& appendTo);
  LIBXML2_IMPL_REFCOUNT;

 private:
  LIBXML2_DOMNamespaceContext* mParent;
  bool mOverrideDefaultNamespace;
  String mDefaultNamespace;
  std::map<String, String> mURIfromPrefix;
  std::map<String, String> mPrefixfromURI;
  std::list<String> mNamespacesNeedingPrefixes;
};

ARCANE_END_NAMESPACE_DOM
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/utils/UtilsTypes.h>

#include "arcane/XmlException.h"
#include "arcane/utils/PlatformUtils.h"
#include <arcane/utils/TraceInfo.h>

#include <map>
#include <string>

#include <libxml/parserInternals.h> //For xmlCreateFileParserCtxt().
#include <libxml/xinclude.h>
#include <libxml/xmlschemas.h>

#include <algorithm>

//#include <codecvt> // not available with gcc at least 4.8

#ifdef USE_WCHAR
#include <cwchar>
#define U(S) L##S
typedef std::wstring ustring;

#else /* USE_WCHAR */
#define U(S) u##S
typedef std::u16string ustring;

static_assert(sizeof(char16_t) == sizeof(short), "Inconsistent size");
typedef unsigned short unsigned_char16_t;

int
wcscmp(const char16_t* s1, const char16_t* s2)
{
  while (*s1 == *s2++)
    if (*s1++ == 0)
      return (0);
  return (*(const unsigned_char16_t*)s1 - *(const unsigned_char16_t*)--s2);
}

int
wcsncmp(const char16_t* s1, const char16_t* s2, size_t n)
{
  if (n == 0)
    return (0);
  do {
    if (*s1 != *s2++) {
      return (*(const unsigned_char16_t*)s1 - *(const unsigned_char16_t*)--s2);
    }
    if (*s1++ == 0)
      break;
  } while (--n != 0);
  return (0);
}

char16_t*
wcschr(const char16_t* s, const char16_t c)
{
  while (*s != c && *s != L'\0')
    s++;
  if (*s == c)
    return ((char16_t*)s);
  return (NULL);
}

size_t
wcslen(const char16_t* s)
{
  const char16_t* p;
  p = s;
  while (*p)
    p++;
  return p - s;
}

int
wcstombs(char* dest, const char16_t* src, size_t n)
{
  int i = n;
  while (--i >= 0) {
    if (!(*dest++ = *src++))
      break;
  }
  return n - i - 1;
}
#endif /* USE_WCHAR */

// Begin interface with Arcane
ARCANE_BEGIN_NAMESPACE
ARCANE_BEGIN_NAMESPACE_DOM

// Prevent compatibility check for next reinterpret_cast
// static_assert(sizeof(XMLCh) == sizeof(decltype(*String().utf16().begin())), "Inconsistent data size");
// static_assert(sizeof(XMLCh) == sizeof(UChar), "Inconsistent data size");
static_assert(sizeof(XMLByte) == sizeof(Byte), "Inconsistent data size");
static_assert(sizeof(xmlChar) == sizeof(char), "Inconsistent data size");

LIBXML2_Node* WrapXML2Node(LIBXML2_Document* doc, xmlNode* x2node);

struct LIBXML2_PartialLoad
{
  LIBXML2_DOMImplementation* mImpl;
  String mMsgTo;
};

static void
LIBXML2_XMLStructuredHandler(void* userData, xmlErrorPtr error)
{
  LIBXML2_PartialLoad* pl = reinterpret_cast<LIBXML2_PartialLoad*>(userData);
  pl->mImpl->ProcessXMLError(pl->mMsgTo, error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static LIBXML2_Node*
impl(NodePrv* p)
{
  return reinterpret_cast<LIBXML2_Node*>(p);
}
static LIBXML2_Attr*
impl(AttrPrv* p)
{
  return reinterpret_cast<LIBXML2_Attr*>(p);
}
static LIBXML2_Element*
impl(ElementPrv* p)
{
  return reinterpret_cast<LIBXML2_Element*>(p);
}
static LIBXML2_Document*
impl(DocumentPrv* p)
{
  return reinterpret_cast<LIBXML2_Document*>(p);
}
static LIBXML2_DocumentType*
impl(DocumentTypePrv* p)
{
  return reinterpret_cast<LIBXML2_DocumentType*>(p);
}
static LIBXML2_DOMImplementation*
impl(ImplementationPrv* p)
{
  return reinterpret_cast<LIBXML2_DOMImplementation*>(p);
}
static LIBXML2_CharacterData*
impl(CharacterDataPrv* p)
{
  return reinterpret_cast<LIBXML2_CharacterData*>(p);
}
static LIBXML2_Text*
impl(TextPrv* p)
{
  return reinterpret_cast<LIBXML2_Text*>(p);
}
static LIBXML2_NodeList*
impl(NodeListPrv* p)
{
  return reinterpret_cast<LIBXML2_NodeList*>(p);
}
static LIBXML2_DocumentFragment*
impl(DocumentFragmentPrv* p)
{
  return reinterpret_cast<LIBXML2_DocumentFragment*>(p);
}
static LIBXML2_NamedNodeMap*
impl(NamedNodeMapPrv* p)
{
  return reinterpret_cast<LIBXML2_NamedNodeMap*>(p);
}
static LIBXML2_Comment*
impl(CommentPrv* p)
{
  return reinterpret_cast<LIBXML2_Comment*>(p);
}
static LIBXML2_CDATASection*
impl(CDATASectionPrv* p)
{
  return reinterpret_cast<LIBXML2_CDATASection*>(p);
}
static LIBXML2_ProcessingInstruction*
impl(ProcessingInstructionPrv* p)
{
  return reinterpret_cast<LIBXML2_ProcessingInstruction*>(p);
}
static LIBXML2_EntityReference*
impl(EntityReferencePrv* p)
{
  return reinterpret_cast<LIBXML2_EntityReference*>(p);
}
static LIBXML2_Entity*
impl(EntityPrv* p)
{
  return reinterpret_cast<LIBXML2_Entity*>(p);
}
static LIBXML2_Notation*
impl(NotationPrv* p)
{
  return reinterpret_cast<LIBXML2_Notation*>(p);
}
static LIBXML2_DOMWriter*
impl(DOMWriterPrv* p)
{
  return reinterpret_cast<LIBXML2_DOMWriter*>(p);
}

static NodePrv*
cvt(LIBXML2_Node* p)
{
  return reinterpret_cast<NodePrv*>(p);
}
static AttrPrv*
cvt(LIBXML2_Attr* p)
{
  return reinterpret_cast<AttrPrv*>(p);
}
static ElementPrv*
cvt(LIBXML2_Element* p)
{
  return reinterpret_cast<ElementPrv*>(p);
}
static DocumentPrv*
cvt(LIBXML2_Document* p)
{
  return reinterpret_cast<DocumentPrv*>(p);
}
static DocumentTypePrv*
cvt(LIBXML2_DocumentType* p)
{
  return reinterpret_cast<DocumentTypePrv*>(p);
}
static ImplementationPrv*
cvt(LIBXML2_DOMImplementation* p)
{
  return reinterpret_cast<ImplementationPrv*>(p);
}
static CharacterDataPrv*
cvt(LIBXML2_CharacterData* p)
{
  return reinterpret_cast<CharacterDataPrv*>(p);
}
static TextPrv*
cvt(LIBXML2_Text* p)
{
  return reinterpret_cast<TextPrv*>(p);
}
static NodeListPrv*
cvt(LIBXML2_NodeList* p)
{
  return reinterpret_cast<NodeListPrv*>(p);
}
static DocumentFragmentPrv*
cvt(LIBXML2_DocumentFragment* p)
{
  return reinterpret_cast<DocumentFragmentPrv*>(p);
}
static NamedNodeMapPrv*
cvt(LIBXML2_NamedNodeMap* p)
{
  return reinterpret_cast<NamedNodeMapPrv*>(p);
}
static CommentPrv*
cvt(LIBXML2_Comment* p)
{
  return reinterpret_cast<CommentPrv*>(p);
}
static CDATASectionPrv*
cvt(LIBXML2_CDATASection* p)
{
  return reinterpret_cast<CDATASectionPrv*>(p);
}
static ProcessingInstructionPrv*
cvt(LIBXML2_ProcessingInstruction* p)
{
  return reinterpret_cast<ProcessingInstructionPrv*>(p);
}
static EntityPrv*
cvt(LIBXML2_Entity* p)
{
  return reinterpret_cast<EntityPrv*>(p);
}
static EntityReferencePrv*
cvt(LIBXML2_EntityReference* p)
{
  return reinterpret_cast<EntityReferencePrv*>(p);
}
static NotationPrv*
cvt(LIBXML2_Notation* p)
{
  return reinterpret_cast<NotationPrv*>(p);
}
static DOMWriterPrv*
cvt(LIBXML2_DOMWriter* p)
{
  return reinterpret_cast<DOMWriterPrv*>(p);
}

// ------------------------------------------------------------
//  Static constants
// ------------------------------------------------------------
static const XMLCh g1_0[] = // Points to "1.0"
        { chDigit_1, chPeriod, chDigit_0, chNull };
static const XMLCh g2_0[] = // Points to "2.0"
        { chDigit_2, chPeriod, chDigit_0, chNull };
static const XMLCh g3_0[] = // Points to "3.0"
        { chDigit_3, chPeriod, chDigit_0, chNull };
static const XMLCh gTrav[] = // Points to "Traversal"
        { chLatin_T, chLatin_r, chLatin_a, chLatin_v, chLatin_e,
          chLatin_r, chLatin_s, chLatin_a, chLatin_l, chNull };
static const XMLCh gCore[] = // Points to "Core"
        { chLatin_C, chLatin_o, chLatin_r, chLatin_e, chNull };
static const XMLCh gRange[] = // Points to "Range"
        { chLatin_R, chLatin_a, chLatin_n, chLatin_g, chLatin_e, chNull };
static const XMLCh gLS[] = // Points to "LS"
        { chLatin_L, chLatin_S, chNull };
static const XMLCh gXPath[] = // Points to "XPath"
        { chLatin_X, chLatin_P, chLatin_a, chLatin_t, chLatin_h, chNull };
const XMLCh gXMLString[] = { chLatin_x, chLatin_m, chLatin_l, chNull };
// static XMLCh null_xc[1] = { 0 };

String
format_xml_error(const xmlError* error)
{
  if (!error)
    error = xmlGetLastError();

  if (!error || error->code == XML_ERR_OK)
    return String(); // No error
  String str, str2;
  String strError;

  if (error->file && *error->file != '\0') {
    str = str + "File ";
    str = str + error->file;
  }
  if (error->line > 0) {
    strError = String::format("{0}", error->line);
    str = str + (str.empty() ? "Line " : ", line ") + strError;
    if (error->int2 > 0) {
      strError = String::format("{0}", error->int2);
      str = str + ", column " + strError;
    }
  }
  const bool two_lines = !str.empty();
  if (two_lines)
    str = str + " ";
  switch (error->level) {
  case XML_ERR_WARNING:
    str = str + "(warning):";
    break;
  case XML_ERR_ERROR:
    str = str + "(error):";
    break;
  case XML_ERR_FATAL:
    str = str + "(fatal):";
    break;
  default:
    str = str + "():";
    break;
  }
  str = str + (two_lines ? "\n" : " ");
  if (error->message && *error->message != '\0')
    str = str + error->message;
  else {
    strError = String::format("{0}", error->code);
    str = str + "Error code " + strError;
  }
  // If the string does not end with end-of-line, append an end-of-line.
  if (!str.endsWith("\n"))
    str = str + "\n";
  return str;
}

String
format_xml_parser_error(const xmlParserCtxt* parser_context)
{
  if (!parser_context)
    return "Error. format_xml_parser_error() called with parser_context == nullptr\n";
  const xmlErrorPtr error = xmlCtxtGetLastError(const_cast<xmlParserCtxt*>(parser_context));
  if (!error)
    return String(); // No error
  String str;
  if (!parser_context->wellFormed)
    str = str + "Document not well-formed.\n";
  return str + format_xml_error(error);
}

NodePrv*
toNodePrv(const Node& node)
{
  return node._impl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlDocumentHolderLibXml2::
~XmlDocumentHolderLibXml2()
{
  if (m_document_node)
    delete impl(m_document_node);
}

IXmlDocumentHolder* XmlDocumentHolderLibXml2::
clone()
{
  LIBXML2_Node* n = impl(m_document_node)->cloneNode(true);
  return new XmlDocumentHolderLibXml2(cvt(n));
}

void XmlDocumentHolderLibXml2::
save(ByteArray& bytes)
{
  domutils::saveDocument(bytes, documentNode().domNode());
}

String XmlDocumentHolderLibXml2::
save()
{
  // TODO verifier qu'on sauve toujours en UTF8.
  ByteUniqueArray bytes;
  domutils::saveDocument(bytes, documentNode().domNode());
  // Ajoute le 0 terminal
  bytes.add('\0');
  String new_s = String::fromUtf8(bytes);
  return new_s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DOMImplementation::
DOMImplementation()
: m_p(0)
{
  m_p = cvt(getDomImplementation());
}

DOMImplementation::
DOMImplementation(ImplementationPrv* prv)
: m_p(prv)
{}

DOMImplementation::
~DOMImplementation() {}

void
DOMImplementation::
_checkValid() const
{
  if (!m_p)
    arcaneNullPointerError();
}

ImplementationPrv*
DOMImplementation::
_impl() const
{
  return m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
bool DOMImplementation::
hasFeature(const DOMString& feature, const DOMString& version) const
{
  _checkValid();
  const XMLCh* xmlfeature = reinterpret_cast<const XMLCh*>(feature.utf16().begin());
  const XMLCh* xmlversion = reinterpret_cast<const XMLCh*>(version.utf16().begin());
  if (!xmlfeature)
    return false;
  // ignore the + modifier
  if (*xmlfeature == chPlus)
    xmlfeature++;
  bool anyVersion = (xmlversion == 0 || !*xmlversion);
  bool version1_0 = wcscmp(xmlversion, g1_0) == 0;
  bool version2_0 = wcscmp(xmlversion, g2_0) == 0;
  bool version3_0 = wcscmp(xmlversion, g3_0) == 0;
  // Currently, we support only XML Level 1 version 1.0
  if (wcscmp(xmlfeature, gXMLString) == 0 && (anyVersion || version1_0 || version2_0))
    return true;
  if (wcscmp(xmlfeature, gCore) == 0 && (anyVersion || version1_0 || version2_0 || version3_0))
    return true;
  if (wcscmp(xmlfeature, gTrav) == 0 && (anyVersion || version2_0))
    return true;
  if (wcscmp(xmlfeature, gRange) == 0 && (anyVersion || version2_0))
    return true;
  if (wcscmp(xmlfeature, gLS) == 0 && (anyVersion || version3_0))
    return true;
  if (wcscmp(xmlfeature, gXPath) == 0 && (anyVersion || version3_0))
    return true;
  return true;
}

Document DOMImplementation::
createDocument(const DOMString& namespace_uri, const DOMString& qualifiedname,
               const DocumentType& doctype) const
{
  _checkValid();
  LIBXML2_Document* doc = nullptr;
  try {
    LIBXML2_DocumentType* doctypei = impl(doctype._impl());
    doc = impl(m_p)->createDocument(namespace_uri, qualifiedname, doctypei);
  } catch (const DOMException& ex) {
    cerr << "** DOMException call in createDocument(const DOMString& namespace_uri, const DOMString& "
            "qualifiedname, const DocumentType& doctype)"
         << ex.code << '\n';
    throw ex;
  }
  impl(m_p)->SetDocument(doc);
  return cvt(doc);
}

DocumentType DOMImplementation::
createDocumentType(const DOMString& qualifiedname, const DOMString& public_id,
                   const DOMString& system_id) const
{
  _checkValid();
  LIBXML2_DocumentType* doctype = impl(m_p)->createDocumentType(qualifiedname, public_id, system_id);
  return cvt(doctype);
}

IXmlDocumentHolder* DOMImplementation::
_newDocument()
{
  // create the Empty document
  LIBXML2_Document* doc = new LIBXML2_Document();
  if (doc == NULL)
    throw Arcane::Exception::exception();
  XmlDocumentHolderLibXml2* xml_doc = new XmlDocumentHolderLibXml2(reinterpret_cast<NodePrv*>(doc));
  if (xml_doc == NULL)
    throw Arcane::Exception::exception();
  return xml_doc;
}

enum XML_PARSE_TYPE
{
  XML_PARSE_SCHEMA_FILE,
  XML_PARSE_SCHEMA_MEMORY
};
static xmlParserCtxt*
_xmlSchemaValidationAndParsing(const String& fname, ITraceMng* msg, XML_PARSE_TYPE type,
                               const String& schemaname, ByteConstArrayView schemaData)
{
  xmlSchemaPtr schema = NULL;
  xmlSchemaValidCtxtPtr vctxt = NULL;
  xmlSchemaParserCtxtPtr pctxt = NULL;
  xmlParserCtxt* context = NULL;
  xmlDocPtr myDoc = NULL;
  int result = 0;
  // create the file parser context
  if ((context = xmlCreateFileParserCtxt(fname.localstr())) == NULL) {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 Could not create File parser context '{0}'\n{1}",
                                              fname, format_xml_error(xmlGetLastError())));
  }
  if (type == XML_PARSE_SCHEMA_FILE) {
    // create schema parser context
    if ((pctxt = xmlSchemaNewParserCtxt(schemaname.localstr())) == NULL) {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 Could not create context Schema '{0}'\n{1}",
                                                schemaname, format_xml_error(xmlGetLastError())));
    }
  } else {
    // create schema parser context
    if ((pctxt = xmlSchemaNewMemParserCtxt(
                 const_cast<char*>(reinterpret_cast<const char*>(schemaData.begin())), schemaData.size())) ==
        NULL) {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 Could not create context Schema '{0}'\n{1}",
                                                schemaname, format_xml_error(xmlGetLastError())));
    }
  }
  xmlSchemaSetParserErrors(pctxt, (xmlSchemaValidityErrorFunc)fprintf, (xmlSchemaValidityWarningFunc)fprintf,
                           stderr);
  if ((schema = xmlSchemaParse(pctxt)) == NULL) {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 Could not parse Schema '{0}'\n{1}", schemaname,
                                              format_xml_error(xmlGetLastError())));
  }

  if ((myDoc = xmlReadFile(fname.localstr(), NULL, 0)) == NULL) {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 Could not read xml file '{0}'\n{1}", fname,
                                              format_xml_error(xmlGetLastError())));
  }
  if ((vctxt = xmlSchemaNewValidCtxt(schema)) == NULL) {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 Could not validate context Schema '{0}'\n{1}",
                                              schemaname, format_xml_error(xmlGetLastError())));
  }
  xmlSchemaSetParserErrors(pctxt, (xmlSchemaValidityErrorFunc)fprintf, (xmlSchemaValidityWarningFunc)fprintf,
                           stderr);
  // populate xml tree with default values
  xmlSchemaSetValidOptions(vctxt, XML_SCHEMA_VAL_VC_I_CREATE);
  result = xmlSchemaValidateDoc(vctxt, myDoc);
  if (pctxt != NULL)
    xmlSchemaFreeParserCtxt(pctxt);
  if (schema != NULL)
    xmlSchemaFree(schema);
  if (vctxt != NULL)
    xmlSchemaFreeValidCtxt(vctxt);
  if (result == 0) {
    if (msg) {
      msg->info() << "LIBXML2 ---------------- _xmlSchemaValidation OK";
    }
  } else if (result > 0) {
    /// the XML file does not match the XML Schema structure
    ///  cannot validate the xml document with given schema!\n";
    // throw Arcane::XmlException(A_FUNCINFO, String::format("-LIBXML2 the XML file '{0}' does not match the
    // XML Schema '{1}' error='{2}' \n",fname,schemaname, format_xml_error(xmlGetLastError())));

    ///< this is a warning not an error>
  } else {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 '{0}' validation generated an internal error \n{1}",
                                              fname, format_xml_error(xmlGetLastError())));
  }
  xmlResetLastError();
  if (context)
    context->myDoc = myDoc;
  return context;
}

IXmlDocumentHolder* DOMImplementation::
_load(const String& fname, ITraceMng* msg, const String& schemaname)
{
  _checkValid();
  std::unique_ptr<XmlDocumentHolderLibXml2> xml_doc(new XmlDocumentHolderLibXml2());
  if (xml_doc == NULL)
    throw Arcane::Exception::exception();
  xmlParserCtxt* context = NULL;
  char* encoding = NULL; // "UTF-8";// "ISO-8859-1";
  String xinclude_baseurl = String();
  if (!fname.empty())
    xinclude_baseurl = platform::getFileDirName(fname) + "/"; // Get the test directory
  bool _useSchema = true;
  ByteConstArrayView schemaData;
  if (schemaname.null()) {
    _useSchema = false;
  }
  if (fname.null()) {
    throw Arcane::XmlException(A_FUNCINFO, String::format("LIBXML2 XML file not defined '{0}'\n", fname));
  }
  if (_useSchema) {
    /// Verification de la validité du schema et parse du fichier xml
    if (msg)
      msg->info() << "LIBXML2 ---------------- the parser uses schema (1): " << schemaname;
    if ((context = _xmlSchemaValidationAndParsing(fname, msg, XML_PARSE_SCHEMA_FILE, schemaname,
                                                  schemaData)) == NULL) {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 XML validation schema '{0}' failed \n", schemaname));
    }
    if (msg) {
      msg->info() << String::format("LIBXML2 '{0}' validates\n", fname);
    }
  } else {
    // create the file parser context
    if ((context = xmlCreateFileParserCtxt(fname.localstr())) == NULL) {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 Could not create File parser context '{0}'\n{1}",
                                                fname, format_xml_error(xmlGetLastError())));
    }

    // The following is based on the implementation of xmlParseFile(), in xmlSAXParseFileWithData():
    if (encoding)
      context->encoding = reinterpret_cast<xmlChar*>(encoding);

    int options = context->options;
    options |=
            XML_PARSE_DTDLOAD | XML_PARSE_NOENT | XML_PARSE_DTDATTR | XML_PARSE_DTDVALID | XML_PARSE_NOBLANKS;
    if (xinclude_baseurl.null())
      options &= ~XML_PARSE_XINCLUDE;
    else
      options |= XML_PARSE_XINCLUDE;
    xmlCtxtUseOptions(context, options);

    if (!xmlParseDocument(context)) {
      ARCANE_ASSERT((context->myDoc != NULL && context->myDoc->URL == NULL), ("Inconsistent initial state"));

      if (!xinclude_baseurl.null()) {
        std::unique_ptr<xmlChar> URL(new xmlChar[xinclude_baseurl.len() + 1]);
        if (URL == NULL)
          throw Arcane::Exception::exception();
        context->myDoc->URL = URL.get();
        strcpy(const_cast<char*>(reinterpret_cast<const char*>(context->myDoc->URL)),
               xinclude_baseurl.localstr()); // change the current directory to the test directory. The method
                                             // xmlXIncludeProcess need this
        const int retcode = xmlXIncludeProcess(context->myDoc); // 0 if no substitution were done, -1 if some
                                                                // processing failed or the number of
                                                                // substitutions done.
        URL.release();
        context->myDoc->URL = NULL;
        if (retcode < 0) {
          throw Arcane::XmlException(A_FUNCINFO,
                                     String::format("LIBXML2 XInclude processing failed '{0}'\n{1}", fname,
                                                    format_xml_error(xmlGetLastError())));
        }
      }
    } else {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 xmlParseDocument failed 1 '{0}'\n{1}", fname,
                                                format_xml_error(xmlGetLastError())));
    }
  }
  // We just defined an LIBXML2 document . Convert to our data-structure...
  LIBXML2_Document* doc =
          static_cast<LIBXML2_Document*>(WrapXML2Node(NULL, reinterpret_cast<xmlNode*>(context->myDoc)));
  xml_doc->assignNode(reinterpret_cast<NodePrv*>(doc));
  doc->Impl_(context->myDoc);
  doc->Context_(context);
  return xml_doc.release();
}

IXmlDocumentHolder* DOMImplementation::
_load(const String& fname, ITraceMng* msg, const String& schemaname,
      ByteConstArrayView schemaData)
{
  _checkValid();
  std::unique_ptr<XmlDocumentHolderLibXml2> xml_doc(new XmlDocumentHolderLibXml2());
  if (xml_doc == NULL)
    throw Arcane::Exception::exception();
  xmlParserCtxt* context = NULL;
  String aErrorMessage;
  LIBXML2_PartialLoad pl = { getDomImplementation(), aErrorMessage };
  xmlSetStructuredErrorFunc(reinterpret_cast<void*>(&pl), LIBXML2_XMLStructuredHandler);
  char* encoding = NULL; // "UTF-8";// "ISO-8859-1";
  String xinclude_baseurl = String();
  if (!fname.empty())
    xinclude_baseurl = platform::getFileDirName(fname) + "/"; // Get the test directory
  bool _useSchema = true;
  if (schemaData.empty()) {
    _useSchema = false;
  }
  if (fname.null()) {
    throw Arcane::XmlException(A_FUNCINFO, String::format("LIBXML2 XML file not defined '{0}'\n", fname));
  }
  if (_useSchema) {
    /// Verification de la validité du schema et parse du fichier xml
    if (msg) {
      msg->info() << "LIBXML2 ---------------- the parser uses schema (2): " << schemaname;
    }
    if ((context = _xmlSchemaValidationAndParsing(fname, msg, XML_PARSE_SCHEMA_MEMORY, schemaname,
                                                  schemaData)) == NULL) {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 XML validation schema '{0}' failed \n", schemaname));
    }
    if (msg) {
      msg->info() << String::format("LIBXML2 '{0}' validates", fname);
    }
  } else {
    // create the file parser context
    if ((context = xmlCreateFileParserCtxt(fname.localstr())) == NULL) {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 xmlCreateFileParserCtxt failed '{0}'\n{1}", fname,
                                                format_xml_error(xmlGetLastError())));
    }
    // The following is based on the implementation of xmlParseFile(), in xmlSAXParseFileWithData():
    if (encoding)
      context->encoding = reinterpret_cast<xmlChar*>(encoding);
    int options = context->options;
    options |= XML_PARSE_DTDLOAD | XML_PARSE_NOENT | XML_PARSE_DTDATTR | XML_PARSE_DTDVALID;
    if (xinclude_baseurl.null())
      options &= ~XML_PARSE_XINCLUDE;
    else
      options |= XML_PARSE_XINCLUDE;
    xmlCtxtUseOptions(context, options);
    if (!xmlParseDocument(context)) {
      ARCANE_ASSERT((context->myDoc != NULL && context->myDoc->URL == NULL), ("Inconsistent initial state"));
      if (!xinclude_baseurl.null()) {
        std::unique_ptr<xmlChar> URL(new xmlChar[xinclude_baseurl.len() + 1]);
        if (URL == NULL)
          throw Arcane::Exception::exception();
        context->myDoc->URL = URL.get();
        strcpy(const_cast<char*>(reinterpret_cast<const char*>(context->myDoc->URL)),
               xinclude_baseurl.localstr()); // change the current directory to the test directory. The method
                                             // xmlXIncludeProcess need this
        const int retcode = xmlXIncludeProcess(context->myDoc); // 	0 if no substitution were done, -1 if some
                                                                // processing failed or the number of
                                                                // substitutions done.
        URL.release();
        context->myDoc->URL = NULL;
        if (retcode < 0) {
          throw Arcane::XmlException(A_FUNCINFO,
                                     String::format("LIBXML2 XInclude processing failed '{0}'\n{1}", fname,
                                                    format_xml_error(xmlGetLastError())));
        }
      }
    } else {
      throw Arcane::XmlException(A_FUNCINFO,
                                 String::format("LIBXML2 xmlParseDocument failed 2 '{0}'\n{1}", fname,
                                                format_xml_error(xmlGetLastError())));
    }
  }
  // We just defined an LIBXML2 document . Convert to our data-structure...
  LIBXML2_Document* doc =
          static_cast<LIBXML2_Document*>(WrapXML2Node(NULL, reinterpret_cast<xmlNode*>(context->myDoc)));
  xml_doc->assignNode(reinterpret_cast<NodePrv*>(doc));
  doc->Impl_(context->myDoc);
  doc->Context_(context);
  return xml_doc.release();
}

IXmlDocumentHolder* DOMImplementation::
_load(ByteConstArrayView buffer, const String& fname, ITraceMng* trace)
{
  ARCANE_UNUSED(trace);
  _checkValid();
  std::unique_ptr<XmlDocumentHolderLibXml2> xml_doc(new XmlDocumentHolderLibXml2());
  if (xml_doc == NULL)
    throw Arcane::Exception::exception();
  if (buffer.empty())
    return xml_doc.release();
  xmlParserCtxt* context = NULL;
  String aErrorMessage;
  LIBXML2_PartialLoad pl = { getDomImplementation(), aErrorMessage };
  xmlSetStructuredErrorFunc(reinterpret_cast<void*>(&pl), LIBXML2_XMLStructuredHandler);
  char* encoding = NULL; // "UTF-8";// "ISO-8859-1";

  String xinclude_baseurl = String();
  if (!fname.empty())
    xinclude_baseurl = platform::getFileDirName(fname) + "/"; // Get the test directory

  // lecture en memoire du document
  xmlResetLastError();
  const XMLByte* src = reinterpret_cast<const XMLByte*>(buffer.begin());
  context = xmlCreateMemoryParserCtxt(
          reinterpret_cast<const char*>(src),
          buffer.size()); // tous les champs non utilisés sont initialisés à 0 lors de la création
  if (!context) {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 Could not create parser context '{0}'\n{1}", fname,
                                              format_xml_error(xmlGetLastError())));
  }
  if (encoding)
    context->encoding = reinterpret_cast<xmlChar*>(encoding);
  int options = context->options;
  options |= XML_PARSE_DTDLOAD | XML_PARSE_NOENT | XML_PARSE_DTDATTR | XML_PARSE_DTDVALID;
  if (xinclude_baseurl.null())
    options &= ~XML_PARSE_XINCLUDE;
  else
    options |= XML_PARSE_XINCLUDE;

  xmlCtxtUseOptions(context, options);
  if (!xmlParseDocument(context)) {
    ARCANE_ASSERT((context->myDoc != NULL && context->myDoc->URL == NULL), ("Inconsistent initial state"));
    if (!xinclude_baseurl.null()) {
      std::unique_ptr<xmlChar> URL(new xmlChar[xinclude_baseurl.len() + 1]);
      if (URL == NULL)
        throw Arcane::Exception::exception();
      context->myDoc->URL = URL.get();
      strcpy(const_cast<char*>(reinterpret_cast<const char*>(context->myDoc->URL)),
             xinclude_baseurl.localstr()); // change the current directory to the test directory. The method
                                           // xmlXIncludeProcess need this
      const int retcode = xmlXIncludeProcess(context->myDoc); // 	0 if no substitution were done, -1 if some
                                                              // processing failed or the number of
                                                              // substitutions done.
      URL.release();
      context->myDoc->URL = NULL;
      if (retcode < 0) {
        throw Arcane::XmlException(A_FUNCINFO,
                                   String::format("LIBXML2 XInclude processing failed '{0}'\n{1}", fname,
                                                  format_xml_error(xmlGetLastError())));
      }
    }
  } else {
    throw Arcane::XmlException(A_FUNCINFO,
                               String::format("LIBXML2 xmlParseDocument failed 3 '{0}'{1}\n", fname,
                                              format_xml_error(xmlGetLastError())));
  }
  // We just defined an LIBXML2 document . Convert to our data-structure...
  LIBXML2_Document* doc =
          static_cast<LIBXML2_Document*>(WrapXML2Node(NULL, reinterpret_cast<xmlNode*>(context->myDoc)));
  xml_doc->assignNode(reinterpret_cast<NodePrv*>(doc));
  doc->Impl_(context->myDoc);
  doc->Context_(context);
  return xml_doc.release();
}

/*---------------------------------------------------------------------------*/

void DOMImplementation::
_save(std::ostream& ostr, const Document& document, int indent_level)
{
  bool indented = (indent_level == 1);
  LIBXML2_DOMWriter dw(indented);
  StringBuilder text;

  dw.writeDocument(NULL, reinterpret_cast<LIBXML2_Document*>(document._impl()), text);
  int len = text.toString().len();
  if (len > 1) {
    ostr.write(text.toString().localstr(), len);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/* Node implementation																												*/
/*---------------------------------------------------------------------------*/

Node::
Node()
: m_p(0)
{}

Node::
Node(NodePrv* prv)
: m_p(prv)
{}

Node::
Node(const Node& from)
: m_p(from.m_p)
{}

const Node& Node::
operator=(const Node& from)
{
  _assign(from);
  return (*this);
}

Node::
~Node() {}

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
  return impl(m_p)->nodeType();
}

Node Node::
firstChild() const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->firstChild();
  return cvt(n);
}

Node Node::
lastChild() const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->lastChild();
  return cvt(n);
}

Node Node::
previousSibling() const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->previousSibling();
  return cvt(n);
}

Node Node::
nextSibling() const
{
  LIBXML2_Node* n = impl(m_p)->nextSibling();
  return cvt(n);
}

Node Node::
parentNode() const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->parentNode();
  return cvt(n);
}

NodeList Node::
childNodes() const
{
  _checkValid();
  LIBXML2_NodeList* n = impl(m_p)->childNodes();
  return cvt(n);
}

DOMString Node::
nodeName() const
{
  _checkValid();
  return impl(m_p)->nodeName();
}

NamedNodeMap Node::
attributes() const
{
  LIBXML2_Node* node = impl(m_p);
  LIBXML2_Element* aElement = dynamic_cast<LIBXML2_Element*>(node);
  LIBXML2_NamedNodeMap* NamedNodeMap = aElement->attributes();
  return cvt(NamedNodeMap);
}

Document Node::
ownerDocument() const
{
  _checkValid();
  LIBXML2_Document* d = impl(m_p)->ownerDocument();
  return cvt(d);
}

DOMString Node::
nodeValue() const
{
  _checkValid();
  return impl(m_p)->nodeValue();
}

void Node::
nodeValue(const DOMString& str) const
{
  _checkValid();
  impl(m_p)->nodeValue(str);
}

void Node::
_assign(const Node& node)
{
  m_p = node.m_p;
}

Node Node::
insertBefore(const Node& new_child, const Node& ref_child) const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->insertBefore(impl(new_child._impl()), impl(ref_child._impl()));
  ;
  return cvt(n);
}

Node Node::
replaceChild(const Node& new_child, const Node& old_child) const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->replaceChild(impl(new_child._impl()), impl(old_child._impl()));
  return cvt(n);
}

Node Node::
removeChild(const Node& old_child) const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->removeChild(impl(old_child._impl()));
  return cvt(n);
}

Node Node::
appendChild(const Node& new_child) const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->appendChild(impl(new_child._impl()));
  return cvt(n);
}

bool Node::
hasChildNodes() const
{
  _checkValid();
  return impl(m_p)->hasChildNodes();
}

Node Node::
cloneNode(bool deep) const
{
  _checkValid();
  LIBXML2_Node* n = impl(m_p)->cloneNode(deep);
  return cvt(n);
}

DOMString Node::
prefix() const
{
  _checkValid();
  return impl(m_p)->prefix();
}

void Node::
prefix(const DOMString& new_prefix) const
{
  _checkValid();
  impl(m_p)->prefix(new_prefix);
}

void Node::
normalize() const
{
  _checkValid();
  impl(m_p)->normalize();
}

bool Node::
isSupported(const DOMString& feature, const DOMString& version) const
{
  _checkValid();
  return impl(m_p)->isSupported(feature, version);
}

DOMString Node::
namespaceURI() const
{
  _checkValid();
  return impl(m_p)->namespaceURI();
}

DOMString Node::
localName() const
{
  _checkValid();
  return impl(m_p)->localName();
}

DOMString Node::
baseURI() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

DOMString Node::
textContent() const
{
  _checkValid();
  return impl(m_p)->nodeValue();
}

void Node::
textContent(const DOMString& value) const
{
  _checkValid();
  impl(m_p)->nodeValue(value);
  return;
}

bool Node::
isSameNode(const Node& node) const
{
  return (this == &node);
}

#ifndef ARCANE_USE_LIBXML2
UShort Node::
compareTreePosition(const Node& other) const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
Node Node::
getInterface(const DOMString& feature) const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif

bool Node::
isEqualNode(const Node& other) const
{
  return (this == &other);
}

bool Node::
isDefaultNamespace(const DOMString& namespace_uri) const
{
  ARCANE_UNUSED(namespace_uri);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#ifndef ARCANE_USE_LIBXML2
DOMString Node::
lookupNamespacePrefix(const DOMString& namespace_uri, bool use_default) const
{
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(use_default);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
DOMString Node::
lookupNamespaceURI(const DOMString& prefix) const
{
  ARCANE_UNUSED(prefix);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

DOMObject Node::
setUserData(const DOMString& key, const DOMObject& data, const UserDataHandler& handler) const
{
  ARCANE_UNUSED(key);
  ARCANE_UNUSED(data);
  ARCANE_UNUSED(handler);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

DOMObject Node::
getUserData(const DOMString& key) const
{
  ARCANE_UNUSED(key);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

bool
operator==(const Node& n1, const Node& n2)
{
  return impl(n1.m_p) == impl(n2.m_p);
}

bool
operator!=(const Node& n1, const Node& n2)
{
  return !operator==(n1, n2);
}

/*---------------------------------------------------------------------------*/
/* Character implementation																										*/
/*---------------------------------------------------------------------------*/

CharacterData::
CharacterData(CharacterDataPrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

CharacterData::
CharacterData()
: Node()
{}

CharacterData::
CharacterData(const CharacterData& node)
: Node(node)
{}

CharacterData::
CharacterData(const Node& node)
: Node()
{
  CDATASectionPrv* ni = reinterpret_cast<CDATASectionPrv*>(node._impl());
  if (ni && (impl(ni)->nodeType() == CDATA_SECTION_NODE))
    _assign(node);
}

CharacterDataPrv* CharacterData::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_CharacterData*>(impl(m_p)));
}

/*---------------------------------------------------------------------------*/
/* Text implementation																												*/
/*---------------------------------------------------------------------------*/

Text::
Text(TextPrv* prv)
: CharacterData(reinterpret_cast<CharacterDataPrv*>(impl(prv)))
{}

Text::
Text()
: CharacterData()
{}

Text::
Text(const Text& node)
: CharacterData(node)
{}

Text::
Text(const Node& node)
: CharacterData()
{
  TextPrv* ni = reinterpret_cast<TextPrv*>(node._impl());
  if (ni && (impl(ni)->nodeType() == TEXT_NODE))
    _assign(node);
}

TextPrv* Text::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_Text*>(impl(m_p)));
}

/*---------------------------------------------------------------------------*/
/* Document implementation																										*/
/*---------------------------------------------------------------------------*/

Document::
Document() {}

Document::
Document(DocumentPrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

Document::
Document(const Node& node)
: Node()
{
  DocumentPrv* ni = reinterpret_cast<DocumentPrv*>(node._impl());
  if (ni && (impl(ni)->nodeType() == DOCUMENT_NODE))
    _assign(node);
}

DocumentPrv* Document::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_Document*>(impl(m_p)));
}

DocumentType Document::
doctype() const
{
  _checkValid();
  LIBXML2_DocumentType* dt = reinterpret_cast<LIBXML2_DocumentType*>(impl(_impl())->doctype());
  return cvt(dt);
}

DOMImplementation Document::
implementation() const
{
  _checkValid();
  return cvt(getDomImplementation());
}

Element Document::
documentElement() const
{
  _checkValid();
  LIBXML2_Element* el = impl(_impl())->documentElement();
  return cvt(el);
}

Element Document::
createElement(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Element* el = impl(_impl())->createElement(name);
  return cvt(el);
}

DocumentFragment Document::
createDocumentFragment() const
{
  _checkValid();
  LIBXML2_DocumentFragment* df = impl(_impl())->createDocumentFragment();
  return cvt(df);
}

Text Document::
createTextNode(const DOMString& data) const
{
  _checkValid();
  LIBXML2_Text* tn = impl(_impl())->createTextNode(data);
  return cvt(tn);
}

Comment Document::
createComment(const DOMString& data) const
{
  _checkValid();
  LIBXML2_Comment* cn = impl(_impl())->createComment(data);
  return cvt(cn);
}

CDATASection Document::
createCDATASection(const DOMString& data) const
{
  _checkValid();
  LIBXML2_CDATASection* cds = impl(_impl())->createCDATASection(data);
  return cvt(cds);
}

ProcessingInstruction Document::
createProcessingInstruction(const DOMString& target, const DOMString& data) const
{
  _checkValid();
  LIBXML2_ProcessingInstruction* pi = impl(_impl())->createProcessingInstruction(target, data);
  return cvt(pi);
}

Attr Document::
createAttribute(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Attr* at = impl(_impl())->createAttribute(name);
  return cvt(at);
}

EntityReference Document::
createEntityReference(const DOMString& name) const
{
  _checkValid();
  LIBXML2_EntityReference* er = impl(_impl())->createEntityReference(name);
  return cvt(er);
}

NodeList Document::
getElementsByTagName(const DOMString& tagname) const
{
  _checkValid();
  LIBXML2_NodeList* nl = impl(_impl())->getElementsByTagName(tagname);
  return cvt(nl);
}

Node Document::
importNode(const Node& imported_node, bool deep) const
{
  _checkValid();
  LIBXML2_Node* n = impl(_impl())->importNode(impl(toNodePrv(imported_node)), deep);
  return cvt(n);
}

Element Document::
createElementNS(const DOMString& namespace_uri, const DOMString& qualifiedname) const
{
  _checkValid();
  LIBXML2_Element* el = impl(_impl())->createElementNS(namespace_uri, qualifiedname);
  return cvt(el);
}

Attr Document::
createAttributeNS(const DOMString& namespace_uri, const DOMString& qualifiedname) const
{
  _checkValid();
  LIBXML2_Attr* at = impl(_impl())->createAttributeNS(namespace_uri, qualifiedname);
  return cvt(at);
}

NodeList Document::
getElementsByTagNameNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  _checkValid();
  LIBXML2_NodeList* nl = impl(_impl())->getElementsByTagNameNS(namespace_uri, localname);
  return cvt(nl);
}

Element Document::
getElementById(const DOMString& element_id) const
{
  _checkValid();
  LIBXML2_Element* el = impl(_impl())->getElementById(element_id);
  return cvt(el);
}

DOMString Document::
actualEncoding() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

void Document::
actualEncoding(const DOMString& value) const
{
  ARCANE_UNUSED(value);
  _checkValid();
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
  ARCANE_UNUSED(value);
  _checkValid();
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
  ARCANE_UNUSED(value);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

bool Document::
strictErrorChecking() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

void Document::
strictErrorChecking(bool value) const
{
  ARCANE_UNUSED(value);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#ifndef ARCANE_USE_LIBXML2
DOMString Document::
version() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
void Document::
version(const DOMString& value) const
{
  ARCANE_UNUSED(value);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif

void Document::
documentURI(const DOMString& document_uri) const
{
  ARCANE_UNUSED(document_uri);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

DOMString Document::
documentURI() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

Node Document::
adoptNode(const Node& source) const
{
  ARCANE_UNUSED(source);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

void Document::
normalizeDocument()
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

Node Document::
renameNode(const Node& node, const DOMString& namespace_uri, const DOMString& name)
{
  ARCANE_UNUSED(node);
  ARCANE_UNUSED(namespace_uri);
  ARCANE_UNUSED(name);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* DocumentFragment implementation																						*/
/*---------------------------------------------------------------------------*/

DocumentFragment::
DocumentFragment()
: Node()
{}

DocumentFragment::
DocumentFragment(DocumentFragmentPrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/* Comment implementation																											*/
/*---------------------------------------------------------------------------*/

Comment::
Comment()
: CharacterData()
{}

Comment::
Comment(CommentPrv* prv)
: CharacterData(reinterpret_cast<CharacterDataPrv*>(impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/* CDATASection implementation																								*/
/*---------------------------------------------------------------------------*/

CDATASection::
CDATASection()
: Text()
{}

CDATASection::
CDATASection(CDATASectionPrv* prv)
: Text(reinterpret_cast<TextPrv*>(impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/* EntityReference implementation																							*/
/*---------------------------------------------------------------------------*/

EntityReference::
EntityReference()
: Node()
{}

EntityReference::
EntityReference(EntityReferencePrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

/*---------------------------------------------------------------------------*/
/* NodeList implementation																										*/
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
  LIBXML2_Node* n = impl(m_p)->item(index);
  return cvt(n);
}

ULong NodeList::
length() const
{
  _checkValid();
  return impl(m_p)->length();
}

/*---------------------------------------------------------------------------*/
/* CharacterData implementation																								*/
/*---------------------------------------------------------------------------*/

DOMString CharacterData::
data() const
{
  _checkValid();
  return impl(_impl())->Data();
}

void CharacterData::
data(const DOMString& value) const
{
  _checkValid();
  impl(_impl())->Data(value);
}

ULong CharacterData::
length() const
{
  _checkValid();
  return impl(_impl())->length();
}

DOMString CharacterData::
substringData(ULong offset, ULong count) const
{
  _checkValid();
  return impl(_impl())->substringdata(offset, count);
}

void CharacterData::
appendData(const DOMString& arg) const
{
  _checkValid();
  impl(_impl())->appenddata(arg);
}

void CharacterData::
insertData(ULong offset, const DOMString& arg) const
{
  _checkValid();
  impl(_impl())->insertdata(offset, arg);
}

void CharacterData::
deleteData(ULong offset, ULong count) const
{
  _checkValid();
  impl(_impl())->deletedata(offset, count);
}

void CharacterData::
replaceData(ULong offset, ULong count, const DOMString& arg) const
{
  _checkValid();
  impl(_impl())->replacedata(offset, count, arg);
}

/*---------------------------------------------------------------------------*/
/* Attr implementation																												*/
/*---------------------------------------------------------------------------*/

Attr::
Attr()
: Node()
{}

Attr::
Attr(AttrPrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

Attr::
Attr(const Attr& f)
: Node((const Node&)f)
{}

Attr::
Attr(const Node& node)
: Node()
{
  AttrPrv* ni = reinterpret_cast<AttrPrv*>(node._impl());
  if (ni && (impl(ni)->nodeType() == ATTRIBUTE_NODE))
    _assign(node);
}

Attr::
~Attr() {}

AttrPrv* Attr::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_Attr*>(impl(m_p)));
}

DOMString Attr::
name() const
{
  _checkValid();
  return impl(_impl())->name();
}

bool Attr::
specified() const
{
  _checkValid();
  return impl(_impl())->specified();
}

DOMString Attr::
value() const
{
  _checkValid();
  return impl(_impl())->value();
}

void Attr::
value(const DOMString& str) const
{
  _checkValid();
  impl(_impl())->value(str);
}

Element Attr::
ownerElement() const
{
  _checkValid();
  LIBXML2_Element* el = impl(_impl())->ownerElement();
  return cvt(el);
}

/*---------------------------------------------------------------------------*/
/* Element implementation																											*/
/*---------------------------------------------------------------------------*/

Element::
Element()
: Node()
{}

Element::
Element(ElementPrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

Element::
Element(const Node& node)
: Node()
{
  ElementPrv* ni = reinterpret_cast<ElementPrv*>(node._impl());
  if (ni && (impl(ni)->nodeType() == ELEMENT_NODE))
    _assign(node);
}

Element::
Element(const Element& node)
: Node(node)
{}

ElementPrv* Element::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_Element*>(impl(m_p)));
}

DOMString Element::
tagName() const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  return el->tagName();
}

DOMString Element::
getAttribute(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  return el->getAttribute(name);
}

void Element::
setAttribute(const DOMString& name, const DOMString& value) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  el->setAttribute(name, value);
}

void Element::
removeAttribute(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  el->removeAttribute(name);
}

Attr Element::
getAttributeNode(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_Attr* attr = el->getAttributeNode(name);
  return cvt(attr);
}

Attr Element::
setAttributeNode(const Attr& new_attr) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_Attr* attr = el->setAttributeNode(impl(new_attr._impl()));
  return cvt(attr);
}

Attr Element::
removeAttributeNode(const Attr& old_attr) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_Attr* attr = el->removeAttributeNode(impl(old_attr._impl()));
  return cvt(attr);
}

NodeList Element::
getElementsByTagName(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_NodeList* ndl = el->getElementsByTagName(name);
  return NodeList(reinterpret_cast<NodeListPrv*>(ndl));
}

DOMString Element::
getAttributeNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  return el->getAttributeNS(namespace_uri, localname);
}

void Element::
setAttributeNS(const DOMString& namespace_uri, const DOMString& localname,
               const DOMString& value) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  el->setAttributeNS(namespace_uri, localname, value);
}

void Element::
removeAttributeNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  el->removeAttributeNS(namespace_uri, localname);
}

Attr Element::
getAttributeNodeNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_Attr* attr = el->getAttributeNodeNS(namespace_uri, localname);
  return cvt(attr);
}

Attr Element::
setAttributeNodeNS(const Attr& new_attr) const
{
  _checkValid();
  new_attr._checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_Attr* attr = el->setAttributeNodeNS(impl(new_attr._impl()));
  return cvt(attr);
}

NodeList Element::
getElementsByTagNameNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  LIBXML2_NodeList* ndl = el->getElementsByTagNameNS(namespace_uri, localname);
  return NodeList(cvt(ndl));
}

bool Element::
hasAttribute(const DOMString& name) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  return el->hasAttribute(name);
}

bool Element::
hasAttributeNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  _checkValid();
  LIBXML2_Element* el = reinterpret_cast<LIBXML2_Element*>(_impl());
  return el->hasAttributeNS(namespace_uri, localname);
}

/*---------------------------------------------------------------------------*/
/* Text implementation																												*/
/*---------------------------------------------------------------------------*/

Text Text::
splitText(ULong offset) const
{
  _checkValid();
  LIBXML2_Text* txt = reinterpret_cast<LIBXML2_Text*>(_impl());
  LIBXML2_Text* txt2 = txt->splitText(offset);
  return Text(cvt(txt2));
}

#ifndef ARCANE_USE_LIBXML2
bool Text::
isWhiteSpaceInElementContent() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif

DOMString Text::
wholeText() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

Text Text::
replaceWholeText(const DOMString& content) const
{
  ARCANE_UNUSED(content);
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* DocumentType implementation			  																				*/
/*---------------------------------------------------------------------------*/

DocumentType::
DocumentType()
: Node()
{}

DocumentType::
DocumentType(DocumentTypePrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

DocumentTypePrv* DocumentType::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_DocumentType*>(m_p));
}

DOMString DocumentType::
name() const
{
  _checkValid();
  return impl(_impl())->name();
}

NamedNodeMap DocumentType::
entities() const
{
  _checkValid();
  LIBXML2_NamedNodeMap* nnodeMap = impl(_impl())->entities();
  return NamedNodeMap(reinterpret_cast<NamedNodeMapPrv*>(nnodeMap));
}

NamedNodeMap DocumentType::
notations() const
{
  _checkValid();
  LIBXML2_NamedNodeMap* nnodeMap = impl(_impl())->notations();
  return NamedNodeMap(reinterpret_cast<NamedNodeMapPrv*>(nnodeMap));
}

DOMString DocumentType::
publicId() const
{
  _checkValid();
  return impl(_impl())->publicId();
}

DOMString DocumentType::
systemId() const
{
  _checkValid();
  return impl(_impl())->systemId();
}

DOMString DocumentType::
internalSubset() const
{
  _checkValid();
  return impl(_impl())->internalSubset();
}

/*---------------------------------------------------------------------------*/
/* Notation implementation					  																				*/
/*---------------------------------------------------------------------------*/

NotationPrv* Notation::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_Notation*>(impl(m_p)));
}

DOMString Notation::
publicId() const
{
  _checkValid();
  return impl(_impl())->publicId();
}

DOMString Notation::
systemId() const
{
  _checkValid();
  return impl(_impl())->systemId();
}

/*---------------------------------------------------------------------------*/
/* Entity implementation			   		  																				*/
/*---------------------------------------------------------------------------*/

EntityPrv* Entity::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_Entity*>(impl(m_p)));
}

DOMString Entity::
publicId() const
{
  _checkValid();
  return impl(_impl())->publicId();
}

DOMString Entity::
systemId() const
{
  _checkValid();
  return impl(_impl())->systemId();
}

DOMString Entity::
notationName() const
{
  _checkValid();
  return impl(_impl())->notationName();
}

#ifndef ARCANE_USE_LIBXML2
DOMString Entity::
actualEncoding() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
void Entity::
actualEncoding(const DOMString& value) const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
DOMString Entity::
encoding() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
void Entity::
encoding(const DOMString& value) const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
DOMString Entity::
version() const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif
#ifndef ARCANE_USE_LIBXML2
void Entity::
version(const DOMString& value) const
{
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}
#endif

/*---------------------------------------------------------------------------*/
/* ProcessingInstruction implementation																				*/
/*---------------------------------------------------------------------------*/

ProcessingInstruction::
ProcessingInstruction()
: Node()
{}

ProcessingInstruction::
ProcessingInstruction(ProcessingInstructionPrv* prv)
: Node(reinterpret_cast<NodePrv*>(impl(prv)))
{}

ProcessingInstructionPrv* ProcessingInstruction::
_impl() const
{
  return cvt(reinterpret_cast<LIBXML2_ProcessingInstruction*>(impl(m_p)));
}

DOMString ProcessingInstruction::
target() const
{
  _checkValid();
  return impl(_impl())->target();
}

DOMString ProcessingInstruction::
data() const
{
  _checkValid();
  return impl(_impl())->Data();
}

void ProcessingInstruction::
data(const DOMString& value) const
{
  _checkValid();
  impl(_impl())->Data(value);
}

/*---------------------------------------------------------------------------*/
/* NamedNodeMap implementation																								*/
/*---------------------------------------------------------------------------*/

NamedNodeMap::
NamedNodeMap() {}

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
~NamedNodeMap() {}

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
  return impl(m_p)->length();
}

Node NamedNodeMap::
getNamedItem(const DOMString& name) const
{
  if (_null())
    return Node();
  LIBXML2_Node* n = impl(m_p)->getNamedItem(name);
  return cvt(n);
}

Node NamedNodeMap::
setNamedItem(const Node& arg) const
{
  if (_null() || arg._null())
    return Node();
  LIBXML2_Node* n = impl(_impl())->setNamedItem(impl(arg._impl()));
  return Node(reinterpret_cast<NodePrv*>(n));
}

Node NamedNodeMap::
removeNamedItem(const DOMString& name) const
{
  if (_null())
    return Node();
  LIBXML2_Node* n = impl(_impl())->removeNamedItem(name);
  return Node(cvt(n));
}

Node NamedNodeMap::
item(ULong index) const
{
  if (_null())
    return Node();
  LIBXML2_Node* n = impl(m_p)->item(index);
  return Node(reinterpret_cast<NodePrv*>(n));
}

Node NamedNodeMap::
getNamedItemNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  if (_null())
    return Node();
  LIBXML2_Node* n = impl(_impl())->getNamedItemNS(namespace_uri, localname);
  return Node(reinterpret_cast<NodePrv*>(n));
}

Node NamedNodeMap::
setNamedItemNS(const Node& arg) const
{
  if (_null())
    return Node();
  if (arg._null())
    return Node();
  LIBXML2_Node* n = impl(_impl())->setNamedItemNS(impl(arg._impl()));
  return Node(cvt(n));
}

Node NamedNodeMap::
removeNamedItemNS(const DOMString& namespace_uri, const DOMString& localname) const
{
  if (_null())
    return Node();
  LIBXML2_Node* n = impl(_impl())->removeNamedItemNS(namespace_uri, localname);
  return Node(cvt(n));
}

/*---------------------------------------------------------------------------*/
/* DOMImplementationSource implementation																			*/
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
handle(UShort operation, const DOMString& key, const DOMObject& data, const Node& src,
       const Node& dest) const
{
  ARCANE_UNUSED(operation);
  ARCANE_UNUSED(key);
  ARCANE_UNUSED(data);
  ARCANE_UNUSED(src);
  ARCANE_UNUSED(dest);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* DOMWriter implementation																										*/
/*---------------------------------------------------------------------------*/

DOMWriter::
DOMWriter()
: m_p(0)
{}

DOMWriter::
DOMWriter(DOMWriterPrv* p)
: m_p(p)
{}

DOMWriter::
DOMWriter(const DOMWriter& from)
: m_p(from.m_p)
{}

const DOMWriter& DOMWriter::
operator=(const DOMWriter& from)
{
  m_p = from.m_p;
  return (*this);
}

DOMWriter::
~DOMWriter() {}

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
  _checkValid();
  throw NotImplementedException(A_FUNCINFO);
}

void DOMWriter::
encoding(const String& encoding)
{
  ARCANE_UNUSED(encoding);
  throw NotImplementedException(A_FUNCINFO);
}

String DOMWriter::
encoding() const
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* DOMError implementation																										*/
/*---------------------------------------------------------------------------*/

DOMError::
DOMError()
: m_p(0)
{}

DOMError::
DOMError(DOMErrorPrv* p)
: m_p(p)
{}

DOMError::
DOMError(const DOMError& from)
: m_p(from.m_p)
{}

const DOMError& DOMError::
operator=(const DOMError& from)
{
  m_p = from.m_p;
  return (*this);
}

DOMError::
~DOMError() {}

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
  throw NotImplementedException(A_FUNCINFO);
}

DOMString DOMError::
message() const
{
  throw NotImplementedException(A_FUNCINFO);
}

DOMObject DOMError::
relatedException() const
{
  throw NotImplementedException(A_FUNCINFO);
}

DOMLocator DOMError::
location() const
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* DOMErrorHandler implementation																							*/
/*---------------------------------------------------------------------------*/

bool DOMErrorHandler::
handleError(const DOMError& error) const
{
  ARCANE_UNUSED(error);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* DOMLocator implementation					     																		*/
/*---------------------------------------------------------------------------*/

DOMLocator::
DOMLocator()
: m_p(0)
{}

DOMLocator::
DOMLocator(DOMLocatorPrv* p)
: m_p(p)
{}

DOMLocator::
DOMLocator(const DOMLocator& from)
: m_p(from.m_p)
{}

const DOMLocator& DOMLocator::
operator=(const DOMLocator& from)
{
  m_p = from.m_p;
  return (*this);
}

DOMLocator::
~DOMLocator() {}

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
  throw NotImplementedException(A_FUNCINFO);
}

long DOMLocator::
columnNumber() const
{
  throw NotImplementedException(A_FUNCINFO);
}

#ifndef ARCANE_USE_LIBXML2
long DOMLocator::
offset() const
{
  throw NotImplementedException(A_FUNCINFO);
}
#endif

#ifndef ARCANE_USE_LIBXML2
Node DOMLocator::
errorNode() const
{
  throw NotImplementedException(A_FUNCINFO);
}
#endif

DOMString DOMLocator::
uri() const
{
  return DOMString();
}

/*---------------------------------------------------------------------------*/
/* XPathEvaluator implementation			     																		*/
/*---------------------------------------------------------------------------*/

XPathExpression XPathEvaluator::
createExpression(const DOMString& expression, const XPathNSResolver& resolver) const
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
         const XPathNSResolver& resolver, UShort type, const XPathResult& result) const
{
  ARCANE_UNUSED(expression);
  ARCANE_UNUSED(context_node);
  ARCANE_UNUSED(resolver);
  ARCANE_UNUSED(type);
  ARCANE_UNUSED(result);
  throw NotImplementedException(A_FUNCINFO);
}

XPathResult XPathEvaluator::
evaluateExpression(const XPathExpression& expression, const Node& context_node, UShort type,
                   const XPathResult& result) const
{
  ARCANE_UNUSED(expression);
  ARCANE_UNUSED(context_node);
  ARCANE_UNUSED(type);
  ARCANE_UNUSED(result);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* XPathNSResolver implementation			     																		*/
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
/* XPathSetIterator implementation			     																	*/
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
/* XPathNamespace implementation			     																		*/
/*---------------------------------------------------------------------------*/

Element XPathNamespace::
ownerElement() const
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/* LIBXML2_DOM implementation			     																				*/
/*---------------------------------------------------------------------------*/
LIBXML2_DOMImplementation*
getDomImplementation()
{
  return LIBXML2_DOMImplementation::sDOMImplementation;
}

LIBXML2_DOMImplementation::
~LIBXML2_DOMImplementation()
{
  LIBXML2_Document* _xDoc = GetDocument();
  if (_xDoc) {
    xmlParserCtxtPtr ctxt = _xDoc->Context_();
    if (ctxt) {
      if (ctxt->myDoc)
        xmlFreeDoc(ctxt->myDoc);
      if (ctxt)
        xmlFreeParserCtxt(ctxt);
    } else {
      _xmlDoc* myDoc = _xDoc->Impl_();
      if (myDoc)
        xmlFreeDoc(myDoc);
    }
    delete _xDoc;
    SetDocument(nullptr);
  }
}

std::unique_ptr<LIBXML2_DOMImplementation> sDom(new LIBXML2_DOMImplementation());
LIBXML2_DOMImplementation* LIBXML2_DOMImplementation::sDOMImplementation = sDom.release();

LIBXML2_Element*
LIBXML2_NewElement(LIBXML2_Document* doc, const String& nsURI, const String& elname)
{
  ARCANE_UNUSED(nsURI);
  ARCANE_UNUSED(elname);
  return new LIBXML2_Element(doc);
}

LIBXML2_Document*
LIBXML2_NewDocument(const String& nsURI)
{
  ARCANE_UNUSED(nsURI);
  return new LIBXML2_Document();
}

bool LIBXML2_DOMImplementation::
hasFeature(const String& feature, const String& version)
{
  return (feature == "Core" && (version == "1.0" || version == "2.0" || version == "3.0"));
}

LIBXML2_DocumentType* LIBXML2_DOMImplementation::
createDocumentType(const String& qualifiedname, const String& publicId,
                   const String& systemId)
{
  return new LIBXML2_DocumentType(NULL, qualifiedname, publicId, systemId);
}

LIBXML2_Document* LIBXML2_DOMImplementation::
createDocument(const String& namespaceURI, const String& qualifiedname,
               LIBXML2_DocumentType* doctype)
{
  const XMLCh* cqname = reinterpret_cast<const XMLCh*>(qualifiedname.utf16().begin());
  const XMLCh* cpos = wcschr(cqname, L':');
  if (cpos == NULL)
    cpos = cqname;
  else
    cpos++;
  ByteUniqueArray utf8_array(cpos[0]);
  String localName = String::fromUtf8(utf8_array);
  if (namespaceURI == String()) {
    throw Arcane::Exception::exception();
  }
  if (cpos - cqname == 3 && !wcsncmp(cqname, U("xml"), 3) &&
      qualifiedname != "http://www.w3.org/XML/1998/namespace") {
    throw Arcane::Exception::exception();
  }

  if (doctype && doctype->mDocument != NULL) {
    throw Arcane::Exception::exception();
  }
  LIBXML2_Document* doc = LIBXML2_NewDocument(namespaceURI);
  if (doc == NULL)
    throw Arcane::Exception::exception();
  if (doctype != NULL) {
    doctype->mDocument = doc;
    doctype->mDocumentIsAncestor = false;
    doc->add_ref();
    doc->insertBeforePrivate(doctype, NULL)->release_ref();
  }

  LIBXML2_Element* docel = new LIBXML2_Element(doc);
  if (docel == NULL)
    throw Arcane::Exception::exception();
  docel->mNamespaceURI = namespaceURI;
  docel->mNodeName = qualifiedname;
  docel->mLocalName = localName;

  doc->mNamespaceURI = namespaceURI;
  doc->mNodeName = qualifiedname;
  doc->mLocalName = localName;
  doc->appendChild(docel)->release_ref();
  doc->add_ref();
  _xmlDoc* myDoc = xmlNewDoc(reinterpret_cast<const xmlChar*>("1.0"));
  if (myDoc == NULL)
    throw Arcane::Exception::exception();
  doc->Impl_(myDoc);
  SetDocument(doc);
  return doc;
}

LIBXML2_Node::
LIBXML2_Node(LIBXML2_Document* aDocument)
: mParent(NULL)
, mDocumentIsAncestor(false)
, mDocument(aDocument)
, mNodeType(DOCUMENT_NODE)
{
  if (mDocument)
    mDocument->add_ref();
}

LIBXML2_Node::
~LIBXML2_Node()
{
  if (!mDocumentIsAncestor && mDocument)
    mDocument->release_ref();
  // Now start deleting the children...
  for (std::list<LIBXML2_Node*>::iterator i2(mNodeList.begin()); i2 != mNodeList.end(); i2++)
    delete (*i2);
}

String LIBXML2_Node::
nodeName()
{
  return mNodeName;
}

String LIBXML2_Node::
nodeValue()
{
  return mNodeValue;
}

void LIBXML2_Node::
nodeValue(const String& attr)
{
  mNodeValue = attr;
}

UInt16 LIBXML2_Node::
nodeType()
{
  return mNodeType;
}

LIBXML2_Node* LIBXML2_Node::
parentNode()
{
  if (mParent != NULL)
    mParent->add_ref();
  return mParent;
}

LIBXML2_NodeList* LIBXML2_Node::
childNodes()
{
  return new LIBXML2_NodeList(this);
}

LIBXML2_Node* LIBXML2_Node::
firstChild()
{
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++) {
    UInt16 type = (*i)->nodeType();
    if (type != LIBXML2_Node::ATTRIBUTE_NODE && type != LIBXML2_Node::NOTATION_NODE &&
        type != LIBXML2_Node::ENTITY_NODE) {
      (*i)->add_ref();
      return (*i);
    }
  }
  return NULL;
}

LIBXML2_Node* LIBXML2_Node::
lastChild()
{
  std::list<LIBXML2_Node*>::iterator i = mNodeList.end();
  while (i != mNodeList.begin()) {
    i--;
    UInt16 type = (*i)->nodeType();
    if (type != LIBXML2_Node::ATTRIBUTE_NODE && type != LIBXML2_Node::NOTATION_NODE &&
        type != LIBXML2_Node::ENTITY_NODE) {
      (*i)->add_ref();
      return (*i);
    }
  }
  return NULL;
}

LIBXML2_Node* LIBXML2_Node::
previousSibling()
{
  if (mParent == NULL)
    return NULL;
  if (nodeType() == LIBXML2_Node::ATTRIBUTE_NODE)
    return NULL;
  std::list<LIBXML2_Node*>::iterator i = mPositionInParent;
  while (true) {
    if (i == mParent->mNodeList.begin())
      return NULL;
    i--;
    UInt16 type = (*i)->nodeType();
    if (type != LIBXML2_Node::ATTRIBUTE_NODE && type != LIBXML2_Node::ENTITY_NODE &&
        type != LIBXML2_Node::NOTATION_NODE) {
      (*i)->add_ref();
      return (*i);
    }
  }
}

LIBXML2_Node* LIBXML2_Node::
nextSibling()
{
  if (mParent == NULL)
    return NULL;
  if (nodeType() == LIBXML2_Node::ATTRIBUTE_NODE)
    return NULL;
  std::list<LIBXML2_Node*>::iterator i = mPositionInParent;
  while (true) {
    i++;
    if (i == mParent->mNodeList.end())
      return NULL;
    UInt16 type = (*i)->nodeType();
    if (type != LIBXML2_Node::ATTRIBUTE_NODE && type != LIBXML2_Node::ENTITY_NODE &&
        type != LIBXML2_Node::NOTATION_NODE) {
      (*i)->add_ref();
      return (*i);
    }
  }
}

LIBXML2_NamedNodeMap* LIBXML2_Node::
attributes()
{
  return (LIBXML2_NamedNodeMap*)(new LIBXML2_EmptyNamedNodeMap()); // CHECK
}

LIBXML2_Document* LIBXML2_Node::
ownerDocument()
{
  if (mDocument != NULL)
    mDocument->add_ref();
  return mDocument;
}

void LIBXML2_Node::
updateDocumentAncestorStatus(bool aStatus)
{
  if (mDocumentIsAncestor == aStatus)
    return;
  if (aStatus && mDocument) {
    // We are now under the document's refcount...
    mDocument->release_ref();
  } else if (mDocument && !aStatus) {
    // The document is no longer sharing our refcount, so needs an explicit ref
    mDocument->add_ref();
  }
  mDocumentIsAncestor = aStatus;
  // Now tell our children...
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++)
    (*i)->updateDocumentAncestorStatus(aStatus);
}

LIBXML2_Node* LIBXML2_Node::
insertBefore(LIBXML2_Node* newChild, LIBXML2_Node* refChild)
{
  if (newChild == NULL)
    throw Arcane::Exception::exception();

  UInt16 type = newChild->nodeType();

  // Get rid of nodes which can't be added this way...
  if (type == LIBXML2_Node::ATTRIBUTE_NODE || type == LIBXML2_Node::DOCUMENT_NODE ||
      type == LIBXML2_Node::DOCUMENT_TYPE_NODE || type == LIBXML2_Node::NOTATION_NODE ||
      type == LIBXML2_Node::ENTITY_NODE)
    throw Arcane::Exception::exception();
  if (type == LIBXML2_Node::DOCUMENT_FRAGMENT_NODE) {
    if (newChild == NULL)
      throw Arcane::Exception::exception();
    // We skip the d.f. and recurse onto its children...
    for (std::list<LIBXML2_Node*>::iterator i = newChild->mNodeList.begin(); i != newChild->mNodeList.end();
         i++)
      insertBefore(*i, refChild)->release_ref();
    newChild->add_ref();
    return newChild;
  }
  return insertBeforePrivate(newChild, refChild);
}

LIBXML2_Node* LIBXML2_Node::
insertBeforePrivate(LIBXML2_Node* newChild, LIBXML2_Node* refChild)
{
  // Check the new child...
  if (newChild == NULL)
    throw Arcane::Exception::exception();
  // If there is a refchild, it must belong to us...
  if (refChild && refChild->mParent != this)
    throw Arcane::Exception::exception();
  if (newChild->mDocument != mDocument)
    throw Arcane::Exception::exception();
  if (newChild == refChild) {
    // It is already in the right place...
    newChild->add_ref();
    return newChild;
  }
  if (newChild->mParent != NULL) {
    // Check that it is not our ancestor
    LIBXML2_Node* n = this;
    while (n) {
      if (n == newChild)
        throw Arcane::Exception::exception();
      n = n->mParent;
    }
  }
  std::list<LIBXML2_Node*>::iterator posit;
  if (newChild->mParent != NULL) {
    ARCANE_ASSERT(!refChild || (refChild->mParent == this),
                  ("The child belongs to us, but isn't on the list!"));
    newChild->mParent->removeChild(newChild)->release_ref();
    ARCANE_ASSERT(!refChild || (refChild->mParent == this),
                  ("The child belongs to us, but isn't on the list!"));
  }
  // Just in case the remove failed.
  if (newChild->mParent != NULL)
    throw Arcane::Exception::exception();
  if (refChild != NULL) {
    posit = std::find(mNodeList.begin(), mNodeList.end(), refChild);
    if (posit == mNodeList.end()) {
      // The child belongs to us, but isn't on the list!
      ARCANE_ASSERT(refChild->mParent == this, ("The child belongs to us, but isn't on the list!"));
      throw Arcane::Exception::exception();
    }
  } else
    posit = mNodeList.end();
  // Update nodes' mDocumentIsAncestor...
  if (mDocumentIsAncestor)
    newChild->updateDocumentAncestorStatus(true);
  newChild->mParent = this;
  newChild->mPositionInParent = mNodeList.insert(posit, newChild);
  UInt32 i, rc = newChild->_libxml2_refcount;
  for (i = 0; i < rc; i++)
    add_ref();
  newChild->add_ref();
  return newChild;
}

LIBXML2_Node* LIBXML2_Node::
replaceChild(LIBXML2_Node* newChild, LIBXML2_Node* oldChild)
{
  if (newChild == oldChild) {
    oldChild->add_ref();
    return oldChild;
  }
  if (oldChild == NULL)
    throw Arcane::Exception::exception();
  insertBefore(newChild, oldChild)->release_ref();
  return removeChild(oldChild);
}

LIBXML2_Node* LIBXML2_Node::
removeChild(LIBXML2_Node* oldChild)
{
  if (oldChild == NULL)
    throw Arcane::Exception::exception();
  UInt16 type = oldChild->nodeType();
  if (type == LIBXML2_Node::ATTRIBUTE_NODE || type == LIBXML2_Node::DOCUMENT_TYPE_NODE ||
      type == LIBXML2_Node::NOTATION_NODE || type == LIBXML2_Node::ENTITY_NODE)
    throw Arcane::Exception::exception();
  return removeChildPrivate(oldChild);
}

LIBXML2_Node* LIBXML2_Node::
removeChildPrivate(LIBXML2_Node* oldChild)
{
  std::list<LIBXML2_Node*>::iterator posit = std::find(mNodeList.begin(), mNodeList.end(), oldChild);
  if (posit == mNodeList.end())
    throw Arcane::Exception::exception();
  mNodeList.erase(posit);
  oldChild->mParent = NULL;
  UInt32 i, rc = oldChild->_libxml2_refcount;
  for (i = 0; i < rc; i++)
    release_ref();
  if (mDocumentIsAncestor)
    oldChild->updateDocumentAncestorStatus(false);
  oldChild->add_ref();
  return oldChild;
}

LIBXML2_Node* LIBXML2_Node::
appendChild(LIBXML2_Node* inewChild)
{
  return insertBefore(inewChild, NULL);
}

bool LIBXML2_Node::
hasChildNodes()
{
  return !mNodeList.empty();
}

LIBXML2_Node* LIBXML2_Node::
cloneNode(bool deep)
{
  return cloneNodePrivate(mDocument, deep);
}

LIBXML2_Node* LIBXML2_Node::
cloneNodePrivate(LIBXML2_Document* aDoc, bool deep)
{
  LIBXML2_Node* c = shallowCloneNode(aDoc);
  if (!deep) {
    c->add_ref();
    return c;
  }
  // Clone all children...
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++) {
    // See if its an attribute(in which case we already cloned it).
    if ((*i)->nodeType() == LIBXML2_Node::ATTRIBUTE_NODE)
      continue;
    LIBXML2_Node* n = (*i)->cloneNodePrivate(c->mDocument, true);
    c->insertBeforePrivate(n, NULL)->release_ref();
  }
  c->add_ref();
  return c;
}

void LIBXML2_Node::
normalize()
{
  // Normalize the children...
  std::list<LIBXML2_Node*>::iterator i;
  for (i = mNodeList.begin(); i != mNodeList.end(); i++)
    (*i)->normalize();

  // Now scan through our nodes and look for adjacent text nodes to fold into
  // single nodes, or delete...
  LIBXML2_TextBase* lastText = NULL;
  for (i = mNodeList.begin(); i != mNodeList.end(); i++) {
    LIBXML2_TextBase* tb = dynamic_cast<LIBXML2_TextBase*>(*i);
    if (tb == NULL) {
      lastText = NULL;
      continue;
    }
    if (tb->mNodeValue == String()) {
      removeChild(tb)->release_ref();
      continue;
    }
    if (lastText != NULL) {
      removeChild(tb)->release_ref();
      lastText->mNodeValue = lastText->mNodeValue + tb->mNodeValue;
      continue;
    }
    lastText = tb;
  }
}

bool LIBXML2_Node::
isSupported(const String& feature, const String& version)
{
  if ((feature == "xml" && (version == "1.0")) || (version == "2.0") || (version == "3.0"))
    return true;
  if ((feature == "Core" && (version == "1.0")) || (version == "2.0") || (version == "3.0"))
    return true;
  if (feature == "Trav" && (version == "2.0"))
    return true;
  if (feature == "Range" && (version == "2.0"))
    return true;
  if (feature == "LS" && (version == "2.0"))
    return true;
  if (feature == "XPath" && (version == "3.0"))
    return true;
  return false;
}

String LIBXML2_Node::
namespaceURI()
{
  return mNamespaceURI;
}

String LIBXML2_Node::
prefix()
{
  const XMLCh* cNodename = reinterpret_cast<const XMLCh*>(mNodeName.utf16().begin());
  const XMLCh* cpos = wcschr(cNodename, U(':'));
  if (cpos == NULL)
    return String();
  cpos++;
  return mNodeName.substring(0, wcslen(cNodename) - wcslen(cpos));
}

void LIBXML2_Node::
prefix(const String& attr)
{
  if (mNamespaceURI == String())
    throw Arcane::Exception::exception();
  if (attr == "xml" && mNamespaceURI != "http://www.w3.org/XML/1998/namespace")
    throw Arcane::Exception::exception();
  if (mLocalName == "xmlns" && attr != String())
    throw Arcane::Exception::exception();
  if (attr == "xmlns" && mNamespaceURI != "http://www.w3.org/2000/xmlns/")
    throw Arcane::Exception::exception();
  mNodeName = attr;
  mNodeName = mNodeName + ":";
  mNodeName = mNodeName + mLocalName;
}

String LIBXML2_Node::
localName()
{
  return mLocalName;
}

void LIBXML2_Node::
recursivelyChangeDocument(LIBXML2_Document* aNewDocument)
{
  ARCANE_ASSERT(!mDocumentIsAncestor, ("The document is the ancestor !"));
  if (mDocument == aNewDocument)
    return;
  if (mDocument != NULL)
    mDocument->release_ref();
  mDocument = aNewDocument;
  if (mDocument != NULL)
    mDocument->add_ref();
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++)
    (*i)->recursivelyChangeDocument(mDocument);
}

LIBXML2_Element* LIBXML2_Node::
searchForElementById(const String& elementId)
{
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++) {
    LIBXML2_Element* e = (*i)->searchForElementById(elementId);
    if (e != NULL)
      return e;
  }
  return NULL;
}

#ifdef DEBUG_NODELEAK
void LIBXML2_Node::
find_leaked()
{
  UInt32 sum = 0;
  for (std::list<LIBXML2_Node*>::const_iterator i(mNodeList.begin()); i != mNodeList.end(); i++) {
    sum += (*i)->_libxml2_refcount;
    (*i)->find_leaked();
  }
  ARCANE_ASSERT((_libxml2_refcount != sum), ("Warning: object leaked ");
}
#endif // DEBUG_NODELEAK

LIBXML2_Node* LIBXML2_NodeList::
  item(UInt32 index)
{
  UInt32 saveIndex = index;
  if (mParent == NULL)
    return NULL;
  for (std::list<LIBXML2_Node*>::iterator i = mParent->mNodeList.begin(); i != mParent->mNodeList.end();
       i++) {
    // See if it is a type we ignore...
    UInt16 type = (*i)->nodeType();
    if (type == LIBXML2_Node::ATTRIBUTE_NODE || type == LIBXML2_Node::DOCUMENT_TYPE_NODE)
      continue;
    if (index == 0) {
      m_hintIterator = i;
      m_hintIndex = saveIndex;
      (*i)->add_ref();
      return (*i);
    }
    index--;
  }
  return NULL;
}

UInt32 LIBXML2_NodeList::
 length()
{
  UInt32 length = 0;
  for (std::list<LIBXML2_Node*>::iterator i = mParent->mNodeList.begin(); i != mParent->mNodeList.end();
       i++) {
    UInt16 type = (*i)->nodeType();
    if (type == LIBXML2_Node::ATTRIBUTE_NODE || type == LIBXML2_Node::DOCUMENT_TYPE_NODE)
      continue;
    length++;
  }
  return length;
}

LIBXML2_Node* LIBXML2_NodeListDFSSearch::
 item(UInt32 index)
{
  if (mParent == NULL)
    return NULL;
  // We have a list of iterators...
  std::list<std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>> iteratorStack;
  iteratorStack.push_front(std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>(
          mParent->mNodeList.begin(), mParent->mNodeList.end()));
  while (!iteratorStack.empty()) {
    std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>& itp =
            iteratorStack.front();
    if (itp.first == itp.second) {
      iteratorStack.pop_front();
      continue;
    }
    std::list<LIBXML2_Node*>::iterator p = itp.first;
    itp.first++;
    // This is a pre-order traversal, so consider the element first...
    if ((*p)->nodeType() == LIBXML2_Node::ELEMENT_NODE) {
      bool hit = true;
      switch (mFilterType) {
      case LEVEL_1_NAME_FILTER:
        if ((*p)->mNodeName != mNameFilter && (*p)->mNodeName != "*")
          hit = false;
        break;
      case LEVEL_2_NAME_FILTER:
        if ((*p)->mLocalName != mNameFilter && (mNameFilter != "*"))
          hit = false;
        if ((*p)->mNamespaceURI != mNamespaceFilter && (mNamespaceFilter != "*"))
          hit = false;
        break;
      }
      if (hit) {
        if (index == 0) {
          (*p)->add_ref();
          return *p;
        }
        index--;
      }
    }
    // Next, we need to recurse...
    iteratorStack.push_front(
            std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>(
                    (*p)->mNodeList.begin(), (*p)->mNodeList.end()));
  }
  return NULL;
}

UInt32 LIBXML2_NodeListDFSSearch::
 length()
{
  if (mParent == NULL)
    return 0;
  UInt32 length = 0;
  // We have a list of iterators...
  std::list<std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>> iteratorStack;
  iteratorStack.push_front(std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>(
          mParent->mNodeList.begin(), mParent->mNodeList.end()));
  while (!iteratorStack.empty()) {
    std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>& itp =
            iteratorStack.front();
    if (itp.first == itp.second) {
      iteratorStack.pop_front();
      continue;
    }
    // This is a pre-order traversal, so consider the element first...
    if ((*(itp.first))->nodeType() == LIBXML2_Node::ELEMENT_NODE) {
      bool hit = true;
      switch (mFilterType) {
      case LEVEL_1_NAME_FILTER:
        if ((*(itp.first))->mNodeName != mNameFilter && (*(itp.first))->mNodeName != "*")
          hit = false;
        break;
      case LEVEL_2_NAME_FILTER:
        if ((*(itp.first))->mLocalName != mNameFilter && (mNameFilter != "*"))
          hit = false;
        if ((*(itp.first))->mNamespaceURI != mNamespaceFilter && (mNamespaceFilter != "*"))
          hit = false;
        break;
      }
      if (hit) {
        length++;
      }
    }
    // Next, we need to recurse...
    LIBXML2_Node* n = *(itp.first);
    itp.first++;
    iteratorStack.push_front(
            std::pair<std::list<LIBXML2_Node*>::iterator, std::list<LIBXML2_Node*>::iterator>(
                    n->mNodeList.begin(), n->mNodeList.end()));
  }
  return length;
}

LIBXML2_NamedNodeMap::
 LIBXML2_NamedNodeMap(LIBXML2_Element* aElement)
: mElement(aElement)
, m_hintIndex(0)
{
  mElement->add_ref();
}

LIBXML2_NamedNodeMap::
 ~LIBXML2_NamedNodeMap()
{
  mElement->release_ref();
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
 getNamedItem(const String& name)
{
  std::map<LIBXML2_Element::LocalName, LIBXML2_Attr*>::iterator i =
          mElement->attributeMap.find(LIBXML2_Element::LocalName(name));
  if (i == mElement->attributeMap.end())
    return NULL;
  (*i).second->add_ref();
  return (*i).second;
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
 setNamedItem(LIBXML2_Node* arg)
{
  LIBXML2_Attr* attr = dynamic_cast<LIBXML2_Attr*>(arg);
  return mElement->setAttributeNode(attr);
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
 removeNamedItem(const String& name)
{
  std::map<LIBXML2_Element::LocalName, LIBXML2_Attr*>::iterator i =
          mElement->attributeMap.find(LIBXML2_Element::LocalName(name));
  if (i == mElement->attributeMap.end())
    throw Arcane::Exception::exception();
  // Remove the child(which sorts out the refcounting)...
  LIBXML2_Attr* at = (*i).second;
  mElement->removeChildPrivate(at)->release_ref();
  LIBXML2_Element::LocalName ln((*i).first);
  mElement->attributeMap.erase(i);
  std::map<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>::iterator j =
          mElement->attributeMapNS.find(LIBXML2_Element::QualifiedName(at->mNamespaceURI, at->mLocalName));
  LIBXML2_Element::QualifiedName qn((*j).first);
  mElement->attributeMapNS.erase(j);
  return at;
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
 item(UInt32 index)
{
  UInt32 saveIndex = index;
  for (std::map<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>::iterator i = mElement->attributeMapNS.begin();
       i != mElement->attributeMapNS.end(); i++) {
    if (index == 0) {
      m_hintIterator = i;
      m_hintIndex = saveIndex;

      (*i).second->add_ref();
      return (*i).second;
    }
    index--;
  }
  return NULL;
}

UInt32 LIBXML2_NamedNodeMap::
length()
{
  return mElement->attributeMap.size();
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
getNamedItemNS(const String& namespaceURI, const String& localName)
{
  std::map<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>::iterator i =
          mElement->attributeMapNS.find(LIBXML2_Element::QualifiedName(namespaceURI, localName));
  if (i == mElement->attributeMapNS.end())
    return NULL;
  (*i).second->add_ref();
  return (*i).second;
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
setNamedItemNS(LIBXML2_Node* arg)
{
  LIBXML2_Attr* attr = dynamic_cast<LIBXML2_Attr*>(arg);
  return mElement->setAttributeNodeNS(attr);
}

LIBXML2_Node* LIBXML2_NamedNodeMap::
removeNamedItemNS(const String& namespaceURI, const String& localName)
{
  std::map<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>::iterator i =
          mElement->attributeMapNS.find(LIBXML2_Element::QualifiedName(namespaceURI, localName));
  if (i == mElement->attributeMapNS.end())
    throw Arcane::Exception::exception();
  // Remove the child(which sorts out the refcounting)...
  LIBXML2_Attr* at = (*i).second;
  mElement->removeChildPrivate(at)->release_ref();
  std::map<LIBXML2_Element::LocalName, LIBXML2_Attr*>::iterator j =
          mElement->attributeMap.find(LIBXML2_Element::LocalName(at->mNodeName));
  mElement->attributeMapNS.erase(i);
  mElement->attributeMap.erase(j);
  return at;
}

LIBXML2_NamedNodeMapDT::
LIBXML2_NamedNodeMapDT(LIBXML2_DocumentType* aDocType, UInt16 aType)
: mDocType(aDocType)
, mType(aType)
{
  mDocType->add_ref();
}

LIBXML2_NamedNodeMapDT::
~LIBXML2_NamedNodeMapDT()
{
  mDocType->release_ref();
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
getNamedItem(const String& name)
{
  for (std::list<LIBXML2_Node*>::iterator i = mDocType->mNodeList.begin(); i != mDocType->mNodeList.end();
       i++) {
    if ((*i)->nodeType() != mType)
      continue;
    if ((*i)->mNodeName != name)
      continue;
    (*i)->add_ref();
    return (*i);
  }
  return NULL;
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
setNamedItem(LIBXML2_Node* arg)
{
  ARCANE_UNUSED(arg);
  throw NotImplementedException(A_FUNCINFO);
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
removeNamedItem(const String& name)
{
  ARCANE_UNUSED(name);
  throw NotImplementedException(A_FUNCINFO);
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
item(UInt32 index)
{
  for (std::list<LIBXML2_Node*>::iterator i = mDocType->mNodeList.begin(); i != mDocType->mNodeList.end();
       i++) {
    if ((*i)->nodeType() != mType)
      continue;
    if (index != 0) {
      index--;
      continue;
    }
    (*i)->add_ref();
    return *i;
  }
  return NULL;
}

UInt32 LIBXML2_NamedNodeMapDT::
length()
{
  UInt32 l = 0;
  for (std::list<LIBXML2_Node*>::iterator i = mDocType->mNodeList.begin(); i != mDocType->mNodeList.end();
       i++) {
    if ((*i)->nodeType() != mType)
      continue;
    l++;
  }
  return l;
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
getNamedItemNS(const String& namespaceURI, const String& localName)
{
  for (std::list<LIBXML2_Node*>::iterator i = mDocType->mNodeList.begin(); i != mDocType->mNodeList.end();
       i++) {
    if ((*i)->nodeType() != mType)
      continue;
    if ((*i)->mNamespaceURI != namespaceURI)
      continue;
    if ((*i)->mLocalName != localName)
      continue;
    (*i)->add_ref();
    return (*i);
  }
  return NULL;
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
setNamedItemNS(LIBXML2_Node* arg)
{
  ARCANE_UNUSED(arg);
  throw NotImplementedException(A_FUNCINFO);
}

LIBXML2_Node* LIBXML2_NamedNodeMapDT::
removeNamedItemNS(const String& namespaceURI, const String& localName)
{
  ARCANE_UNUSED(namespaceURI);
  ARCANE_UNUSED(localName);
  throw NotImplementedException(A_FUNCINFO);
}

String LIBXML2_CharacterData::
Data()
{
  return mNodeValue;
}

void LIBXML2_CharacterData::
Data(const String& attr)
{
  mNodeValue = attr;
}

void LIBXML2_CharacterData::
nodeValue(const String& attr)
{
  mNodeValue = attr;
}

UInt32 LIBXML2_CharacterData::
length()
{
  return mNodeValue.len();
}

String LIBXML2_CharacterData::
substringdata(UInt32 offset, UInt32 count)
{
  try {
    return mNodeValue.substring(offset, count);
  }
  catch (const Arcane::dom::DOMException& ex) {
    cerr << "** EXCEPTION " << ex.code << '\n';
    throw ex;
  }
}

void LIBXML2_CharacterData::
appenddata(const String& arg)
{
  mNodeValue = mNodeValue + arg;
}

void LIBXML2_CharacterData::
insertdata(UInt32 offset, const String& arg)
{
  try {
    String t = mNodeValue.substring(offset);
    mNodeValue = mNodeValue.substring(0, offset);
    mNodeValue = mNodeValue + arg;
    mNodeValue = mNodeValue + t;
  }
  catch (const Arcane::dom::DOMException& ex) {
    cerr << "** EXCEPTION " << ex.code << '\n';
    throw ex;
  }
}

void LIBXML2_CharacterData::
deletedata(UInt32 offset, UInt32 count)
{
  try {
    String t = mNodeValue.substring(offset + count);
    mNodeValue = mNodeValue.substring(0, offset);
    mNodeValue = mNodeValue + t;
  }
  catch (const Arcane::dom::DOMException& ex) {
    cerr << "** EXCEPTION " << ex.code << '\n';
    throw ex;
  }
}

void LIBXML2_CharacterData::
replacedata(UInt32 offset, UInt32 count, const String& arg)
{
  try {
    String t = mNodeValue.substring(offset + count);
    mNodeValue = mNodeValue.substring(0, offset);
    mNodeValue = mNodeValue + arg;
    mNodeValue = mNodeValue + t;
  }
  catch (const Arcane::dom::DOMException& ex) {
    cerr << "** EXCEPTION " << ex.code << '\n';
    throw ex;
  }
}

LIBXML2_Node* LIBXML2_Attr::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_Attr* attr = new LIBXML2_Attr(aDoc);
  attr->mLocalName = mLocalName;
  attr->mNamespaceURI = mNamespaceURI;
  attr->mNodeName = mNodeName;
  attr->mNodeValue = mNodeValue;
  attr->mSpecified = mSpecified;
  return attr;
}

String LIBXML2_Attr::
name()
{
  return mNodeName;
}

bool LIBXML2_Attr::
specified()
{
  return mSpecified;
}

String LIBXML2_Attr::
value()
{
  return mNodeValue;
}

void LIBXML2_Attr::
value(const String& attr)
{
  mNodeValue = attr;
  mSpecified = true;
}

LIBXML2_Element* LIBXML2_Attr::
ownerElement()
{
  LIBXML2_Element* el = dynamic_cast<LIBXML2_Element*>(mParent);
  if (el != NULL)
    el->add_ref();
  return el;
}

LIBXML2_Element::
~LIBXML2_Element() {}

LIBXML2_Node* LIBXML2_Element::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_Element* el = LIBXML2_NewElement(aDoc, mNamespaceURI, mLocalName);
  el->mLocalName = mLocalName;
  el->mNamespaceURI = mNamespaceURI;
  el->mNodeName = mNodeName;
  el->mNodeValue = mNodeValue;
  // Attributes get cloned too...
  for (std::map<QualifiedName, LIBXML2_Attr*>::iterator i = attributeMapNS.begin(); i != attributeMapNS.end();
       i++) {
    LIBXML2_Attr* at = dynamic_cast<LIBXML2_Attr*>((*i).second->shallowCloneNode(aDoc));
    LIBXML2_Attr* atTmp = el->setAttributeNodeNS(at);
    ARCANE_ASSERT(atTmp == NULL,
                  ("InternalError in LIBXML2_Element::shallowCloneNode(). The attribute is Null !"));
  }
  return el;
}

LIBXML2_NamedNodeMap* LIBXML2_Element::
attributes()
{
  return new LIBXML2_NamedNodeMap(this);
}

bool LIBXML2_Element::
hasAttributes()
{
  return !attributeMap.empty();
}

String LIBXML2_Element::
tagName()
{
  return mNodeName;
}

String LIBXML2_Element::
getAttribute(const String& name)
{
  std::map<LocalName, LIBXML2_Attr*>::iterator i = attributeMap.find(LocalName(name));
  if (i == attributeMap.end())
    return String();
  return (*i).second->value();
}

void LIBXML2_Element::
setAttribute(const String& name, const String& value)
{
  std::map<LocalName, LIBXML2_Attr*>::iterator i = attributeMap.find(LocalName(name));
  if (i == attributeMap.end()) {
    LIBXML2_Attr* a = new LIBXML2_Attr(mDocument);
    if (a == NULL)
      throw Arcane::Exception::exception();
    a->mNodeValue = value;
    a->mNodeName = name;
    insertBeforePrivate(a, NULL)->release_ref();
    attributeMapNS.insert(std::pair<QualifiedName, LIBXML2_Attr*>(QualifiedName(String(), name), a));
    attributeMap.insert(std::pair<LocalName, LIBXML2_Attr*>(LocalName(name), a));
    return;
  }
  String oldvalue = (*i).second->mNodeValue;
  (*i).second->mNodeValue = value;
}

void LIBXML2_Element::
removeAttribute(const String& name)
{
  std::map<LocalName, LIBXML2_Attr*>::iterator i = attributeMap.find(LocalName(name));
  if (i == attributeMap.end()) {
    // DOM doesn't say its an error to remove a non-existant attribute.
    return;
  }
  // Remove the child(which sorts out the refcounting)...
  LIBXML2_Attr* at = (*i).second;
  removeChildPrivate(at)->release_ref();
  attributeMap.erase(i);
  std::map<QualifiedName, LIBXML2_Attr*>::iterator j;
  if (at->mLocalName != String())
    j = attributeMapNS.find(QualifiedName(at->mNamespaceURI, at->mLocalName));
  else
    j = attributeMapNS.find(QualifiedName(at->mNamespaceURI, at->mNodeName));
  attributeMapNS.erase(j);
}

LIBXML2_Attr* LIBXML2_Element::
getAttributeNode(const String& name)
{
  std::map<LocalName, LIBXML2_Attr*>::iterator i = attributeMap.find(LocalName(name));
  if (i == attributeMap.end())
    return NULL;
  (*i).second->add_ref();
  return (*i).second;
}

LIBXML2_Attr* LIBXML2_Element::
setAttributeNode(LIBXML2_Attr* newAttr)
{
  if (newAttr == NULL)
    throw Arcane::Exception::exception();
  String name = newAttr->name();
  std::map<LocalName, LIBXML2_Attr*>::iterator i = attributeMap.find(LocalName(name));
  std::map<QualifiedName, LIBXML2_Attr*>::iterator j = attributeMapNS.find(QualifiedName(String(), name));
  if (i == attributeMap.end()) {
    insertBeforePrivate(newAttr, NULL)->release_ref();
    attributeMap.insert(std::pair<LocalName, LIBXML2_Attr*>(name, newAttr));
    attributeMapNS.insert(std::pair<QualifiedName, LIBXML2_Attr*>(QualifiedName(String(), name), newAttr));
    return NULL;
  }
  LIBXML2_Attr* at = (*i).second;
  removeChildPrivate(at)->release_ref();
  attributeMap.erase(i);
  if (j != attributeMapNS.end()) {
    attributeMapNS.erase(j);
  }
  insertBeforePrivate(newAttr, NULL)->release_ref();
  attributeMap.insert(std::pair<LocalName, LIBXML2_Attr*>(LocalName(name), newAttr));
  attributeMapNS.insert(std::pair<QualifiedName, LIBXML2_Attr*>(QualifiedName(String(), name), newAttr));
  return at;
}

LIBXML2_Attr* LIBXML2_Element::
removeAttributeNode(LIBXML2_Attr* oldAttr)
{
  if (oldAttr == NULL)
    throw Arcane::Exception::exception();
  String name = oldAttr->name();
  String lname = oldAttr->localName();
  String nsuri = oldAttr->namespaceURI();
  std::map<LocalName, LIBXML2_Attr*>::iterator i = attributeMap.find(LocalName(name));
  if (i == attributeMap.end()) {
    throw Arcane::Exception::exception();
  }
  LIBXML2_Attr* at = (*i).second;
  LocalName ln((*i).first);
  removeChildPrivate(at)->release_ref();
  attributeMap.erase(i);

  std::map<QualifiedName, LIBXML2_Attr*>::iterator j = attributeMapNS.find(QualifiedName(nsuri, lname));
  QualifiedName qn((*j).first);
  attributeMapNS.erase(j);
  return at;
}

LIBXML2_NodeList* LIBXML2_Element::
getElementsByTagName(const String& name)
{
  return new LIBXML2_NodeListDFSSearch(this, name);
}

String LIBXML2_Element::
getAttributeNS(const String& namespaceURI, const String& localName)
{
  std::map<QualifiedName, LIBXML2_Attr*>::iterator i =
          attributeMapNS.find(QualifiedName(namespaceURI, localName));
  if (i == attributeMapNS.end())
    return String();
  return (*i).second->value();
}

void LIBXML2_Element::
setAttributeNS(const String& namespaceURI, const String& qualifiedName, const String& value)
{
  const XMLCh* cqname = reinterpret_cast<const XMLCh*>(qualifiedName.utf16().begin());
  const XMLCh* cpos = wcschr(cqname, L':');
  if (cpos == NULL)
    cpos = cqname;
  else
    cpos++;
  ByteUniqueArray utf8_array(cpos[0]);
  String localName = String::fromUtf8(utf8_array);
  std::map<QualifiedName, LIBXML2_Attr*>::iterator i =
          attributeMapNS.find(QualifiedName(namespaceURI, localName));
  if (i == attributeMapNS.end()) {
    LIBXML2_Attr* a = new LIBXML2_Attr(mDocument);
    if (a == NULL)
      throw Arcane::Exception::exception();
    a->value(value);
    a->mNodeName = qualifiedName;
    a->mLocalName = localName;
    a->mNamespaceURI = namespaceURI;
    insertBeforePrivate(a, NULL)->release_ref();
    attributeMapNS.insert(std::pair<QualifiedName, LIBXML2_Attr*>(QualifiedName(namespaceURI, localName), a));
    std::map<LocalName, LIBXML2_Attr*>::iterator j = attributeMap.find(LocalName(qualifiedName));
    if (j != attributeMap.end()) {
      attributeMap.erase(j);
    }
    attributeMap.insert(std::pair<LocalName, LIBXML2_Attr*>(LocalName(qualifiedName), a));
    return;
  }
  String oldValue = (*i).second->mNodeValue;
  (*i).second->mNodeValue = value;
}

void LIBXML2_Element::
removeAttributeNS(const String& namespaceURI, const String& localName)
{
  std::map<QualifiedName, LIBXML2_Attr*>::iterator i =
          attributeMapNS.find(QualifiedName(namespaceURI, localName));
  if (i == attributeMapNS.end()) {
    // DOM doesn't say its an error to remove a non-existant attribute.
    return;
  }
  // Remove the child(which sorts out the refcounting)...
  LIBXML2_Attr* at = (*i).second;
  removeChildPrivate(at)->release_ref();
  QualifiedName qn((*i).first);
  std::map<LocalName, LIBXML2_Attr*>::iterator j = attributeMap.find(LocalName((*i).second->mNodeName));
  attributeMapNS.erase(i);
  LocalName ln((*j).first);
  attributeMap.erase(j);
}

LIBXML2_Attr* LIBXML2_Element::
getAttributeNodeNS(const String& namespaceURI, const String& localName)
{
  std::map<QualifiedName, LIBXML2_Attr*>::iterator i =
          attributeMapNS.find(QualifiedName(namespaceURI, localName));
  if (i == attributeMapNS.end())
    return NULL;
  (*i).second->add_ref();
  return (*i).second;
}

LIBXML2_Attr* LIBXML2_Element::
setAttributeNodeNS(LIBXML2_Attr* newAttr)
{
  if (newAttr == NULL)
    throw Arcane::Exception::exception();
  if (newAttr->mLocalName == String())
    newAttr->mLocalName = newAttr->mNodeName;
  std::pair<String, String> p(newAttr->mNamespaceURI, newAttr->mLocalName);
  std::map<QualifiedName, LIBXML2_Attr*>::iterator i =
          attributeMapNS.find(QualifiedName(newAttr->mNamespaceURI, newAttr->mLocalName));
  if (i == attributeMapNS.end()) {
    insertBeforePrivate(newAttr, NULL)->release_ref();
    attributeMapNS.insert(std::pair<QualifiedName, LIBXML2_Attr*>(
            QualifiedName(newAttr->mNamespaceURI, newAttr->mLocalName), newAttr));
    attributeMap.insert(std::pair<LocalName, LIBXML2_Attr*>(LocalName(newAttr->mNodeName), newAttr));
    return NULL;
  }
  LIBXML2_Attr* at = (*i).second;
  QualifiedName qn((*i).first);
  removeChildPrivate(at)->release_ref();
  attributeMapNS.erase(i);
  std::map<LocalName, LIBXML2_Attr*>::iterator j = attributeMap.find(LocalName(at->mNodeName));
  LocalName ln((*j).first);
  attributeMap.erase(j);
  insertBeforePrivate(newAttr, NULL)->release_ref();
  attributeMapNS.insert(std::pair<QualifiedName, LIBXML2_Attr*>(
          QualifiedName(newAttr->mNamespaceURI, newAttr->mLocalName), newAttr));
  attributeMap.insert(std::pair<LocalName, LIBXML2_Attr*>(LocalName(at->mNodeName), newAttr));
  return at;
}

LIBXML2_NodeList* LIBXML2_Element::
getElementsByTagNameNS(const String& namespaceURI, const String& localName)
{
  return new LIBXML2_NodeListDFSSearch(this, namespaceURI, localName);
}

bool LIBXML2_Element::
hasAttribute(const String& name)
{
  return attributeMap.find(LocalName(name)) != attributeMap.end();
}

bool LIBXML2_Element::
hasAttributeNS(const String& namespaceURI, const String& localName)
{
  return attributeMapNS.find(QualifiedName(namespaceURI, localName)) != attributeMapNS.end();
}

LIBXML2_Element* LIBXML2_Element::
searchForElementById(const String& elementId)
{
  // XXX DOM says: 'Attributes with the name "ID" are not of type ID unless so
  //     defined.', but we don't deal with DTDs, so we deviate from the DOM and
  //     just assume that "id" in namespace "" is of type ID.
  String ourId = getAttribute("id");
  if (ourId == String()) {
    String tmp = getAttribute("xml:id");
    if (tmp != String())
      ourId = tmp;
  }
  if (ourId == elementId) {
    add_ref();
    return this;
  }
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++) {
    LIBXML2_Element* e = (*i)->searchForElementById(elementId);
    if (e != NULL)
      return e;
  }
  return NULL;
}

LIBXML2_Text* LIBXML2_TextBase::
splitText(UInt32 offset)
{
  if (mParent == NULL)
    throw Arcane::Exception::exception();
  LIBXML2_TextBase* tb = static_cast<LIBXML2_TextBase*>(shallowCloneNode(mDocument));
  tb->mNodeValue = mNodeValue.substring(offset);
  mNodeValue = mNodeValue.substring(0, offset);
  LIBXML2_Node* n2 = nextSibling();
  mParent->insertBefore(tb, n2)->release_ref();
  tb->add_ref();
  return reinterpret_cast<LIBXML2_Text*>(tb);
}

LIBXML2_Text* LIBXML2_CDATASection::
splitText(UInt32 offset)
{
  if (mParent == NULL)
    throw Arcane::Exception::exception();
  LIBXML2_TextBase* tb = static_cast<LIBXML2_TextBase*>(shallowCloneNode(mDocument));
  tb->mNodeValue = mNodeValue.substring(offset);
  mNodeValue = mNodeValue.substring(0, offset);
  LIBXML2_Node* n2 = nextSibling();
  mParent->insertBefore(tb, n2)->release_ref();
  tb->add_ref();
  return reinterpret_cast<LIBXML2_Text*>(tb);
}

LIBXML2_Node* LIBXML2_Text::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_Text* txt = new LIBXML2_Text(aDoc);
  if (txt == NULL)
    throw Arcane::Exception::exception();
  txt->mLocalName = mLocalName;
  txt->mNamespaceURI = mNamespaceURI;
  txt->mNodeName = mNodeName;
  txt->mNodeValue = mNodeValue;
  return txt;
}

LIBXML2_Node* LIBXML2_Comment::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_Comment* com = new LIBXML2_Comment(aDoc);
  if (com == NULL)
    throw Arcane::Exception::exception();
  com->mLocalName = mLocalName;
  com->mNamespaceURI = mNamespaceURI;
  com->mNodeName = mNodeName;
  com->mNodeValue = mNodeValue;
  return com;
}

LIBXML2_Node* LIBXML2_CDATASection::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_CDATASection* cds = new LIBXML2_CDATASection(aDoc);
  if (cds == NULL)
    throw Arcane::Exception::exception();
  cds->mLocalName = mLocalName;
  cds->mNamespaceURI = mNamespaceURI;
  cds->mNodeName = mNodeName;
  cds->mNodeValue = mNodeValue;
  return cds;
}

LIBXML2_Node* LIBXML2_DocumentType::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_DocumentType* doctype = new LIBXML2_DocumentType(aDoc, mNodeName, mPublicId, mSystemId);
  if (doctype == NULL)
    throw Arcane::Exception::exception();
  doctype->mLocalName = mLocalName;
  doctype->mNamespaceURI = mNamespaceURI;
  doctype->mNodeValue = mNodeValue;
  return doctype;
}

String LIBXML2_DocumentType::
name()
{
  return mNodeName;
}

LIBXML2_NamedNodeMap* LIBXML2_DocumentType::
entities()
{
  return new LIBXML2_NamedNodeMapDT(this, LIBXML2_Node::ENTITY_NODE);
}

LIBXML2_NamedNodeMap* LIBXML2_DocumentType::
notations()
{
  return new LIBXML2_NamedNodeMapDT(this, LIBXML2_Node::NOTATION_NODE);
}

String LIBXML2_DocumentType::
publicId()
{
  return mPublicId;
}

String LIBXML2_DocumentType::
systemId()
{
  return mSystemId;
}

String LIBXML2_DocumentType::
internalSubset()
{
  // The DOM basically leaves this up to the API, and since we don't store this
  // information as it is irrelevant to appli, lets just skip it...
  return String();
}

LIBXML2_Node* LIBXML2_Notation::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_Notation* nota = new LIBXML2_Notation(aDoc, mPublicId, mSystemId);
  if (nota == NULL)
    throw Arcane::Exception::exception();
  nota->mLocalName = mLocalName;
  nota->mNamespaceURI = mNamespaceURI;
  nota->mNodeName = mNodeName;
  nota->mNodeValue = mNodeValue;
  return nota;
}

String LIBXML2_Notation::
publicId()
{
  return mPublicId;
}

String LIBXML2_Notation::
systemId()
{
  return mSystemId;
}

LIBXML2_Node*
LIBXML2_Entity::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_Entity* ent = new LIBXML2_Entity(aDoc, mPublicId, mSystemId, mNotationName);
  if (ent == NULL)
    throw Arcane::Exception::exception();
  ent->mLocalName = mLocalName;
  ent->mNamespaceURI = mNamespaceURI;
  ent->mNodeName = mNodeName;
  ent->mNodeValue = mNodeValue;
  return ent;
}

String LIBXML2_Entity::
publicId()
{
  return mPublicId;
}

String LIBXML2_Entity::
systemId()
{
  return mSystemId;
}

String LIBXML2_Entity::
notationName()
{
  return mNotationName;
}

LIBXML2_Node* LIBXML2_EntityReference::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_EntityReference* er = new LIBXML2_EntityReference(aDoc);
  if (er == NULL)
    throw Arcane::Exception::exception();
  er->mLocalName = mLocalName;
  er->mNamespaceURI = mNamespaceURI;
  er->mNodeName = mNodeName;
  er->mNodeValue = mNodeValue;
  return er;
}

LIBXML2_Node* LIBXML2_ProcessingInstruction::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_ProcessingInstruction* pi = new LIBXML2_ProcessingInstruction(aDoc, mNodeName, mNodeValue);
  if (pi == NULL)
    throw Arcane::Exception::exception();
  pi->mLocalName = mLocalName;
  pi->mNamespaceURI = mNamespaceURI;
  return pi;
}

String LIBXML2_ProcessingInstruction::
target()
{
  return mNodeName;
}

String LIBXML2_ProcessingInstruction::
Data()
{
  return mNodeValue;
}

void LIBXML2_ProcessingInstruction::
Data(const String& attr)
{
  mNodeValue = attr;
}

LIBXML2_Node* LIBXML2_DocumentFragment::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  LIBXML2_DocumentFragment* docfrag = new LIBXML2_DocumentFragment(aDoc);
  if (docfrag == NULL)
    throw Arcane::Exception::exception();
  docfrag->mLocalName = mLocalName;
  docfrag->mNamespaceURI = mNamespaceURI;
  docfrag->mNodeName = mNodeName;
  docfrag->mNodeValue = mNodeValue;
  return docfrag;
}

LIBXML2_Document::
LIBXML2_Document(const String& namespaceURI, const String& qualifiedName,
                 LIBXML2_DocumentType* doctype)
: LIBXML2_Node(this)
{
  const XMLCh* cqname = reinterpret_cast<const XMLCh*>(qualifiedName.utf16().begin());
  const XMLCh* cpos = wcschr(cqname, L':');
  if (cpos == NULL)
    cpos = cqname;
  else
    cpos++;
  ByteUniqueArray utf8_array(cpos[0]);
  String localName = String::fromUtf8(utf8_array);

  // We are our own document ancestor, so fix the refcounts...
  mDocumentIsAncestor = true;
  mDocument->release_ref();
  if (doctype && doctype->mDocument != NULL)
    throw Arcane::Exception::exception();
  if (doctype) {
    doctype->mDocument = this;
    doctype->mDocumentIsAncestor = false;
    doctype->mDocument->add_ref();
  }
  if (doctype != NULL)
    insertBeforePrivate(doctype, NULL)->release_ref();
  LIBXML2_Element* docel = LIBXML2_NewElement(this, namespaceURI, localName);
  if (docel == NULL)
    throw Arcane::Exception::exception();
  docel->mNamespaceURI = namespaceURI;
  docel->mNodeName = qualifiedName;
  docel->mLocalName = localName;
  appendChild(docel)->release_ref();
}

LIBXML2_Text* LIBXML2_Document::
doctype()
{
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++)
    if ((*i)->nodeType() == LIBXML2_Node::DOCUMENT_TYPE_NODE) {
      LIBXML2_DocumentType* dt = static_cast<LIBXML2_DocumentType*>(*i);
      dt->add_ref();
      return dynamic_cast<LIBXML2_Text*>(dt);
    }
  return NULL;
}

LIBXML2_DOMImplementation* LIBXML2_Document::
implementation()
{
  LIBXML2_DOMImplementation::sDOMImplementation->add_ref();
  return LIBXML2_DOMImplementation::sDOMImplementation;
}

LIBXML2_Element* LIBXML2_Document::
documentElement()
{
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++)
    if ((*i)->nodeType() == LIBXML2_Node::ELEMENT_NODE) {
      LIBXML2_Element* el = static_cast<LIBXML2_Element*>(*i);
      el->add_ref();
      return el;
    }
  return NULL;
}

LIBXML2_Element* LIBXML2_Document::
createElement(const String& tagName)
{
  LIBXML2_Element* el = LIBXML2_NewElement(this, String(), tagName);
  if (el == NULL)
    throw Arcane::Exception::exception();
  el->mNodeName = tagName;
  return el;
}

LIBXML2_DocumentFragment* LIBXML2_Document::
createDocumentFragment()
{
  LIBXML2_DocumentFragment* df = new LIBXML2_DocumentFragment(this);
  if (df == NULL)
    throw Arcane::Exception::exception();
  return df;
}

LIBXML2_Text* LIBXML2_Document::
createTextNode(const String& data)
{
  LIBXML2_Text* tn = new LIBXML2_Text(this);
  if (tn == NULL)
    throw Arcane::Exception::exception();
  tn->mNodeValue = data;
  return tn;
}

LIBXML2_Comment* LIBXML2_Document::
createComment(const String& data)
{
  LIBXML2_Comment* com = new LIBXML2_Comment(this);
  if (com == NULL)
    throw Arcane::Exception::exception();
  com->mNodeValue = data;
  return com;
}

LIBXML2_CDATASection* LIBXML2_Document::
createCDATASection(const String& data)
{
  LIBXML2_CDATASection* cds = new LIBXML2_CDATASection(this);
  if (cds == NULL)
    throw Arcane::Exception::exception();
  cds->mNodeValue = data;
  return cds;
}

LIBXML2_ProcessingInstruction* LIBXML2_Document::
createProcessingInstruction(const String& target, const String& data)
{
  return new LIBXML2_ProcessingInstruction(this, target, data);
}

LIBXML2_Attr* LIBXML2_Document::
createAttribute(const String& name)
{
  LIBXML2_Attr* at = new LIBXML2_Attr(this);
  if (at == NULL)
    throw Arcane::Exception::exception();
  at->mNodeName = name;
  return at;
}

LIBXML2_EntityReference* LIBXML2_Document::
createEntityReference(const String& name)
{
  LIBXML2_EntityReference* er = new LIBXML2_EntityReference(this);
  if (er == NULL)
    throw Arcane::Exception::exception();
  er->mNodeName = name;
  return er;
}

LIBXML2_NodeList* LIBXML2_Document::
getElementsByTagName(const String& tagname)
{
  LIBXML2_NodeList* nl = new LIBXML2_NodeListDFSSearch(this, tagname);
  if (nl == NULL)
    throw Arcane::Exception::exception();
  return nl;
}

LIBXML2_Node* LIBXML2_Document::
importNode(LIBXML2_Node* importedNode, bool deep)
{
  // the incoming node implements our interface but is not from this implementation.
  // In this case, we would have to do the clone ourselves using only
  // the standard interfaces.
  // Next we need to change the document all the way through...
  LIBXML2_Node* n = importedNode->cloneNode(deep);
  // It makes no sense to change the document associated with a document...
  if (n->nodeType() == LIBXML2_Node::DOCUMENT_NODE)
    throw Arcane::Exception::exception();
  n->recursivelyChangeDocument(this);
  n->add_ref();
  return n;
}

LIBXML2_Element* LIBXML2_Document::
createElementNS(const String& namespaceURI, const String& qualifiedName)
{
  const XMLCh* cqname = reinterpret_cast<const XMLCh*>(qualifiedName.utf16().begin());
  const XMLCh* cpos = wcschr(cqname, L':');
  if (cpos == NULL)
    cpos = cqname;
  else
    cpos++;
  ByteUniqueArray utf8_array(cpos[0]);
  String localName = String::fromUtf8(utf8_array);
  LIBXML2_Element* el = LIBXML2_NewElement(this, namespaceURI, localName);
  el->mNamespaceURI = namespaceURI;
  el->mNodeName = qualifiedName;
  el->mLocalName = localName;
  return el;
}

LIBXML2_Attr* LIBXML2_Document::
createAttributeNS(const String& namespaceURI, const String& qualifiedName)
{
  LIBXML2_Attr* at = new LIBXML2_Attr(mDocument);
  if (at == NULL)
    throw Arcane::Exception::exception();
  at->mNamespaceURI = namespaceURI;
  at->mNodeName = qualifiedName;
  const XMLCh* cqname = reinterpret_cast<const XMLCh*>(qualifiedName.utf16().begin());
  const XMLCh* cpos = wcschr(cqname, L':');
  if (cpos == NULL)
    cpos = cqname;
  else
    cpos++;
  ByteUniqueArray utf8_array(cpos[0]);
  String localName = String::fromUtf8(utf8_array);
  at->mLocalName = localName;
  return at;
}

LIBXML2_NodeList* LIBXML2_Document::
getElementsByTagNameNS(const String& namespaceURI, const String& localName)
{
  LIBXML2_NodeListDFSSearch* nlDFSS = new LIBXML2_NodeListDFSSearch(this, namespaceURI, localName);
  if (nlDFSS == NULL)
    throw Arcane::Exception::exception();
  return nlDFSS;
}

LIBXML2_Element* LIBXML2_Document::
getElementById(const String& elementId)
{
  return searchForElementById(elementId);
}

LIBXML2_Node* LIBXML2_Document::
shallowCloneNode(LIBXML2_Document* aDoc)
{
  // TODO: regarder pourquoi aDoc n'est pas utilisé.
  LIBXML2_Document* Doc = new LIBXML2_Document();
  if (Doc == NULL)
    throw Arcane::Exception::exception();
  Doc->mLocalName = mLocalName;
  Doc->mNodeValue = mNodeValue;
  return Doc;
}

LIBXML2_Element* LIBXML2_Document::
searchForElementById(const String& elementId)
{
  for (std::list<LIBXML2_Node*>::iterator i = mNodeList.begin(); i != mNodeList.end(); i++) {
    LIBXML2_Element* el = (*i)->searchForElementById(elementId);
    if (el != NULL)
      return el;
  }
  return NULL;
}

// These tables were adapted from the Mozilla code.
static const XMLCh* kEntities[] = {
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),     U(""), U(""),    U(""), U(""),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),     U(""), U(""),    U(""), U(""),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),     U(""), U(""),    U(""), U("&amp;"),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),     U(""), U(""),    U(""), U(""),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U("&lt;"), U(""), U("&gt;")
};
static const XMLCh* kAttrEntities[] = {
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),       U(""), U(""),    U(""), U(""),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),       U(""), U(""),    U(""), U(""),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U("&quot;"), U(""), U(""),    U(""), U("&amp;"),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""),       U(""), U(""),    U(""), U(""),
  U(""), U(""), U(""), U(""), U(""), U(""), U(""), U(""), U("&lt;"),   U(""), U("&gt;")
};

String
TranslateEntities(const String& data, bool isAttribute = false)
{
  const XMLCh* Data = reinterpret_cast<const XMLCh*>(data.utf16().begin());
  const XMLCh** table;
  ustring o;
  if (isAttribute)
    table = kAttrEntities;
  else
    table = kEntities;

  UInt32 i, l = data.len();
  XMLCh c;
  for (i = 0; i < l; i++) {
    c = Data[i];
    if (c > 62 || (table[c][0] == 0)) {
      o += c;
      continue;
    }
    o += table[c];
  }
  const XMLCh* p = o.c_str();
  l = o.size();
  char* tp = new char[l + 1];
  size_t w = wcstombs(tp, p, l);
  if (w != l) {
    delete[] tp;
    return String();
  } else {
    tp[l] = '\0';
    std::string ret(tp);
    delete[] tp;
    return String(ret);
  }
}

void LIBXML2_DOMWriter::
writeNode(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Node* n, StringBuilder& appendTo)
{
  UInt16 nt = n->nodeType();

#define NODETYPE_CODE(nt, ntn)                                                                               \
  case LIBXML2_Node::nt##_NODE: {                                                                            \
    LIBXML2_##ntn* t = dynamic_cast<LIBXML2_##ntn*>(n);                                                      \
    if (t == NULL)                                                                                           \
      throw Arcane::Exception::exception();                                                                  \
    DeepNode++;                                                                                              \
    write##ntn(dnc, t, appendTo);                                                                            \
  } break;

  switch (nt) {
    NODETYPE_CODE(ELEMENT, Element)
    NODETYPE_CODE(ATTRIBUTE, Attr)
    NODETYPE_CODE(TEXT, Text)
    NODETYPE_CODE(CDATA_SECTION, CDATASection)
    NODETYPE_CODE(ENTITY_REFERENCE, EntityReference)
    NODETYPE_CODE(ENTITY, Entity)
    NODETYPE_CODE(PROCESSING_INSTRUCTION, ProcessingInstruction)
    NODETYPE_CODE(COMMENT, Comment)
    NODETYPE_CODE(DOCUMENT, Document)
    NODETYPE_CODE(DOCUMENT_TYPE, DocumentType)
    NODETYPE_CODE(DOCUMENT_FRAGMENT, DocumentFragment)
    NODETYPE_CODE(NOTATION, Notation)
  }
}

void LIBXML2_DOMWriter::
DoIndentation(bool StartEnd, StringBuilder& appendTo)
{
  /// Indentation Management
  int limite = 0;
  bool check = (DeepNode >= 1);
  if ((indent) && check) {
    if (!appendTo.toString().endsWith("\n")){ // try to find last CR if exist 3 = 2 bytes for CR in unicode + 1
      appendTo += "\n"; // adds CR;
    }
    if (StartEnd)
      limite = 1;
    else
      limite = 2;
    for ( int i = 0; i + limite < DeepNode; i++)
      appendTo += " "; // Space;
  }
}

void LIBXML2_DOMWriter::
writeElement(LIBXML2_DOMNamespaceContext* parentContext, LIBXML2_Element* el,
             StringBuilder& appendTo)
{
  LIBXML2_DOMNamespaceContext elementContext(parentContext);
  // Firstly scan for xmlns attributes...
  LIBXML2_NamedNodeMap* elnl = el->attributes();
  UInt32 l = elnl->length(), i;

  // See if this element has a prefix...
  String elpr = el->prefix();
  String elns = el->namespaceURI();
  if (elpr != String()) {
    // See if it is already defined...
    String existns = elementContext.findNamespaceForPrefix(elpr);
    if (existns == String()) {
      // They suggested a prefix, and it is available. Add it.
      elementContext.recordPrefix(elpr, elns);
    } else if (existns != elns) {
      // Can't use the suggested prefix. Do we need a prefix anyway?
      if (elementContext.getDefaultNamespace() == elns)
        elpr = String();
      else {
        elpr = elementContext.findPrefixForNamespace(elns);
        if (elpr == String()) {
          // Can't use any existing prefix. Set the default namespace instead.
          elementContext.setDefaultNamespace(elns);
        }
        // otherwise, don't need to do anything as elpr is now suitably defined.
      }
    }
    // otherwise don't need to do anything, as prefix is already valid.
  } else {
    // Only do anything if the namespace is non-default...
    if (elementContext.getDefaultNamespace() != elns) {
      // We have an element in the non-default namespace, with no suggested
      // prefix. See if there is an existing prefix for the namespace...
      elpr = elementContext.findPrefixForNamespace(elns);
      if (elpr == String()) {
        // Set the default namespace to the one used for this element.
        elementContext.setDefaultNamespace(elns);
      }
      // otherwise, elpr is valid.
    }
    // otherwise elpr=String() and the element is in the default namespace.
  }

  // Start the tag...
  DoIndentation(true, appendTo);
  appendTo += "<";
  StringBuilder qname;

  qname += elpr;
  if (elpr != String())
    qname += ":";
  String ln = el->localName();
  if (ln == String())
    ln = el->nodeName();
  qname += ln;
  appendTo += qname;
  for (i = 0; i < l; i++) {
    LIBXML2_Node* atn = elnl->item(i);
    if (atn == NULL)
      break;
    LIBXML2_Attr* at = dynamic_cast<LIBXML2_Attr*>(atn);
    // See if the attribute is in the XMLNS namespace...
    String nsURI = at->namespaceURI();
    String ln = at->localName(); // CHECK (hide previous ln variable)
    if (ln == String())
      ln = at->nodeName();
    if (nsURI == "http://www.w3.org/2000/xmlns/" ||
        /* This is tecnically incorrect but needed in practice... */
        (nsURI == String() && ln == "xmlns")) {
      String value = at->value();
      if (ln == "xmlns")
        elementContext.setDefaultNamespace(value);
      else
        elementContext.recordPrefix(ln, value);
    }
  }
  for (i = 0; i < l; i++) {
    LIBXML2_Node* atn = elnl->item(i);
    if (atn == NULL)
      break;
    LIBXML2_Attr* at = dynamic_cast<LIBXML2_Attr*>(atn);
    // See if the attribute is in the XMLNS namespace...
    String nsURI = at->namespaceURI();
    ln = at->localName();
    if (ln == String())
      ln = at->nodeName();
    if (nsURI == "http://www.w3.org/2000/xmlns/")
      continue;
    /* This is tecnically incorrect but needed in practice... */
    if (nsURI == String() && ln == "xmlns")
      continue;
    // If the attribute has a prefix, see if it is defined...
    String atpr = at->prefix();
    String atns = at->namespaceURI();
    // If no namespace(like most attributes will probably have) do nothing...
    if (atns == String())
      continue;
    if (atpr != String()) {
      String existpr = elementContext.findNamespaceForPrefix(atpr);
      if (existpr == String()) {
        // They suggested a prefix, and it is available, so use it...
        elementContext.recordPrefix(atpr, atns);
      } else if (existpr != atns) {
        // Can't use the desired prefix, so see if we can find another suitable
        // prefix...
        atpr = elementContext.findPrefixForNamespace(atns);
        if (atpr == String())
          elementContext.possiblyInventPrefix(atns);
      }
      // otherwise the prefix is correct and already defined.
    } else // no suggested prefix.
    {
      // if can't find a suitable prefix, invent one...
      atpr = elementContext.findPrefixForNamespace(atns);
      if (atpr == String())
        elementContext.possiblyInventPrefix(atns);
    }
  }
  elementContext.resolveOrInventPrefixes();
  elementContext.writeXMLNS(appendTo);
  // Now once more through the attributes...
  for (i = 0; i < l; i++) {
    LIBXML2_Node* atn = elnl->item(i);
    if (atn == NULL)
      break;
    LIBXML2_Attr* at = dynamic_cast<LIBXML2_Attr*>(atn);
    // See if the attribute is in the XMLNS namespace...
    String nsURI = at->namespaceURI();
    String ln = at->localName();
    if (ln == String())
      ln = at->nodeName();
    if (nsURI == "http://www.w3.org/2000/xmlns/")
      continue;
    /* This is tecnically incorrect but needed in practice... */
    if (nsURI == String() && ln == "xmlns")
      continue;
    // Write out this attribute...
    writeAttr(&elementContext, at, appendTo);
  }
  LIBXML2_NodeList* elcnl = el->childNodes();
  l = elcnl->length();
  if (l == 0) {
    appendTo += "/>";
    return;
  }
  appendTo += ">";

  // This time, we write everything except the attributes...
  for (i = 0; i < l; i++) {
    LIBXML2_Node* n = elcnl->item(i);
    if (n == NULL)
      break;
    // Gdome doesn't put attributes on child nodes. I'm not sure if this is the
    // correct interpretation of the DOM specification.
    // LIBXML2_Attr* at = dynamic_cast<LIBXML2_Attr*> (n);
    // if (at != NULL)
    //  continue;
    writeNode(&elementContext, n, appendTo);
    DeepNode--;
  }
  appendTo += "</" + qname + ">";
  /// Indentation Management.
  DoIndentation(false, appendTo);
}

void LIBXML2_DOMWriter::
writeAttr(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Attr* at, StringBuilder& appendTo)
{
  // Always put a space first...
  appendTo += " ";
  // Next, we might need a prefix.
  String atpr = at->prefix();
  String atns = at->namespaceURI();
  // If no namespace(like most attributes will probably have) use empty prefix.
  if (atns == String())
    atpr = String();
  else if (atpr != String()) {
    String existpr = dnc->findNamespaceForPrefix(atpr);
    if (existpr != atns) {
      atpr = dnc->findPrefixForNamespace(atpr);
    }
    // otherwise we use the specified prefix.
  } else
    atpr = dnc->findPrefixForNamespace(atns);
  if (atpr != String()) {
    appendTo += atpr;
    appendTo += ":";
  }
  String ln = at->localName();
  if (ln == String())
    ln = at->nodeName();
  appendTo += ln;
  appendTo += "=\"";
  appendTo += TranslateEntities(at->value(), true);
  appendTo += "\"";
}

void LIBXML2_DOMWriter::
writeText(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Text* txt, StringBuilder& appendTo)
{
  //TODO: Regarder pourquoi dnc n'est pas utilisé
  appendTo += TranslateEntities(txt->Data());
}

void LIBXML2_DOMWriter::
writeCDATASection(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_CDATASection* cds,
                  StringBuilder& appendTo)
{
  //TODO: Regarder pourquoi dnc n'est pas utilisé
  appendTo += "<![CDATA[";
  appendTo += cds->Data();
  appendTo += "]]>";
  appendTo += "\n";
}

void LIBXML2_DOMWriter::
writeEntity(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Entity* er, StringBuilder& appendTo)
{
  ARCANE_UNUSED(dnc);
  ARCANE_UNUSED(er);
  ARCANE_UNUSED(appendTo);
  throw NotImplementedException(A_FUNCINFO);
}

void LIBXML2_DOMWriter::
writeEntityReference(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_EntityReference* er,
                     StringBuilder& appendTo)
{
  ARCANE_UNUSED(dnc);
  ARCANE_UNUSED(er);
  ARCANE_UNUSED(appendTo);
  throw NotImplementedException(A_FUNCINFO);
}

void LIBXML2_DOMWriter::
writeProcessingInstruction(LIBXML2_DOMNamespaceContext* dnc,
                           LIBXML2_ProcessingInstruction* proci, StringBuilder& appendTo)
{
  appendTo += "<?";
  appendTo += proci->target();
  String data = proci->Data();
  if (data.len()) {
    appendTo += " " + data;
  }
  appendTo += "?>";
  appendTo += "\n";
}

void LIBXML2_DOMWriter::
writeComment(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Comment* comment,
             StringBuilder& appendTo)
{
  appendTo += "<!--";
  appendTo += comment->Data();
  appendTo += "-->";
  appendTo += "\n";
}

void LIBXML2_DOMWriter::
writeDocument(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Document* doc,
              StringBuilder& appendTo)
{
  // Firstly write the header...
  appendTo = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n";
  if (indent)
    appendTo += "\n";
  LIBXML2_NodeList* elnl = doc->childNodes();
  UInt32 l = elnl->length(), i;
  for (i = 0; i < l; i++) {
    LIBXML2_Node* n = elnl->item(i);
    if (n == NULL)
      break;
    writeNode(dnc, n, appendTo);
  }
}

void LIBXML2_DOMWriter::
writeDocumentType(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_DocumentType* dt,
                  StringBuilder& appendTo)
{
  ARCANE_UNUSED(dnc);
  ARCANE_UNUSED(dt);
  ARCANE_UNUSED(appendTo);
  throw NotImplementedException(A_FUNCINFO);
}

void LIBXML2_DOMWriter::
writeDocumentFragment(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_DocumentFragment* df,
                      StringBuilder& appendTo)
{
  LIBXML2_NodeList* elnl = df->childNodes();
  UInt32 l = elnl->length(), i;
  for (i = 0; i < l; i++) {
    LIBXML2_Node* n = elnl->item(i);
    if (n == NULL)
      break;
    writeNode(dnc, n, appendTo);
  }
}

void LIBXML2_DOMWriter::
writeNotation(LIBXML2_DOMNamespaceContext* dnc, LIBXML2_Notation* nt,
              StringBuilder& appendTo)
{
  ARCANE_UNUSED(dnc);
  ARCANE_UNUSED(nt);
  ARCANE_UNUSED(appendTo);
  throw NotImplementedException(A_FUNCINFO);
}

LIBXML2_DOMNamespaceContext::
LIBXML2_DOMNamespaceContext(LIBXML2_DOMNamespaceContext* aParent)
: mParent(aParent)
, mOverrideDefaultNamespace(false)
{}

void LIBXML2_DOMNamespaceContext::
setDefaultNamespace(const String& newNS)
{
  if (!mOverrideDefaultNamespace) {
    mOverrideDefaultNamespace = true;
    mDefaultNamespace = newNS;
  }
}

String LIBXML2_DOMNamespaceContext::
getDefaultNamespace()
{
  if (mOverrideDefaultNamespace)
    return mDefaultNamespace;
  if (mParent == NULL)
    return String();
  return mParent->getDefaultNamespace();
}

String LIBXML2_DOMNamespaceContext::
findNamespaceForPrefix(const String& prefix)
{
  std::map<String, String>::iterator i;
  i = mURIfromPrefix.find(prefix);
  if (i != mURIfromPrefix.end())
    return (*i).second;
  if (mParent == NULL)
    return String();
  return mParent->findNamespaceForPrefix(prefix);
}

String LIBXML2_DOMNamespaceContext::
findPrefixForNamespace(const String& ns)
{
  std::map<String, String>::iterator i;
  i = mPrefixfromURI.find(ns);
  if (i != mPrefixfromURI.end())
    return (*i).second;
  if (mParent == NULL) {
    // Some prefixes are built in...
    if (ns == "http://www.w3.org/XML/1998/namespace")
      return "xml";
    return String();
  }
  return mParent->findPrefixForNamespace(ns);
}

void LIBXML2_DOMNamespaceContext::
recordPrefix(const String& prefix, const String& ns)
{
  mURIfromPrefix.insert(std::pair<String, String>(prefix, ns));
  mPrefixfromURI.insert(std::pair<String, String>(ns, prefix));
}

void LIBXML2_DOMNamespaceContext::
possiblyInventPrefix(const String& prefix)
{
  mNamespacesNeedingPrefixes.push_back(prefix);
}

void LIBXML2_DOMNamespaceContext::
resolveOrInventPrefixes()
{
  while (!mNamespacesNeedingPrefixes.empty()) {
    String nnp = mNamespacesNeedingPrefixes.back();
    mNamespacesNeedingPrefixes.pop_back();
    // If nnp resolves, we are done...
    if (findPrefixForNamespace(nnp) != String())
      continue;
    String suggestion;
    // We now need to invent a name. If we see a well-known namespace, we try
    // to use a common prefix first.
    if (nnp == "http://www.w3.org/1999/xlink")
      suggestion = "xlink";
    else if (nnp == "http://www.w3.org/1999/xhtml")
      suggestion = "html";
    else
      suggestion = "ns";
    StringBuilder attempt = suggestion;
    UInt32 attemptCount = 0;
    while (findPrefixForNamespace(attempt) != String()) {
      attempt = suggestion;
      String buf = String::format("{0}", attemptCount++);
      attempt += buf;
    }
    // We found an unique prefix...
    recordPrefix(attempt, nnp);
  }
}

void LIBXML2_DOMNamespaceContext::
writeXMLNS(StringBuilder& appendTo)
{
  std::map<String, String>::iterator i;
  // If we have changed our default prefix, set that here...
  if (mOverrideDefaultNamespace) {
    appendTo += " xmlns=\"" + TranslateEntities(mDefaultNamespace) + "\"";
  }
  for (i = mURIfromPrefix.begin(); i != mURIfromPrefix.end(); i++) {
    const String prefix = (*i).first;
    if (prefix == "xml")
      continue;
    appendTo += " xmlns:" + prefix + "=\"" + TranslateEntities((*i).second, true) + "\"";
  }
}

LIBXML2_Node*
WrapXML2Node(LIBXML2_Document* doc, xmlNode* x2node)
{
  LIBXML2_Document* docR;
  LIBXML2_Node* n;
  StringBuilder localname, namespaceURI;

  if (x2node->name != NULL) {
    localname = reinterpret_cast<const char*>(x2node->name);
  }
  bool setNS = false;
  bool isNodeStructure = false;
  switch (x2node->type) {
  case XML_ELEMENT_NODE: {
    setNS = true;
    if (x2node->ns && x2node->ns->href) {
      namespaceURI += reinterpret_cast<const char*>(x2node->ns->href);
    }
    isNodeStructure = true;
    LIBXML2_Element* el = LIBXML2_NewElement(doc, namespaceURI, localname);
    if (el == NULL)
      throw Arcane::Exception::exception();
    n = el;
    for (xmlAttr* at = x2node->properties; at != NULL; at = at->next) {
      LIBXML2_Attr* cattr = new LIBXML2_Attr(doc);
      if (cattr == NULL)
        throw Arcane::Exception::exception();
      String nsURI, prefix, localname;
      if (at->ns && at->ns->href) {
        nsURI = reinterpret_cast<const char*>(at->ns->href);
        prefix = reinterpret_cast<const char*>(at->ns->prefix);
      }
      localname = reinterpret_cast<const char*>(at->name);

      cattr->mLocalName = localname;
      if (prefix != String())
        cattr->mNodeName = prefix + ":" + localname;
      else
        cattr->mNodeName = localname;
      cattr->mNamespaceURI = nsURI;
      xmlNode* tmp;
      for (tmp = at->children; tmp != NULL; tmp = tmp->next) {
        LIBXML2_Node* nw = WrapXML2Node(doc, tmp);
        if (nw != NULL)
          cattr->appendChild(nw)->release_ref();
      }
      // Finally, fetch the value as a single string...
      if (at->ns != NULL && at->ns->href != NULL)
        cattr->mNodeValue = reinterpret_cast<const char*>(xmlGetNsProp(at->parent, at->name, at->ns->href));
      else
        cattr->mNodeValue = reinterpret_cast<const char*>(xmlGetNoNsProp(at->parent, at->name));
      el->attributeMapNS.insert(std::pair<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>(
              LIBXML2_Element::QualifiedName(nsURI, localname), cattr));
      el->attributeMap.insert(std::pair<LIBXML2_Element::LocalName, LIBXML2_Attr*>(
              LIBXML2_Element::LocalName(cattr->mNodeName), cattr));
      el->insertBeforePrivate(cattr, NULL)->release_ref();
    }

    for (xmlNs* nsd = x2node->nsDef; nsd != NULL; nsd = nsd->next) {
      LIBXML2_Attr* cattr = new LIBXML2_Attr(doc);
      if (cattr == NULL)
        throw Arcane::Exception::exception();
      StringBuilder nsURI, prefix, localname;
      if (nsd->prefix && strcmp(reinterpret_cast<const char*>(nsd->prefix), "")) {
        nsURI += "http://www.w3.org/2000/xmlns/";
        prefix += "xmlns";
        localname += reinterpret_cast<const char*>(nsd->prefix);
      } else
        localname += "xmlns";
      cattr->mLocalName = localname;
      if (prefix != String()) {
        prefix += ":" + localname;
        cattr->mNodeName = prefix;
      } else
        cattr->mNodeName = localname;
      cattr->mNamespaceURI = nsURI;

      cattr->mNodeValue = reinterpret_cast<const char*>(nsd->href);
      el->attributeMapNS.insert(std::pair<LIBXML2_Element::QualifiedName, LIBXML2_Attr*>(
              LIBXML2_Element::QualifiedName(nsURI, localname), cattr));
      el->attributeMap.insert(std::pair<LIBXML2_Element::LocalName, LIBXML2_Attr*>(
              LIBXML2_Element::LocalName(cattr->mNodeName), cattr));
      el->insertBeforePrivate(cattr, NULL)->release_ref();
    }
  } break;
  case XML_TEXT_NODE: {
    isNodeStructure = true;
    n = new LIBXML2_Text(doc);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_CDATA_SECTION_NODE: {
    isNodeStructure = true;
    n = new LIBXML2_CDATASection(doc);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_ENTITY_REF_NODE: {
    isNodeStructure = true;
    n = new LIBXML2_EntityReference(doc);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_ENTITY_NODE: {
    String notName, pubID, sysID;
    xmlEntity* se = reinterpret_cast<xmlEntity*>(x2node);
    notName = reinterpret_cast<const char*>(se->name);
    pubID = reinterpret_cast<const char*>(se->ExternalID);
    sysID = reinterpret_cast<const char*>(se->SystemID);
    n = new LIBXML2_Entity(doc, pubID, sysID, notName);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_PI_NODE: {
    isNodeStructure = true;
    n = new LIBXML2_ProcessingInstruction(doc, String(), String());
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_COMMENT_NODE: {
    isNodeStructure = true;
    n = new LIBXML2_Comment(doc);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_DOCUMENT_NODE:
#ifdef XML_DOCB_DOCUMENT_NODE
  case XML_DOCB_DOCUMENT_NODE:
#endif
#ifdef XML_HTML_DOCUMENT_NODE
  case XML_HTML_DOCUMENT_NODE:
#endif
  {
    // We need to decide what type of document before we add the children to
    // so do a preliminary scan...
    String rootNS;
    for (xmlNode* tnode = x2node->children; tnode; tnode = tnode->next) {
      if (tnode->type != XML_ELEMENT_NODE)
        continue;
      if (tnode->ns == NULL || tnode->ns->href == NULL)
        break;
      rootNS = reinterpret_cast<const char*>(tnode->ns->href);
      break;
    }
    docR = LIBXML2_NewDocument(rootNS);
    doc = docR;
    xmlDoc* sd = reinterpret_cast<xmlDoc*>(x2node);
    xmlDtd* dt = sd->intSubset;
    if (dt != NULL) {
      LIBXML2_Node* dtwrap = WrapXML2Node(doc, reinterpret_cast<xmlNode*>(dt));
      if (dtwrap != NULL)
        doc->insertBeforePrivate(dtwrap, NULL)->release_ref();
    }
    n = doc;
  } break;
  case XML_DOCUMENT_TYPE_NODE: {
    String pubID, sysID;
    xmlDtd* dt = reinterpret_cast<xmlDtd*>(x2node);
    pubID = reinterpret_cast<const char*>(dt->ExternalID);
    sysID = reinterpret_cast<const char*>(dt->SystemID);
    n = new LIBXML2_DocumentType(doc, localname, pubID, sysID);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  case XML_DOCUMENT_FRAG_NODE: {
    isNodeStructure = true;
    n = new LIBXML2_DocumentFragment(doc);
    if (n == NULL)
      throw Arcane::Exception::exception();
  } break;
  // Handled as part of element...
  case XML_ATTRIBUTE_NODE:
    // Handled as part of document type...
  case XML_NOTATION_NODE:
    // The remaining types are not part of the standard DOM...
  default:
    return NULL;
  }
  if (isNodeStructure) {
    if (!setNS) {
      if (x2node->ns && x2node->ns->href) {
        namespaceURI = reinterpret_cast<const char*>(x2node->ns->href);
      }
    }
    n->mLocalName = localname;
    if (x2node->ns && x2node->ns->href) {
      n->mNamespaceURI = namespaceURI;
    }
    if (x2node->ns && x2node->ns->prefix) {
      StringBuilder prefix;
      prefix = reinterpret_cast<const char*>(x2node->ns->prefix);
      prefix += ":" + localname;
      n->mNodeName = prefix;
    } else
      n->mNodeName = localname;
    if (x2node->content) {
      n->mNodeValue = reinterpret_cast<const char*>(x2node->content);
    }
  }
  // Now visit children...
  x2node = x2node->children;
  while (x2node) {
    LIBXML2_Node* wn = WrapXML2Node(doc, x2node);
    if (wn != NULL)
      n->insertBeforePrivate(wn, NULL)->release_ref();
    x2node = x2node->next;
  }
  n->add_ref();
  return n;
}

void LIBXML2_DOMImplementation::
ProcessContextError(String& aErrorMessage, xmlParserCtxtPtr ctxt)
{
  xmlErrorPtr err = xmlCtxtGetLastError(ctxt);
  if (err == NULL) {
    xmlFreeParserCtxt(ctxt);
    aErrorMessage = "Could not fetch the error message.";
    return;
  }
  ProcessXMLError(aErrorMessage, err);
  xmlFreeParserCtxt(ctxt);
}

void LIBXML2_DOMImplementation::
ProcessXMLError(String& aErrorMessage, xmlError* err)
{
  String fname, msg;
  String buf;
  if (err->file) {
    fname = reinterpret_cast<const char*>(err->file);
  }
  if (err->message) {
    msg = reinterpret_cast<const char*>(err->message);
  }

  switch (err->code) {
  case XML_ERR_INTERNAL_ERROR:
    aErrorMessage = "internalerror";
    break;
  case XML_ERR_NO_MEMORY:
    aErrorMessage = "nomemory";
    break;
  case XML_ERR_DOCUMENT_START:
  case XML_ERR_DOCUMENT_EMPTY:
  case XML_ERR_DOCUMENT_END:
  case XML_ERR_INVALID_HEX_CHARREF:
  case XML_ERR_INVALID_DEC_CHARREF:
  case XML_ERR_INVALID_CHARREF:
  case XML_ERR_INVALID_CHAR:
  case XML_ERR_CHARREF_AT_EOF:
  case XML_ERR_CHARREF_IN_PROLOG:
  case XML_ERR_CHARREF_IN_EPILOG:
  case XML_ERR_CHARREF_IN_DTD:
  case XML_ERR_ENTITYREF_AT_EOF:
  case XML_ERR_ENTITYREF_IN_PROLOG:
  case XML_ERR_ENTITYREF_IN_EPILOG:
  case XML_ERR_ENTITYREF_IN_DTD:
  case XML_ERR_PEREF_AT_EOF:
  case XML_ERR_PEREF_IN_PROLOG:
  case XML_ERR_PEREF_IN_EPILOG:
  case XML_ERR_PEREF_IN_INT_SUBSET:
  case XML_ERR_ENTITYREF_NO_NAME:
  case XML_ERR_ENTITYREF_SEMICOL_MISSING:
  case XML_ERR_PEREF_NO_NAME:
  case XML_ERR_PEREF_SEMICOL_MISSING:
  case XML_ERR_UNDECLARED_ENTITY:
  case XML_WAR_UNDECLARED_ENTITY:
  case XML_ERR_UNPARSED_ENTITY:
  case XML_ERR_ENTITY_IS_EXTERNAL:
  case XML_ERR_ENTITY_IS_PARAMETER:
  case XML_ERR_UNKNOWN_ENCODING:
  case XML_ERR_UNSUPPORTED_ENCODING:
  case XML_ERR_STRING_NOT_STARTED:
  case XML_ERR_STRING_NOT_CLOSED:
  case XML_ERR_NS_DECL_ERROR:
  case XML_ERR_ENTITY_NOT_STARTED:
  case XML_ERR_ENTITY_NOT_FINISHED:
  case XML_ERR_LT_IN_ATTRIBUTE:
  case XML_ERR_ATTRIBUTE_NOT_STARTED:
  case XML_ERR_ATTRIBUTE_NOT_FINISHED:
  case XML_ERR_ATTRIBUTE_WITHOUT_VALUE:
  case XML_ERR_ATTRIBUTE_REDEFINED:
  case XML_ERR_LITERAL_NOT_STARTED:
  case XML_ERR_LITERAL_NOT_FINISHED:
  case XML_ERR_COMMENT_NOT_FINISHED:
  case XML_ERR_PI_NOT_STARTED:
  case XML_ERR_PI_NOT_FINISHED:
  case XML_ERR_NOTATION_NOT_STARTED:
  case XML_ERR_NOTATION_NOT_FINISHED:
  case XML_ERR_ATTLIST_NOT_STARTED:
  case XML_ERR_ATTLIST_NOT_FINISHED:
  case XML_ERR_MIXED_NOT_STARTED:
  case XML_ERR_MIXED_NOT_FINISHED:
  case XML_ERR_ELEMCONTENT_NOT_STARTED:
  case XML_ERR_ELEMCONTENT_NOT_FINISHED:
  case XML_ERR_XMLDECL_NOT_STARTED:
  case XML_ERR_XMLDECL_NOT_FINISHED:
  case XML_ERR_CONDSEC_NOT_STARTED:
  case XML_ERR_CONDSEC_NOT_FINISHED:
  case XML_ERR_EXT_SUBSET_NOT_FINISHED:
  case XML_ERR_DOCTYPE_NOT_FINISHED:
  case XML_ERR_MISPLACED_CDATA_END:
  case XML_ERR_CDATA_NOT_FINISHED:
  case XML_ERR_RESERVED_XML_NAME:
  case XML_ERR_SPACE_REQUIRED:
  case XML_ERR_SEPARATOR_REQUIRED:
  case XML_ERR_NMTOKEN_REQUIRED:
  case XML_ERR_NAME_REQUIRED:
  case XML_ERR_PCDATA_REQUIRED:
  case XML_ERR_URI_REQUIRED:
  case XML_ERR_PUBID_REQUIRED:
  case XML_ERR_LT_REQUIRED:
  case XML_ERR_GT_REQUIRED:
  case XML_ERR_LTSLASH_REQUIRED:
  case XML_ERR_EQUAL_REQUIRED:
  case XML_ERR_TAG_NAME_MISMATCH:
  case XML_ERR_TAG_NOT_FINISHED:
  case XML_ERR_STANDALONE_VALUE:
  case XML_ERR_ENCODING_NAME:
  case XML_ERR_HYPHEN_IN_COMMENT:
  case XML_ERR_INVALID_ENCODING:
  case XML_ERR_EXT_ENTITY_STANDALONE:
  case XML_ERR_CONDSEC_INVALID:
  case XML_ERR_VALUE_REQUIRED:
  case XML_ERR_NOT_WELL_BALANCED:
  case XML_ERR_EXTRA_CONTENT:
  case XML_ERR_ENTITY_CHAR_ERROR:
  case XML_ERR_ENTITY_PE_INTERNAL:
  case XML_ERR_ENTITY_LOOP:
  case XML_ERR_ENTITY_BOUNDARY:
#if LIBXML_VERSION > 20403
  case XML_WAR_CATALOG_PI:
#endif
#if LIBXML_VERSION > 20404
  case XML_ERR_NO_DTD:
#endif
  case XML_ERR_CONDSEC_INVALID_KEYWORD:
  case XML_ERR_VERSION_MISSING:
  case XML_WAR_UNKNOWN_VERSION:
  case XML_WAR_LANG_VALUE:
  case XML_WAR_NS_URI:
  case XML_WAR_NS_URI_RELATIVE:
  case XML_ERR_MISSING_ENCODING:
#if LIBXML_VERSION > 20616
  case XML_WAR_SPACE_VALUE:
  case XML_ERR_NOT_STANDALONE:
  case XML_ERR_ENTITY_PROCESSING:
  case XML_ERR_NOTATION_PROCESSING:
  case XML_WAR_NS_COLUMN:
  case XML_WAR_ENTITY_REDEFINED:
    // case XML_NS_ERR_XML:
#if LIBXML_VERSION > 20600
  case XML_NS_ERR_UNDEFINED_NAMESPACE:
  case XML_NS_ERR_QNAME:
  case XML_NS_ERR_ATTRIBUTE_REDEFINED:
#endif
  case XML_NS_ERR_EMPTY:
#endif
    // case XML_DTD_ATTRIBUTE:
  case XML_DTD_ATTRIBUTE_REDEFINED:
  case XML_DTD_ATTRIBUTE_VALUE:
  case XML_DTD_CONTENT_ERROR:
  case XML_DTD_CONTENT_MODEL:
  case XML_DTD_CONTENT_NOT_DETERMINIST:
  case XML_DTD_DIFFERENT_PREFIX:
  case XML_DTD_ELEM_DEFAULT_NAMESPACE:
  case XML_DTD_ELEM_NAMESPACE:
  case XML_DTD_ELEM_REDEFINED:
  case XML_DTD_EMPTY_NOTATION:
  case XML_DTD_ENTITY_TYPE:
  case XML_DTD_ID_FIXED:
  case XML_DTD_ID_REDEFINED:
  case XML_DTD_ID_SUBSET:
  case XML_DTD_INVALID_CHILD:
  case XML_DTD_INVALID_DEFAULT:
  case XML_DTD_LOAD_ERROR:
  case XML_DTD_MISSING_ATTRIBUTE:
  case XML_DTD_MIXED_CORRUPT:
  case XML_DTD_MULTIPLE_ID:
  case XML_DTD_NO_DOC:
  case XML_DTD_NO_DTD:
  case XML_DTD_NO_ELEM_NAME:
  case XML_DTD_NO_PREFIX:
  case XML_DTD_NO_ROOT:
  case XML_DTD_NOTATION_REDEFINED:
  case XML_DTD_NOTATION_VALUE:
  case XML_DTD_NOT_EMPTY:
  case XML_DTD_NOT_PCDATA:
  case XML_DTD_NOT_STANDALONE:
  case XML_DTD_ROOT_NAME:
  case XML_DTD_STANDALONE_WHITE_SPACE:
  case XML_DTD_UNKNOWN_ATTRIBUTE:
  case XML_DTD_UNKNOWN_ELEM:
  case XML_DTD_UNKNOWN_ENTITY:
  case XML_DTD_UNKNOWN_ID:
  case XML_DTD_UNKNOWN_NOTATION:
  case XML_DTD_STANDALONE_DEFAULTED:
  case XML_DTD_XMLID_VALUE:
  case XML_DTD_XMLID_TYPE:
  case XML_XINCLUDE_PARSE_VALUE:
  case XML_XINCLUDE_ENTITY_DEF_MISMATCH:
  case XML_XINCLUDE_NO_HREF:
  case XML_XINCLUDE_NO_FALLBACK:
  case XML_XINCLUDE_HREF_URI:
  case XML_XINCLUDE_TEXT_FRAGMENT:
  case XML_XINCLUDE_TEXT_DOCUMENT:
  case XML_XINCLUDE_INVALID_CHAR:
  case XML_XINCLUDE_BUILD_FAILED:
  case XML_XINCLUDE_UNKNOWN_ENCODING:
  case XML_XINCLUDE_MULTIPLE_ROOT:
  case XML_XINCLUDE_XPTR_FAILED:
  case XML_XINCLUDE_XPTR_RESULT:
  case XML_XINCLUDE_INCLUDE_IN_INCLUDE:
  case XML_XINCLUDE_FALLBACKS_IN_INCLUDE:
  case XML_XINCLUDE_FALLBACK_NOT_IN_INCLUDE:
  case XML_XINCLUDE_DEPRECATED_NS:
  case XML_XINCLUDE_FRAGMENT_ID:
  // case XML_CATALOG_MISSING:
  case XML_CATALOG_ENTRY_BROKEN:
  case XML_CATALOG_PREFER_VALUE:
  case XML_CATALOG_NOT_CATALOG:
  case XML_CATALOG_RECURSION:
  // case XML_SCHEMAP_PREFIX:
  case XML_SCHEMAP_ATTRFORMDEFAULT_VALUE:
  case XML_SCHEMAP_ATTRGRP_NONAME_NOREF:
  case XML_SCHEMAP_ATTR_NONAME_NOREF:
  case XML_SCHEMAP_COMPLEXTYPE_NONAME_NOREF:
  case XML_SCHEMAP_ELEMFORMDEFAULT_VALUE:
  case XML_SCHEMAP_ELEM_NONAME_NOREF:
  case XML_SCHEMAP_EXTENSION_NO_BASE:
  case XML_SCHEMAP_FACET_NO_VALUE:
  case XML_SCHEMAP_FAILED_BUILD_IMPORT:
  case XML_SCHEMAP_GROUP_NONAME_NOREF:
  case XML_SCHEMAP_IMPORT_NAMESPACE_NOT_URI:
  case XML_SCHEMAP_IMPORT_REDEFINE_NSNAME:
  case XML_SCHEMAP_IMPORT_SCHEMA_NOT_URI:
  case XML_SCHEMAP_INVALID_BOOLEAN:
  case XML_SCHEMAP_INVALID_ENUM:
  case XML_SCHEMAP_INVALID_FACET:
  case XML_SCHEMAP_INVALID_FACET_VALUE:
  case XML_SCHEMAP_INVALID_MAXOCCURS:
  case XML_SCHEMAP_INVALID_MINOCCURS:
  case XML_SCHEMAP_INVALID_REF_AND_SUBTYPE:
  case XML_SCHEMAP_INVALID_WHITE_SPACE:
  case XML_SCHEMAP_NOATTR_NOREF:
  case XML_SCHEMAP_NOTATION_NO_NAME:
  case XML_SCHEMAP_NOTYPE_NOREF:
  case XML_SCHEMAP_REF_AND_SUBTYPE:
  case XML_SCHEMAP_RESTRICTION_NONAME_NOREF:
  case XML_SCHEMAP_SIMPLETYPE_NONAME:
  case XML_SCHEMAP_TYPE_AND_SUBTYPE:
  case XML_SCHEMAP_UNKNOWN_ALL_CHILD:
  case XML_SCHEMAP_UNKNOWN_ANYATTRIBUTE_CHILD:
  case XML_SCHEMAP_UNKNOWN_ATTR_CHILD:
  case XML_SCHEMAP_UNKNOWN_ATTRGRP_CHILD:
  case XML_SCHEMAP_UNKNOWN_ATTRIBUTE_GROUP:
  case XML_SCHEMAP_UNKNOWN_BASE_TYPE:
  case XML_SCHEMAP_UNKNOWN_CHOICE_CHILD:
  case XML_SCHEMAP_UNKNOWN_COMPLEXCONTENT_CHILD:
  case XML_SCHEMAP_UNKNOWN_COMPLEXTYPE_CHILD:
  case XML_SCHEMAP_UNKNOWN_ELEM_CHILD:
  case XML_SCHEMAP_UNKNOWN_EXTENSION_CHILD:
  case XML_SCHEMAP_UNKNOWN_FACET_CHILD:
  case XML_SCHEMAP_UNKNOWN_FACET_TYPE:
  case XML_SCHEMAP_UNKNOWN_GROUP_CHILD:
  case XML_SCHEMAP_UNKNOWN_IMPORT_CHILD:
  case XML_SCHEMAP_UNKNOWN_LIST_CHILD:
  case XML_SCHEMAP_UNKNOWN_NOTATION_CHILD:
  case XML_SCHEMAP_UNKNOWN_PROCESSCONTENT_CHILD:
  case XML_SCHEMAP_UNKNOWN_REF:
  case XML_SCHEMAP_UNKNOWN_RESTRICTION_CHILD:
  case XML_SCHEMAP_UNKNOWN_SCHEMAS_CHILD:
  case XML_SCHEMAP_UNKNOWN_SEQUENCE_CHILD:
  case XML_SCHEMAP_UNKNOWN_SIMPLECONTENT_CHILD:
  case XML_SCHEMAP_UNKNOWN_SIMPLETYPE_CHILD:
  case XML_SCHEMAP_UNKNOWN_TYPE:
  case XML_SCHEMAP_UNKNOWN_UNION_CHILD:
  case XML_SCHEMAP_ELEM_DEFAULT_FIXED:
  case XML_SCHEMAP_REGEXP_INVALID:
  case XML_SCHEMAP_FAILED_LOAD:
  case XML_SCHEMAP_NOTHING_TO_PARSE:
  case XML_SCHEMAP_NOROOT:
  case XML_SCHEMAP_REDEFINED_GROUP:
  case XML_SCHEMAP_REDEFINED_TYPE:
  case XML_SCHEMAP_REDEFINED_ELEMENT:
  case XML_SCHEMAP_REDEFINED_ATTRGROUP:
  case XML_SCHEMAP_REDEFINED_ATTR:
  case XML_SCHEMAP_REDEFINED_NOTATION:
  case XML_SCHEMAP_FAILED_PARSE:
  case XML_SCHEMAP_UNKNOWN_PREFIX:
  case XML_SCHEMAP_DEF_AND_PREFIX:
  case XML_SCHEMAP_UNKNOWN_INCLUDE_CHILD:
  case XML_SCHEMAP_INCLUDE_SCHEMA_NOT_URI:
  case XML_SCHEMAP_INCLUDE_SCHEMA_NO_URI:
  case XML_SCHEMAP_NOT_SCHEMA:
  case XML_SCHEMAP_UNKNOWN_MEMBER_TYPE:
  case XML_SCHEMAP_INVALID_ATTR_USE:
  case XML_SCHEMAP_RECURSIVE:
  case XML_SCHEMAP_SUPERNUMEROUS_LIST_ITEM_TYPE:
  case XML_SCHEMAP_INVALID_ATTR_COMBINATION:
  case XML_SCHEMAP_INVALID_ATTR_INLINE_COMBINATION:
  case XML_SCHEMAP_MISSING_SIMPLETYPE_CHILD:
  case XML_SCHEMAP_INVALID_ATTR_NAME:
  case XML_SCHEMAP_REF_AND_CONTENT:
  case XML_SCHEMAP_CT_PROPS_CORRECT_1:
  case XML_SCHEMAP_CT_PROPS_CORRECT_2:
  case XML_SCHEMAP_CT_PROPS_CORRECT_3:
  case XML_SCHEMAP_CT_PROPS_CORRECT_4:
  case XML_SCHEMAP_CT_PROPS_CORRECT_5:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_1:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_2_1_1:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_2_1_2:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_2_2:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_3:
  case XML_SCHEMAP_WILDCARD_INVALID_NS_MEMBER:
  case XML_SCHEMAP_INTERSECTION_NOT_EXPRESSIBLE:
  case XML_SCHEMAP_UNION_NOT_EXPRESSIBLE:
  case XML_SCHEMAP_SRC_IMPORT_3_1:
  case XML_SCHEMAP_SRC_IMPORT_3_2:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_4_1:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_4_2:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_4_3:
  case XML_SCHEMAP_COS_CT_EXTENDS_1_3:
  // case XML_SCHEMAV:
  case XML_SCHEMAV_UNDECLAREDELEM:
  case XML_SCHEMAV_NOTTOPLEVEL:
  case XML_SCHEMAV_MISSING:
  case XML_SCHEMAV_WRONGELEM:
  case XML_SCHEMAV_NOTYPE:
  case XML_SCHEMAV_NOROLLBACK:
  case XML_SCHEMAV_ISABSTRACT:
  case XML_SCHEMAV_NOTEMPTY:
  case XML_SCHEMAV_ELEMCONT:
  case XML_SCHEMAV_HAVEDEFAULT:
  case XML_SCHEMAV_NOTNILLABLE:
  case XML_SCHEMAV_EXTRACONTENT:
  case XML_SCHEMAV_INVALIDATTR:
  case XML_SCHEMAV_INVALIDELEM:
  case XML_SCHEMAV_NOTDETERMINIST:
  case XML_SCHEMAV_CONSTRUCT:
  case XML_SCHEMAV_INTERNAL:
  case XML_SCHEMAV_NOTSIMPLE:
  case XML_SCHEMAV_ATTRUNKNOWN:
  case XML_SCHEMAV_ATTRINVALID:
  case XML_SCHEMAV_VALUE:
  case XML_SCHEMAV_FACET:
  case XML_SCHEMAV_CVC_DATATYPE_VALID_1_2_1:
  case XML_SCHEMAV_CVC_DATATYPE_VALID_1_2_2:
  case XML_SCHEMAV_CVC_DATATYPE_VALID_1_2_3:
  case XML_SCHEMAV_CVC_TYPE_3_1_1:
  case XML_SCHEMAV_CVC_TYPE_3_1_2:
  case XML_SCHEMAV_CVC_FACET_VALID:
  case XML_SCHEMAV_CVC_LENGTH_VALID:
  case XML_SCHEMAV_CVC_MINLENGTH_VALID:
  case XML_SCHEMAV_CVC_MAXLENGTH_VALID:
  case XML_SCHEMAV_CVC_MININCLUSIVE_VALID:
  case XML_SCHEMAV_CVC_MAXINCLUSIVE_VALID:
  case XML_SCHEMAV_CVC_MINEXCLUSIVE_VALID:
  case XML_SCHEMAV_CVC_MAXEXCLUSIVE_VALID:
  case XML_SCHEMAV_CVC_TOTALDIGITS_VALID:
  case XML_SCHEMAV_CVC_FRACTIONDIGITS_VALID:
  case XML_SCHEMAV_CVC_PATTERN_VALID:
  case XML_SCHEMAV_CVC_ENUMERATION_VALID:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_2_1:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_2_2:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_2_3:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_2_4:
  case XML_SCHEMAV_CVC_ELT_1:
  case XML_SCHEMAV_CVC_ELT_2:
  case XML_SCHEMAV_CVC_ELT_3_1:
  case XML_SCHEMAV_CVC_ELT_3_2_1:
  case XML_SCHEMAV_CVC_ELT_3_2_2:
  case XML_SCHEMAV_CVC_ELT_4_1:
  case XML_SCHEMAV_CVC_ELT_4_2:
  case XML_SCHEMAV_CVC_ELT_4_3:
  case XML_SCHEMAV_CVC_ELT_5_1_1:
  case XML_SCHEMAV_CVC_ELT_5_1_2:
  case XML_SCHEMAV_CVC_ELT_5_2_1:
  case XML_SCHEMAV_CVC_ELT_5_2_2_1:
  case XML_SCHEMAV_CVC_ELT_5_2_2_2_1:
  case XML_SCHEMAV_CVC_ELT_5_2_2_2_2:
  case XML_SCHEMAV_CVC_ELT_6:
  case XML_SCHEMAV_CVC_ELT_7:
  case XML_SCHEMAV_CVC_ATTRIBUTE_1:
  case XML_SCHEMAV_CVC_ATTRIBUTE_2:
  case XML_SCHEMAV_CVC_ATTRIBUTE_3:
  case XML_SCHEMAV_CVC_ATTRIBUTE_4:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_3_1:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_3_2_1:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_3_2_2:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_4:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_5_1:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_5_2:
  case XML_SCHEMAV_ELEMENT_CONTENT:
  case XML_SCHEMAV_DOCUMENT_ELEMENT_MISSING:
  case XML_SCHEMAV_CVC_COMPLEX_TYPE_1:
  case XML_SCHEMAV_CVC_AU:
  case XML_SCHEMAV_CVC_TYPE_1:
  case XML_SCHEMAV_CVC_TYPE_2:
#if LIBXML_VERSION > 20616
  case XML_SCHEMAV_CVC_IDC:
  case XML_SCHEMAV_CVC_WILDCARD:
#endif
  // case XML_XPTR_UNKNOWN:
  case XML_XPTR_CHILDSEQ_START:
  case XML_XPTR_EVAL_FAILED:
  case XML_XPTR_EXTRA_OBJECTS:
  // case XML_C14N_CREATE:
  case XML_C14N_REQUIRES_UTF8:
  case XML_C14N_CREATE_STACK:
  case XML_C14N_INVALID_NODE:
#if LIBXML_VERSION > 20616
  case XML_C14N_UNKNOW_NODE:
  case XML_C14N_RELATIVE_NAMESPACE:
#endif
  case XML_SCHEMAP_SRC_SIMPLE_TYPE_2:
  case XML_SCHEMAP_SRC_SIMPLE_TYPE_3:
  case XML_SCHEMAP_SRC_SIMPLE_TYPE_4:
  case XML_SCHEMAP_SRC_RESOLVE:
  case XML_SCHEMAP_SRC_RESTRICTION_BASE_OR_SIMPLETYPE:
  case XML_SCHEMAP_SRC_LIST_ITEMTYPE_OR_SIMPLETYPE:
  case XML_SCHEMAP_SRC_UNION_MEMBERTYPES_OR_SIMPLETYPES:
  case XML_SCHEMAP_ST_PROPS_CORRECT_1:
  case XML_SCHEMAP_ST_PROPS_CORRECT_2:
  case XML_SCHEMAP_ST_PROPS_CORRECT_3:
  case XML_SCHEMAP_COS_ST_RESTRICTS_1_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_1_2:
  case XML_SCHEMAP_COS_ST_RESTRICTS_1_3_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_1_3_2:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_1_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_1_2:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_2_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_2_2:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_2_3:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_2_4:
  case XML_SCHEMAP_COS_ST_RESTRICTS_2_3_2_5:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_1_2:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_2_2:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_2_1:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_2_3:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_2_4:
  case XML_SCHEMAP_COS_ST_RESTRICTS_3_3_2_5:
  case XML_SCHEMAP_COS_ST_DERIVED_OK_2_1:
  case XML_SCHEMAP_COS_ST_DERIVED_OK_2_2:
  case XML_SCHEMAP_S4S_ELEM_NOT_ALLOWED:
  case XML_SCHEMAP_S4S_ELEM_MISSING:
  case XML_SCHEMAP_S4S_ATTR_NOT_ALLOWED:
  case XML_SCHEMAP_S4S_ATTR_MISSING:
  case XML_SCHEMAP_S4S_ATTR_INVALID_VALUE:
  case XML_SCHEMAP_SRC_ELEMENT_1:
  case XML_SCHEMAP_SRC_ELEMENT_2_1:
  case XML_SCHEMAP_SRC_ELEMENT_2_2:
  case XML_SCHEMAP_SRC_ELEMENT_3:
  case XML_SCHEMAP_P_PROPS_CORRECT_1:
  case XML_SCHEMAP_P_PROPS_CORRECT_2_1:
  case XML_SCHEMAP_P_PROPS_CORRECT_2_2:
  case XML_SCHEMAP_E_PROPS_CORRECT_2:
  case XML_SCHEMAP_E_PROPS_CORRECT_3:
  case XML_SCHEMAP_E_PROPS_CORRECT_4:
  case XML_SCHEMAP_E_PROPS_CORRECT_5:
  case XML_SCHEMAP_E_PROPS_CORRECT_6:
  case XML_SCHEMAP_SRC_INCLUDE:
  case XML_SCHEMAP_SRC_ATTRIBUTE_1:
  case XML_SCHEMAP_SRC_ATTRIBUTE_2:
  case XML_SCHEMAP_SRC_ATTRIBUTE_3_1:
  case XML_SCHEMAP_SRC_ATTRIBUTE_3_2:
  case XML_SCHEMAP_SRC_ATTRIBUTE_4:
  case XML_SCHEMAP_NO_XMLNS:
  case XML_SCHEMAP_NO_XSI:
  case XML_SCHEMAP_COS_VALID_DEFAULT_1:
  case XML_SCHEMAP_COS_VALID_DEFAULT_2_1:
  case XML_SCHEMAP_COS_VALID_DEFAULT_2_2_1:
  case XML_SCHEMAP_COS_VALID_DEFAULT_2_2_2:
  case XML_SCHEMAP_CVC_SIMPLE_TYPE:
  case XML_SCHEMAP_COS_CT_EXTENDS_1_1:
  case XML_SCHEMAP_SRC_IMPORT_1_1:
  case XML_SCHEMAP_SRC_IMPORT_1_2:
  case XML_SCHEMAP_SRC_IMPORT_2:
  case XML_SCHEMAP_SRC_IMPORT_2_1:
  case XML_SCHEMAP_SRC_IMPORT_2_2:
  case XML_SCHEMAP_INTERNAL:
  case XML_SCHEMAP_NOT_DETERMINISTIC:
  case XML_SCHEMAP_SRC_ATTRIBUTE_GROUP_1:
  case XML_SCHEMAP_SRC_ATTRIBUTE_GROUP_2:
  case XML_SCHEMAP_SRC_ATTRIBUTE_GROUP_3:
  case XML_SCHEMAP_MG_PROPS_CORRECT_1:
  case XML_SCHEMAP_MG_PROPS_CORRECT_2:
  case XML_SCHEMAP_SRC_CT_1:
  case XML_SCHEMAP_DERIVATION_OK_RESTRICTION_2_1_3:
  case XML_SCHEMAP_AU_PROPS_CORRECT_2:
  case XML_SCHEMAP_A_PROPS_CORRECT_2:
#if XML_VERSION > 20616
  case XML_SCHEMAP_C_PROPS_CORRECT:
  case XML_SCHEMAP_SRC_REDEFINE:
#if XML_VERSION > 20621
  case XML_SCHEMAP_SRC_IMPORT:
  case XML_SCHEMAP_WARN_SKIP_SCHEMA:
  case XML_SCHEMAP_WARN_UNLOCATED_SCHEMA:
  case XML_SCHEMAP_WARN_ATTR_REDECL_PROH:
  case XML_SCHEMAP_WARN_ATTR_POINTLESS_PROH:
  case XML_SCHEMAP_AG_PROPS_CORRECT:
  case XML_SCHEMAP_COS_CT_EXTENDS_1_2:
  case XML_SCHEMAP_AU_PROPS_CORRECT:
  case XML_SCHEMAP_A_PROPS_CORRECT_3:
  case XML_SCHEMAP_COS_ALL_LIMITED:
#endif
  case XML_MODULE_OPEN:
  case XML_MODULE_CLOSE:
#endif
  case XML_CHECK_FOUND_ATTRIBUTE:
  case XML_CHECK_FOUND_TEXT:
  case XML_CHECK_FOUND_CDATA:
  case XML_CHECK_FOUND_ENTITYREF:
  case XML_CHECK_FOUND_ENTITY:
  case XML_CHECK_FOUND_PI:
  case XML_CHECK_FOUND_COMMENT:
  case XML_CHECK_FOUND_DOCTYPE:
  case XML_CHECK_FOUND_FRAGMENT:
  case XML_CHECK_FOUND_NOTATION:
  case XML_CHECK_UNKNOWN_NODE:
  case XML_CHECK_ENTITY_TYPE:
  case XML_CHECK_NO_PARENT:
  case XML_CHECK_NO_DOC:
  case XML_CHECK_NO_NAME:
  case XML_CHECK_NO_ELEM:
  case XML_CHECK_WRONG_DOC:
  case XML_CHECK_NO_PREV:
  case XML_CHECK_WRONG_PREV:
  case XML_CHECK_NO_NEXT:
  case XML_CHECK_WRONG_NEXT:
  case XML_CHECK_NOT_DTD:
  case XML_CHECK_NOT_ATTR:
  case XML_CHECK_NOT_ATTR_DECL:
  case XML_CHECK_NOT_ELEM_DECL:
  case XML_CHECK_NOT_ENTITY_DECL:
  case XML_CHECK_NOT_NS_DECL:
  case XML_CHECK_NO_HREF:
  case XML_CHECK_WRONG_PARENT:
  case XML_CHECK_NS_SCOPE:
  case XML_CHECK_NS_ANCESTOR:
  case XML_CHECK_NOT_UTF8:
  case XML_CHECK_NO_DICT:
  case XML_CHECK_NOT_NCNAME:
  case XML_CHECK_OUTSIDE_DICT:
  case XML_CHECK_WRONG_NAME:
  case XML_CHECK_NAME_NOT_NULL:
#if LIBXML_VERSION > 20616
  case XML_I18N_NO_HANDLER:
  case XML_I18N_EXCESS_HANDLER:
  case XML_I18N_CONV_FAILED:
  case XML_I18N_NO_OUTPUT:
#endif
    aErrorMessage = "badxml/";
    buf = String::format("{0}", err->line);
    aErrorMessage = aErrorMessage + buf;
    aErrorMessage = aErrorMessage + "/0/";
    aErrorMessage = aErrorMessage + fname + "/";
    aErrorMessage = aErrorMessage + msg;
    break;
  case XML_IO_ENCODER:
  case XML_IO_FLUSH:
  case XML_IO_WRITE:
  case XML_IO_BUFFER_FULL:
  case XML_IO_LOAD_ERROR:
    aErrorMessage = "servererror";
    break;
  case XML_IO_EACCES:
    aErrorMessage = "servererror/EACCESS";
    break;
  case XML_IO_EAGAIN:
    aErrorMessage = "servererror/AGAIN";
    break;
  case XML_IO_EBADF:
    aErrorMessage = "servererror/BADF";
    break;
  case XML_IO_EBADMSG:
    aErrorMessage = "servererror/BADMSG";
    break;
  case XML_IO_EBUSY:
    aErrorMessage = "servererror/BUSY";
    break;
  case XML_IO_ECANCELED:
    aErrorMessage = "servererror/CANCELED";
    break;
  case XML_IO_ECHILD:
    aErrorMessage = "servererror/CHILD";
    break;
  case XML_IO_EDEADLK:
    aErrorMessage = "servererror/DEADLK";
    break;
  case XML_IO_EDOM:
    aErrorMessage = "servererror/DOM";
    break;
  case XML_IO_EEXIST:
    aErrorMessage = "servererror/EXIST";
    break;
  case XML_IO_EFAULT:
    aErrorMessage = "servererror/FAULT";
    break;
  case XML_IO_EFBIG:
    aErrorMessage = "servererror/FBIG";
    break;
  case XML_IO_EINPROGRESS:
    aErrorMessage = "servererror/INPROGRESS";
    break;
  case XML_IO_EINTR:
    aErrorMessage = "servererror/INTR";
    break;
  case XML_IO_EINVAL:
    aErrorMessage = "servererror/INVAL";
    break;
  case XML_IO_EIO:
    aErrorMessage = "servererror/IO";
    break;
  case XML_IO_EISDIR:
    aErrorMessage = "servererror/ISDIR";
    break;
  case XML_IO_EMFILE:
    aErrorMessage = "servererror/MFILE";
    break;
  case XML_IO_EMLINK:
    aErrorMessage = "servererror/MLINK";
    break;
  case XML_IO_EMSGSIZE:
    aErrorMessage = "servererror/MSGSIZE";
    break;
  case XML_IO_ENAMETOOLONG:
    aErrorMessage = "servererror/NAMETOOLONG";
    break;
  case XML_IO_ENFILE:
    aErrorMessage = "servererror/NFILE";
    break;
  case XML_IO_ENODEV:
    aErrorMessage = "servererror/NODEV";
    break;
  case XML_IO_ENOENT:
    aErrorMessage = "servererror/NOENT";
    break;
  case XML_IO_ENOEXEC:
    aErrorMessage = "servererror/NOEXEC";
    break;
  case XML_IO_ENOLCK:
    aErrorMessage = "servererror/NOLCK";
    break;
  case XML_IO_ENOMEM:
    aErrorMessage = "servererror/NOMEM";
    break;
  case XML_IO_ENOSPC:
    aErrorMessage = "servererror/NOSPC";
    break;
  case XML_IO_ENOSYS:
    aErrorMessage = "servererror/NOSYS";
    break;
  case XML_IO_ENOTDIR:
    aErrorMessage = "servererror/NOTDIR";
    break;
  case XML_IO_ENOTEMPTY:
    aErrorMessage = "servererror/NOTEMPTY";
    break;
  case XML_IO_ENOTSUP:
    aErrorMessage = "servererror/NOTSUP";
    break;
  case XML_IO_ENOTTY:
    aErrorMessage = "servererror/NOTTY";
    break;
  case XML_IO_ENXIO:
    aErrorMessage = "servererror/NXIO";
    break;
  case XML_IO_EPERM:
    aErrorMessage = "servererror/PERM";
    break;
  case XML_IO_EPIPE:
    aErrorMessage = "servererror/PIPE";
    break;
  case XML_IO_ERANGE:
    aErrorMessage = "servererror/RANGE";
    break;
  case XML_IO_EROFS:
    aErrorMessage = "servererror/ROFS";
    break;
  case XML_IO_ESPIPE:
    aErrorMessage = "servererror/SPIPE";
    break;
  case XML_IO_ESRCH:
    aErrorMessage = "servererror/SRCH";
    break;
  case XML_IO_ETIMEDOUT:
    aErrorMessage = "servererror/TIMEDOUT";
    break;
  case XML_IO_EXDEV:
    aErrorMessage = "servererror/XDEV";
    break;
  case XML_IO_NO_INPUT:
    aErrorMessage = "servererror/NOINPUT";
    break;
  case XML_IO_ENOTSOCK:
    aErrorMessage = "servererror/NOTSOCK";
    break;
  case XML_IO_EISCONN:
    aErrorMessage = "servererror/ISCONN";
    break;
  case XML_IO_ECONNREFUSED:
    aErrorMessage = "servererror/CONNREFUSED";
    break;
  case XML_IO_ENETUNREACH:
    aErrorMessage = "servererror/NETUNREACHABLE";
    break;
  case XML_IO_EADDRINUSE:
    aErrorMessage = "servererror/ADDRINUSE";
    break;
  case XML_IO_EALREADY:
    aErrorMessage = "servererror/ALREADY";
    break;
  case XML_IO_EAFNOSUPPORT:
    aErrorMessage = "servererror/AFNOSUPPORT";
    break;
  case XML_FTP_EPSV_ANSWER:
    aErrorMessage = "servererror/EPSV_ANSWER";
    break;
  case XML_FTP_ACCNT:
    aErrorMessage = "servererror/FTPACCOUNT";
    break;
  case XML_HTTP_USE_IP:
    aErrorMessage = "servererror/USE_IP";
    break;
  case XML_HTTP_UNKNOWN_HOST:
    aErrorMessage = "servererror/UNKNOWNHOST";
    break;
  case XML_ERR_INVALID_URI:
  case XML_ERR_URI_FRAGMENT:
#if LIBXML_VERSION > 20616
  case XML_FTP_URL_SYNTAX:
#endif
    aErrorMessage = "badurl";
    break;
  case XML_IO_NETWORK_ATTEMPT:
    aErrorMessage = "noperm";
    break;
  default:
    aErrorMessage = "unexpectedxml2error";
  }
}

ARCANE_END_NAMESPACE_DOM
ARCANE_END_NAMESPACE
