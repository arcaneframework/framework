// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AxlOptionsBuilder.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Classes pour créer dynamiquement des options du jeu de données.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/AxlOptionsBuilder.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/DomUtils.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/CaseNodeNames.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AxlOptionsBuilder
{
// TODO: traiter en JSON le cas où une option avec le même nom est présente
// plusieurs fois.

// TODO: utiliser une union dans option pour conserver les valeurs comme les
// réels sans les transformer immédiatement en chaîne de caractères.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneOptionImpl
{
  friend DocumentXmlWriter;
  friend DocumentJSONWriter;

 public:

  void add(const std::initializer_list<OneOption>& options)
  {
    for (const auto& o : options)
      m_options.add(o);
  }

 public:

  UniqueArray<OneOption> m_options;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneOption::
OneOption(Type type, const String& name, const OptionList& options)
: m_type(type)
, m_name(name)
{
  OptionList cloned_list = options.clone();
  m_sub_option = cloned_list.m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInstance::
ServiceInstance(const String& option_name, const String& service_name,
                const std::initializer_list<OneOption>& options)
: OneOption(Type::CO_ServiceInstance, option_name, OptionList{})
{
  m_sub_option->add(options);
  m_service_name = service_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInstance::
ServiceInstance(const String& option_name, const String& service_name,
                const OptionList& options)
: OneOption(Type::CO_ServiceInstance, option_name, options)
{
  m_service_name = service_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceInstance::
ServiceInstance(const String& option_name, const String& service_name)
: OneOption(Type::CO_ServiceInstance, option_name, String{})
{
  m_service_name = service_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Complex::
Complex(const String& name, const std::initializer_list<OneOption>& options)
: OneOption(Type::CO_Complex, name, OptionList{})
{
  m_sub_option->add(options);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Complex::
Complex(const String& name, const OptionList& options)
: OneOption(Type::CO_Complex, name, options)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OptionList::
OptionList()
: m_p(std::make_shared<OneOptionImpl>())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OptionList::
OptionList(const std::initializer_list<OneOption>& options)
: m_p(std::make_shared<OneOptionImpl>())
{
  add(options);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OptionList& OptionList::
add(const String& name, const OptionList& option)
{
  m_p->m_options.add(OneOption(OneOption::Type::CO_Complex, name, option));
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OptionList& OptionList::
add(const std::initializer_list<OneOption>& options)
{
  for (const auto& o : options)
    m_p->m_options.add(o);
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OptionList OptionList::
clone() const
{
  OptionList new_opt;
  new_opt.m_p->m_options = m_p->m_options;
  return new_opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Écrivain au format XML pour un jeu de données.
class DocumentXmlWriter
{
 public:

  static IXmlDocumentHolder* toXml(const Document& d)
  {
    auto* doc = domutils::createXmlDocument();
    XmlNode document_node = doc->documentNode();
    DocumentXmlWriter writer(d.language());
    CaseNodeNames& cnn = writer.m_case_node_names;
    XmlNode root_element = document_node.createAndAppendElement("root", String());
    root_element.setAttrValue(cnn.lang_attribute, d.language());
    XmlNode opt_element = root_element.createAndAppendElement("dynamic-options", String());
    writer._writeToXml(d.m_options.m_p.get(), opt_element);
    return doc;
  }

 private:

  void _writeToXml(OneOptionImpl* opt, XmlNode element)
  {
    for (OneOption& o : opt->m_options) {
      XmlNode current_element = element.createAndAppendElement(o.m_name, o.m_value);
      if (o.m_sub_option.get())
        _writeToXml(o.m_sub_option.get(), current_element);

      if (!o.m_service_name.null())
        current_element.setAttrValue("name", o.m_service_name);

      if (!o.m_function_name.null()) {
        String funcname_attr = m_case_node_names.function_ref;
        current_element.setAttrValue(funcname_attr, o.m_function_name);
      }
    }
  }

 private:

  explicit DocumentXmlWriter(const String& lang)
  : m_case_node_names(lang)
  {}

 private:

  CaseNodeNames m_case_node_names;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Écrivain au format JSON pour un jeu de données.
class DocumentJSONWriter
{
 public:

  static String toJSON(const Document& d)
  {
    DocumentJSONWriter writer(d.language());

    writer._write(d);
    return writer.m_json_writer.getBuffer();
  }

 private:

  void _write(const Document& doc)
  {
    JSONWriter::Object o(m_json_writer);
    m_json_writer.write("language",doc.language());
    m_json_writer.write("version","1");
    {
      JSONWriter::Object o2(m_json_writer,"options");
      _write(doc.m_options.m_p.get());
    }
  }

  void _write(OneOptionImpl* opt)
  {
    if (!opt)
      return;
    // TODO: traiter le cas où une option avec le même nom
    // est présente plusieurs fois.
    // En théorie on peut avoir plusieurs la même clé mais
    // ce n'est pas recommandé. Le mieux dans ce cas est
    // d'utiliser un tableau.
    for (OneOption& o : opt->m_options) {
      _write(o);
    }
  }

  void _write(OneOption& o)
  {
    OneOptionImpl* sub_options = o.m_sub_option.get();
    if (o.m_type==OneOption::Type::CO_ServiceInstance){
      JSONWriter::Object j1(m_json_writer,o.m_name);
      if (!o.m_service_name.null())
        m_json_writer.write("$name",o.m_service_name);
      _write(sub_options);
      return;
    }

    if (o.m_type==OneOption::Type::CO_Complex){
      JSONWriter::Object j1(m_json_writer,o.m_name);
      _write(sub_options);
      return;
    }

    // Valeur simple ou énuméréé ou étendu.
    // Si pas de table de marche, on fait directement { "clé" : "valeur" }.
    // Sinon on utilise un sous-objet:
    // {
    //   "$function" : "nom_de_la_fonction,
    //   "$value" : "valeur"
    // }

    if (!o.m_function_name.null()) {
      JSONWriter::Object j1(m_json_writer,o.m_name);
      m_json_writer.write(m_case_function_json_name,o.m_function_name);
      m_json_writer.write("$value",o.m_value);
      return;
    }

    m_json_writer.write(o.m_name, o.m_value);
  }

 private:

  explicit DocumentJSONWriter(const String& lang)
  : m_case_node_names(lang)
  {
    m_case_function_json_name = String("$") + m_case_node_names.function_ref;
  }

 private:

  CaseNodeNames m_case_node_names;
  JSONWriter m_json_writer;
  String m_case_function_json_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IXmlDocumentHolder*
documentToXml(const Document& d)
{
  return DocumentXmlWriter::toXml(d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::AxlOptionsBuilder

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void
_testAxlOptionsBuilder()
{
  using namespace AxlOptionsBuilder;
  std::cout << "TestOptionList\n";
  OptionList sub_opt({ Simple("max", 35),
                       Simple("min", 13.2),
                       Simple("test", 1) });

  OptionList service_opt({ Simple("service-option1", 42),
                           Simple("service-option2", -1.5) });

  OptionList dco({ Simple("toto", 3),
                   Simple("titi", 3.1).addFunction("func1"),
                   Simple("tutu", "Hello"),
                   Enumeration("a", "vx"),
                   Extended("extended1", "ext"),
                   ServiceInstance("my-service1", "TestService1",
                                   { Simple("service-option1", 25),
                                     Simple("service-option2", 3.2) }),
                   ServiceInstance("my-service2", "TestService2"),
                   // TODO ajouter service avec nom par défaut.
                   ServiceInstance("my-service3", "TestService3", service_opt),
                   Complex("sub-opt1",
                           { Simple("max", 25),
                             Simple("min", 23.2),
                             Simple("test", 4) }),
                   Complex("sub-opt2", sub_opt) });
  Document doc("fr", dco);
  ScopedPtrT<IXmlDocumentHolder> x(DocumentXmlWriter::toXml(doc));
  String s = x->save();
  std::cout << "VALUE:" << s << "\n";

  String ref_str = "<?xml version=\"1.0\"?>\n"
                   "<root xml:lang=\"fr\"><dynamic-options><toto>3</toto><titi fonction=\"func1\">3.1</titi>"
                   "<tutu>Hello</tutu><a>vx</a><extended1>ext</extended1><my-service1 name=\"TestService1\">"
                   "<service-option1>25</service-option1><service-option2>3.2</service-option2></my-service1>"
                   "<my-service2 name=\"TestService2\"/><my-service3 name=\"TestService3\">"
                   "<service-option1>42</service-option1><service-option2>-1.5</service-option2></my-service3>"
                   "<sub-opt1><max>25</max><min>23.2</min><test>4</test></sub-opt1><sub-opt2><max>35</max>"
                   "<min>13.2</min><test>1</test></sub-opt2></dynamic-options></root>\n";
  if (s != ref_str)
    ARCANE_FATAL("BAD VALUE v={0} expected={1}", s, ref_str);

  {
    String json_string = DocumentJSONWriter::toJSON(doc);
    {
      std::ofstream ofile("test1.json");
      ofile << json_string;
    }
    std::cout << "JSON=" << json_string << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
