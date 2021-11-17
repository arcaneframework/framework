using System;
using System.IO;

namespace Arcane.Axl
{
  /**
   * Classe implémentant le Design Pattern du visiteur pour générer le code C++ de
   * déclaration des classes pour les options de type complex.
   */
  public class CppClassDefinitionGenerator : IOptionInfoVisitor
  {
    public CppClassDefinitionGenerator(TextWriter stream)
    {
      m_stream = stream;
    }
    
    public virtual void VisitComplex(ComplexOptionInfo info)
    {
      foreach(Option o in info.Options){
        o.Accept(this);
      }
      bool has_interface = !String.IsNullOrEmpty(info.InterfaceType);
      // Pour test, génère les méthodes des interfaces tout le temps
      bool always_generate_interface_method = false;
      string interface_type = String.Empty;
      if (has_interface)
        interface_type = CppUtils.ConvertType(info.InterfaceType);
      if (info.Type == null)
        return;
      
      if (info.IsMulti) {
        string c_class_name;
        c_class_name = "CaseOption" + info.Type;
        string v_class_name;
        v_class_name = c_class_name + "Value";
        m_stream.Write("    struct " + v_class_name + "\n");
        if (has_interface)
          m_stream.Write("    : private " + interface_type + "\n");
        m_stream.Write("    {\n");
        m_stream.Write("    typedef " + v_class_name + " ThatClass;\n");
        m_stream.Write("    " + v_class_name + "(Arcane::ICaseMng* cm,");
        m_stream.Write("Arcane::ICaseOptionList* icl,const Arcane::XmlNode& element)");
        m_stream.Write(" : m_element(element)\n");
        
        CppOptionBuilderGenerator bg = new CppOptionBuilderGenerator(m_stream, "icl", "m_element");
        info.AcceptChildren(bg);
        
        m_stream.Write("    {\n");
        CppBuilderBodyGenerator bbg = new CppBuilderBodyGenerator(m_stream);
        info.AcceptChildren(bbg);
        m_stream.Write("    }\n");

        if (has_interface || always_generate_interface_method){
          CppInterfaceImplementationGenerator iig = new CppInterfaceImplementationGenerator(m_stream);
          info.AcceptChildren(iig);
        }
        if (has_interface){
          m_stream.Write("    public: "+interface_type+"* _interface() { return this; }\n");
        }
        m_stream.Write("    public:\n");
        m_stream.Write("     const ThatClass* operator->() const { return this; }\n");
        m_stream.Write("     const Arcane::XmlNode& element() const { return m_element; }\n");
        m_stream.Write("    private:\n");
        m_stream.Write("     Arcane::XmlNode m_element;\n");
        m_stream.Write("    public:\n");
        
        CppVariableDefinitionGenerator vdg = new CppVariableDefinitionGenerator(m_stream);
        info.AcceptChildren(vdg);
        
        m_stream.Write("    };\n");
        
        m_stream.Write("  class " + c_class_name + "\n");
        m_stream.Write("  : public Arcane::CaseOptionsMulti\n");
        m_stream.Write("  , private Arcane::ArrayView< ");
        m_stream.Write(v_class_name + "* >\n");
        m_stream.Write("  {\n");
        m_stream.Write("   public:\n");
        m_stream.Write("    typedef Arcane::ArrayView< " + v_class_name + "* > BaseClass;\n");
        m_stream.Write("    typedef " + v_class_name + " value_type;\n");
        m_stream.Write("   public:\n");
        m_stream.Write("    " + c_class_name);
        m_stream.Write("(Arcane::ICaseMng* cm,Arcane::ICaseOptionList* icl,const Arcane::String& s,");
        m_stream.Write("const Arcane::XmlNode& element)\n");
        m_stream.Write("    : Arcane::CaseOptionsMulti(cm,icl,s,element," + info.MinOccurs
                       + "," + info.MaxOccurs + ") {\n");
        CppCodeGenerator.writeNameTranslations(m_stream,
                                               info.m_alternative_names,
                                               "");
        m_stream.Write("}\n");
        m_stream.Write("    Arcane::ArrayView< {0}* > operator()()\n",v_class_name);
        m_stream.Write("    {\n");
        m_stream.Write("      return (*this);\n");
        m_stream.Write("    }\n");
        m_stream.Write("    void multiAllocate(const Arcane::XmlNodeList& elist){\n");
        m_stream.Write("      Arcane::Integer s = elist.size();\n");
        m_stream.Write("      " + v_class_name + "** v = 0;\n");
        m_stream.Write("      if (s!=0)\n");
        m_stream.Write("        v = new " + v_class_name + "*[s];\n");
        m_stream.Write("      _setArray(v,s);\n");
        m_stream.Write("      v = _ptr();\n");
        if (has_interface){
          m_stream.Write("      m_interfaces.resize(s);\n");
        }
        m_stream.Write("      for( Arcane::Integer i=0; i<s; ++i ){\n");
        m_stream.Write("        v[i] = new " + v_class_name);
        m_stream.Write(" (caseMng(),configList(),elist[i]);\n");
        if (has_interface){
          m_stream.Write("        m_interfaces[i] = v[i]->_interface();\n");
        }
        m_stream.Write("      }\n");
        m_stream.Write("    }\n");
        m_stream.Write("   public:\n");
        m_stream.Write("    ~" + c_class_name + "(){\n");
        m_stream.Write("      Arcane::Integer s = count();\n");
        m_stream.Write("      if (s==0)\n");
        m_stream.Write("        return;\n");
        m_stream.Write("      " + v_class_name + "** v = _ptr();\n");
        m_stream.Write("      for( Arcane::Integer i=0; i<s; ++i ){\n");
        m_stream.Write("        delete v[i];\n");
        m_stream.Write("      }\n");
        m_stream.Write("      delete[] v;");
        m_stream.Write("    }\n");
        m_stream.Write("   private:\n");
        m_stream.Write("   public:\n");
        m_stream.Write("    const " + v_class_name + "& operator[](Arcane::Integer i) const\n");
        m_stream.Write("    { return *(BaseClass::operator[](i)); }\n");
        m_stream.Write("    Arcane::Integer count() const\n");
        m_stream.Write("    { return BaseClass::size(); }\n");
        m_stream.Write("    Arcane::Integer size() const\n");
        m_stream.Write("    { return BaseClass::size(); }\n");
        if (has_interface){
          m_stream.Write("  public: Arcane::ConstArrayView< " + interface_type + "* > _interface() { return m_interfaces; }\n");
          m_stream.Write("  private: Arcane::UniqueArray< " + interface_type + "* > m_interfaces;\n");
        }
        m_stream.Write("   private:\n");
        m_stream.Write("  };\n");
        m_stream.Write('\n');
      }
      else {
        string c_class_name;
        c_class_name = "CaseOption" + info.Type;
        m_stream.Write("  class " + c_class_name + "\n");
        m_stream.Write("  : public Arcane::CaseOptions");
        if (has_interface)
          m_stream.Write("  , private " + info.InterfaceType + '\n');
        m_stream.Write("  {\n");
        m_stream.Write("   public:\n");
        m_stream.Write("    " + c_class_name);
        m_stream.Write("(Arcane::ICaseMng* cm,Arcane::ICaseOptionList* icl,const Arcane::String& s,");
        m_stream.Write("const Arcane::XmlNode& element,bool is_optional=false)\n");
        m_stream.Write("    : Arcane::CaseOptions(icl,s,element,is_optional)\n");
        CppOptionBuilderGenerator bg = new CppOptionBuilderGenerator(m_stream, "configList()", "Arcane::XmlNode(0)");
        info.AcceptChildren(bg);
        m_stream.Write("    {\n");
        CppCodeGenerator.writeNameTranslations(m_stream,
                                               info.m_alternative_names,
                                               "");
        CppBuilderBodyGenerator bbg = new CppBuilderBodyGenerator(m_stream);
        info.AcceptChildren(bbg);
        
        m_stream.Write("}\n");
        m_stream.Write("    const " + c_class_name + "& operator()() const { return *this; }\n");
        m_stream.Write("   public:\n");
        if (has_interface){
          m_stream.Write("    "+interface_type+"* _interface() { return this; }\n");
        }
        if (has_interface || always_generate_interface_method){
          CppInterfaceImplementationGenerator iig = new CppInterfaceImplementationGenerator(m_stream);
          info.AcceptChildren(iig);
        }
        m_stream.Write("   public:\n");
        CppVariableDefinitionGenerator vdg = new CppVariableDefinitionGenerator(m_stream);
        info.AcceptChildren(vdg);
        m_stream.Write("  };\n");
        m_stream.Write('\n');
      }
    }
    
    public virtual void VisitExtended(ExtendedOptionInfo info) { }
    public virtual void VisitEnumeration(EnumerationOptionInfo info) { }
    public virtual void VisitScript(ScriptOptionInfo info) { }
    public virtual void VisitSimple(SimpleOptionInfo info) { }
    public virtual void VisitServiceInstance(ServiceInstanceOptionInfo info) { }
    
    private TextWriter m_stream;
  }
}